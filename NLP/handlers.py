from aiogram import Router
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import Command
from aiogram import flags, F
from utils.chat_model import ChatModel
from utils.smart_search import SmartSearch
from utils.toxic_classifirer import CheckToxicity


router = Router()
chat_model = ChatModel()
check_toxicity = CheckToxicity()
smart_search = SmartSearch()
use_rag = False


@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer("Здравствуй, дорогой! Чем могу помочь тебе сегодня?")


@router.message(Command("temperature"))
async def start_handler(msg: Message):
    builder = InlineKeyboardBuilder()
    for i in range(1, 11):
        key_value = str(round(i/10, 1))
        builder.add(InlineKeyboardButton(text=key_value,
                    callback_data=f'temperature_{key_value}'))
    builder.adjust(5)
    keyboard = builder.as_markup(resize_keyboard=True,
                                 one_time_keyboard=True
                                 )
    await msg.answer("Задайте температуру для генерации.", reply_markup=keyboard)


@router.callback_query(F.data.startswith("temperature_"))
async def set_temperature(callback: CallbackQuery):
    temperature = callback.data.split("_")[1]
    await callback.message.edit_text(chat_model.change_temperature(float(temperature)))


@router.message(Command("rag_mode"))
async def start_handler(msg: Message):
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text='Да',callback_data='rag_mode_on'))
    builder.add(InlineKeyboardButton(text='Нет',callback_data='rag_mode_off'))
    # builder.adjust(5)
    keyboard = builder.as_markup(resize_keyboard=True,
                                 one_time_keyboard=True
                                 )
    await msg.answer("Использовать режим Retrieval Augmented Generation (RAG)?",
                     reply_markup=keyboard)


@router.callback_query(F.data.startswith("rag_mode_"))
async def set_rag_mode(callback: CallbackQuery):
    answer = callback.data.split("_")[-1]
    global use_rag
    use_rag = True if answer == 'on' else False
    msg = f'Retrieval Augmented Generation mode: {answer.upper()}' 
    await callback.message.edit_text(msg)


@router.message()
@flags.chat_action("typing")
async def message_handler(msg: Message):
    print(msg.text)
    toxic_answer = check_toxicity.text2toxicity(msg.text)
    if not toxic_answer:
        recipe = ""
        if use_rag:        
            recipe = smart_search.find_recipes(msg.text)
        if recipe:
            await msg.answer(chat_model.generate_rag(msg.text, recipe))
        else:
            await msg.answer(chat_model.generate(msg.text))
    else:
        await msg.answer(' '.join(toxic_answer))
