from aiogram import Router
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import Command
from aiogram import flags, F
from utils.chat_model import ChatModel
from utils.toxic_classifirer import CheckToxicity


router = Router()
chat_model = ChatModel()
check_toxicity = CheckToxicity()


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


@router.message()
@flags.chat_action("typing")
async def message_handler(msg: Message):
    print(msg.text)
    toxic_answer = check_toxicity.text2toxicity(msg.text)
    if not toxic_answer:
        await msg.answer(chat_model.generate(msg.text))
    else:
        await msg.answer(' '.join(toxic_answer))
