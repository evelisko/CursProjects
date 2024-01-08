import re
from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import flags
from utils.chat_model import ChatModel
from utils.toxic_classifirer import CheckToxicity


router = Router()
chat_model = ChatModel()
check_toxicity = CheckToxicity()

@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer("Здравствуй, дорогой! Чем могу помочь тебе сегодня?")


@router.message()
@flags.chat_action("typing")
async def message_handler(msg: Message):
    print(msg.text)

    temperature = re.findall('temperature:.*?(\d\.\d\d?).*', msg.text)

    if temperature:
        print(f'temperature=: {temperature}')
        await msg.answer(chat_model.change_temperature(float(temperature[0])))
    else:    
        toxic_answer = check_toxicity.text2toxicity(msg.text)
        if not toxic_answer:
            await msg.answer(chat_model.generate(msg.text))
        else:
            await msg.answer(' '.join(toxic_answer)) 
