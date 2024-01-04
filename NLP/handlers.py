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
async def start_handler(self, msg: Message):
    await msg.answer("Здравствуй, дорогой! Чем могу помочь тебе сегодня?")


@router.message()
@flags.chat_action("typing")
async def message_handler(msg: Message):
    print(msg.text)
    # Добавим возможность установки температуры генерации.

    toxic_answer = check_toxicity.text2toxicity(msg.text)
    if not toxic_answer:
        await msg.answer(chat_model.generate(msg.text))
    else:
        await msg.answer(' '.join(toxic_answer)) 
