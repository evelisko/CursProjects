from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import flags
from utils.model import Model


router = Router()
model = Model()

@router.message(Command("start"))
async def start_handler(self, msg: Message):
    await msg.answer("Здравствуй, дорогой! Чем могу помочь тебе сегодня?")


@router.message()
@flags.chat_action("typing")
async def message_handler(msg: Message):
    print(msg.text)
    await msg.answer(model.generate(msg.text)) # это будет промпт для нашей модели.

