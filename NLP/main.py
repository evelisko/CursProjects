import os
import asyncio
from dynaconf import Dynaconf
from config_manager import Config
from aiogram import Bot, Dispatcher, Router
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.chat_action import ChatActionMiddleware
import handlers
from utils.model import Model


async def main():
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    config_path = os.path.join(file_dir, 'config', 'config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}")

    config_parser = Dynaconf(
         settings_files=[config_path],
         environments=True,
         auto_cast=True,
         env='default'
         )

    config = Config(config_parser)

    handlers.model.load('IlyaGusev/saiga2_7b_lora', is_lora=True, use_4bit=True) 
    bot = Bot(token=config.token, parse_mode=ParseMode.HTML)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)
    dp.message.middleware(ChatActionMiddleware())
    print('Bot is ready!')
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    

if __name__ == '__main__':
    asyncio.run(main())
