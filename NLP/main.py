import os
import asyncio
import json
from aiogram import Bot, Dispatcher, Router
from aiogram.enums.parse_mode import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.chat_action import ChatActionMiddleware
import handlers


async def main():
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    config_path = os.path.join(file_dir, 'config', 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path}")

    if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf8') as f:
                config = json.load(f)
    else:
        print('File {} not exists'.format(config_path))   
    
    handlers.chat_model.load(config['llm_model'], config['system_prompt'],
                              is_lora=config['is_lora'], use_4bit=config['is_4bit'])
    handlers.chat_model.change_temperature(0.3)
    handlers.check_toxicity.load(config['classifire_model'], config['toxicity_score'], config["toxic_colors"]) 
    bot = Bot(token=config['token'], parse_mode=ParseMode.HTML)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)
    dp.message.middleware(ChatActionMiddleware())
    print('Bot is ready!')
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    

if __name__ == '__main__':
    asyncio.run(main())
