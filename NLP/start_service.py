from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


updater = Updater(token='123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11') # Токен API к Telegram
dispatcher = updater.dispatcher


# Обработка команд
def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Добрый день')
    
def textMessage(bot, update):
    response = 'Ваше сообщение принял ' + update.message.text # формируем текст ответа
    bot.send_message(chat_id=update.message.chat_id, text=response)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hi!')

def echo(update: Update, context: CallbackContext):
    txt = update.message.text
    
    update.message.reply_text('Ваше сообщение! ' + update.message.text)


updater = Updater("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", use_context=True)
dispatcher = updater.dispatcher

# on different commands - answer in Telegram
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

# Start the Bot
updater.start_polling()
updater.idle()





