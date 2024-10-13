#!/usr/bin/env python3
from OpenScanCommon import load_str, load_bool
import telegram
import asyncio


class Messaging:
    def __init__(self) -> None:
        self.telegram_enable: bool = load_bool('telegram_enable')
        if self.telegram_enable:
            self.telegram_api_token: str = load_str('telegram_api_token')
            self.telegram_client_id: str = load_str('telegram_client_id')

    async def send(self, message: str, chat_id: str, token: str):
        """
        Send a message "msg" to a telegram user or group specified by "chat_id"
        msg         [str]: Text of the message to be sent. Max 4096 characters after entities parsing.
        chat_id [int/str]: Unique identifier for the target chat or username of the target channel (in the format @channelusername)
        token       [str]: Bot's unique authentication token.
        """
        chat_id = self.telegram_client_id
        token = self.telegram_client_id
        bot = telegram.Bot(token=token)
        try:
            await bot.sendMessage(chat_id=chat_id, text=message)
            
        except telegram.error.TelegramError as e:
            print(f"Failed to send message: {e}")
        print('Message Sent!')

    def send_sessage(self, message):
        """
        This method is used to send a message using the Telegram API. It takes a message, chat_id, and token as parameters.
        The message is the text of the message to be sent, the chat_id is the unique identifier for the target chat or username of the target channel, and the token is the bot's unique authentication token.
        The method first checks if the Telegram API is enabled. If it is, it creates a new instance of the Telegram bot using the token.
        It then tries to send the message using the bot. If an error occurs, it prints a message to the console.
        Finally, it prints a message to the console indicating that the message has been sent.
        """
        asyncio.run(
            self.send(
                message=message,
                chat_id=self.telegram_client_id,
                token=self.telegram_api_token
            )
        )

