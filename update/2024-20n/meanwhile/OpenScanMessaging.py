#!/usr/bin/env python3
from OpenScanCommon import load_str, load_bool
import telegram
import asyncio

class Messaging:
    def __init__(self) -> None:
        self.telegram_enabled: bool = load_bool('telegram_enabled')
        print(f"Telegram enabled: {self.telegram_enabled}")
        if self.telegram_enabled:
            self.telegram_api_token: str = load_str('telegram_api_token')
            self.telegram_client_id: str = load_str('telegram_client_id')

    async def send(self, message: str):
        """
        Send a message using the Telegram API.
        """
        if not self.telegram_enabled:
            print("Telegram is not enabled.")
            return

        bot = telegram.Bot(token=self.telegram_api_token)
        try:
            await bot.send_message(chat_id=self.telegram_client_id, text=message)
            print('Message Sent!')
        except telegram.error.TelegramError as e:
            print(f"Failed to send message: {e}")

    def send_message(self, message: str):
        """
        Wrapper method to call the asynchronous send method.
        """
        asyncio.run(self.send(message))
