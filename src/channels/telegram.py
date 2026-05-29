"""
Telegram channel adapter — Phase 1 scaffold.

Valid implementation of the ChannelAdapter protocol but NOT in the TG hot path yet.
The TG webhook endpoint still calls telegram_app.process_update() directly.
This adapter proves the protocol is implementable for TG.
Phase 2 will move _process_question logic into a channel-agnostic service.
"""

from __future__ import annotations

import logging
from typing import Any

from telegram import Bot

from channels.protocol import (
    ChannelButton,
    ChannelMessage,
    ChannelSendError,
    format_text_for_channel,
)

logger = logging.getLogger(__name__)


class TelegramAdapter:
    """ChannelAdapter implementation for Telegram.

    Phase 1: Valid protocol implementation, not used in TG hot path.
    """

    def __init__(self, bot_token: str, webhook_secret: str):
        self.bot_token = bot_token
        self.webhook_secret = webhook_secret
        self._bot: Bot | None = None

    @property
    def bot(self) -> Bot:
        if self._bot is None:
            self._bot = Bot(token=self.bot_token)
        return self._bot

    async def parse_incoming(self, webhook_data: dict) -> list[ChannelMessage]:
        """Parse Telegram Update into ChannelMessage list.

        TG always produces a 1-element list (or empty for non-message updates).
        """
        message = webhook_data.get("message")
        if not message:
            return []

        from_user = message.get("from", {})
        user_id = str(from_user.get("id", ""))

        # Text message
        if "text" in message:
            return [ChannelMessage(
                user_id=user_id,
                text=message["text"],
                channel="telegram",
                raw=webhook_data,
            )]

        # Voice note
        if "voice" in message:
            voice = message["voice"]
            file_id = voice.get("file_id", "")
            return [ChannelMessage(
                user_id=user_id,
                text=None,
                media_url=file_id,  # TG uses file_id, resolved via getFile
                media_type="voice",
                channel="telegram",
                raw=webhook_data,
            )]

        # Other media — unsupported for now
        for media_type in ("photo", "document", "video", "sticker"):
            if media_type in message:
                return [ChannelMessage(
                    user_id=user_id,
                    text=None,
                    media_type="image",
                    channel="telegram",
                    raw=webhook_data,
                )]

        return []

    async def download_media(self, media_url: str) -> bytes:
        """Download file from Telegram using file_id."""
        import httpx
        file = await self.bot.get_file(media_url)
        async with httpx.AsyncClient() as client:
            resp = await client.get(file.file_path)
            resp.raise_for_status()
            return resp.content

    async def send_reply(
        self,
        user_id: str,
        text: str,
        buttons: list[ChannelButton] | None = None,
    ) -> dict:
        """Send reply via Telegram Bot API."""
        try:
            if buttons:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                keyboard = []
                for b in buttons[:3]:  # TG allows more, but keep consistent
                    if b.url:
                        keyboard.append([InlineKeyboardButton(text=b.label, url=b.url)])
                    elif b.callback_data:
                        keyboard.append([InlineKeyboardButton(text=b.label, callback_data=b.callback_data)])
                msg = await self.bot.send_message(
                    chat_id=int(user_id),
                    text=text,
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
                )
            else:
                msg = await self.bot.send_message(
                    chat_id=int(user_id),
                    text=text,
                    parse_mode="Markdown",
                )
            return {"message_id": msg.message_id}
        except Exception as e:
            raise ChannelSendError("telegram", user_id, e) from e

    async def send_chat_action(self, user_id: str) -> None:
        """Send 'typing' indicator via Telegram."""
        try:
            await self.bot.send_chat_action(chat_id=int(user_id), action="typing")
        except Exception:
            pass  # Best-effort

    def verify_webhook(self, request: Any) -> bool:
        """Verify TG webhook using secret token comparison."""
        from starlette.requests import Request
        if not isinstance(request, Request):
            return False
        received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        import hmac
        return hmac.compare_digest(received, self.webhook_secret)

    async def handle_verification(self, request: Any) -> None:
        """TG doesn't use GET verification — returns None always."""
        return None

    def format_text(self, text: str) -> str:
        """Telegram formatting — pass through (prompt already handles it)."""
        return format_text_for_channel(text, "telegram")