"""
WhatsApp Cloud API channel adapter.

Implements ChannelAdapter for Meta's WhatsApp Business Platform.
Webhook verification: GET hub.challenge + POST HMAC-SHA256 (app_secret).
Message dedup: in-memory dict checked before BackgroundTask spawn.
24h service window: DB-backed UPSERT on wa_service_windows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone

import httpx
from starlette.requests import Request
from starlette.responses import Response

from channels.protocol import (
    CHANNEL_FORMATTING,
    ChannelButton,
    ChannelMessage,
    ChannelSendError,
    format_text_for_channel,
    normalize_phone,
)

logger = logging.getLogger(__name__)

# ─── Message dedup cache (in-memory, single-worker) ───────────────────────────

_wa_message_cache: dict[str, float] = {}
WA_DEDUP_TTL = 3600  # seconds


def _is_duplicate(message_id: str) -> bool:
    """Check if a WA message ID was already processed. Thread-safe for single worker."""
    now = time.time()
    # Clean expired entries
    expired = [k for k, v in _wa_message_cache.items() if now - v > WA_DEDUP_TTL]
    for k in expired:
        del _wa_message_cache[k]
    if message_id in _wa_message_cache:
        return True
    _wa_message_cache[message_id] = now
    return False


# ─── WhatsApp adapter ──────────────────────────────────────────────────────────


class WhatsAppAdapter:
    """ChannelAdapter implementation for WhatsApp Cloud API."""

    def __init__(self, phone_number_id: str, access_token: str,
                 app_secret: str, verify_token: str,
                 business_id: str | None = None):
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.app_secret = app_secret
        self.verify_token = verify_token
        self.business_id = business_id
        self._base_url = "https://graph.facebook.com/v21.0"
        self._http = httpx.AsyncClient(timeout=30.0)

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # ─── parse_incoming ─────────────────────────────────────────────────────

    async def parse_incoming(self, webhook_data: dict) -> list[ChannelMessage]:
        """Parse WA webhook payload into ChannelMessage list.

        Returns [] for status callbacks (delivered, read, failed).
        Raises ValueError for malformed payloads.
        """
        try:
            entry = webhook_data["entry"][0]
            changes = entry.get("changes", [])
        except (KeyError, IndexError):
            raise ValueError("Invalid WhatsApp webhook payload: missing entry/changes")

        messages: list[ChannelMessage] = []
        for change in changes:
            value = change.get("value", {})
            # Status updates (delivered, read, failed) — no reply needed
            if "statuses" in value:
                continue
            wa_messages = value.get("messages", [])
            for msg in wa_messages:
                msg_type = msg.get("type", "")
                from_number = normalize_phone(msg.get("from", ""))
                msg_id = msg.get("id", "")

                # Dedup check
                if _is_duplicate(msg_id):
                    logger.info("wa_dedup_skip msg_id=%s", msg_id)
                    continue

                if msg_type == "text":
                    text_body = msg.get("text", {}).get("body", "")
                    messages.append(ChannelMessage(
                        user_id=from_number,
                        text=text_body,
                        channel="whatsapp",
                        raw=msg,
                    ))

                elif msg_type == "audio":
                    # Voice note — will be handled by download_media + transcribe
                    media_id = msg.get("audio", {}).get("id", "")
                    mime_type = msg.get("audio", {}).get("mime_type", "audio/ogg")
                    media_type = "voice" if mime_type.startswith("audio/") else "document"
                    messages.append(ChannelMessage(
                        user_id=from_number,
                        text=None,
                        media_url=media_id,  # WA uses media ID, not URL
                        media_type=media_type,
                        channel="whatsapp",
                        raw=msg,
                    ))

                elif msg_type == "image":
                    # Image message — extract media ID and caption
                    media_id = msg.get("image", {}).get("id", "")
                    caption = msg.get("image", {}).get("caption", "")
                    messages.append(ChannelMessage(
                        user_id=from_number,
                        text=caption or None,  # caption as question text
                        media_url=media_id,    # WA media ID for download
                        media_type="image",
                        channel="whatsapp",
                        raw=msg,
                    ))

                elif msg_type in ("document", "video", "sticker"):
                    # Unsupported media — will be handled with help text
                    messages.append(ChannelMessage(
                        user_id=from_number,
                        text=None,
                        media_type="document",  # distinct from "image"
                        channel="whatsapp",
                        raw=msg,
                    ))

                elif msg_type == "button":
                    # Quick-reply button callback
                    button_text = msg.get("button", {}).get("text", "")
                    button_payload = msg.get("button", {}).get("payload", "")
                    messages.append(ChannelMessage(
                        user_id=from_number,
                        text=button_text,
                        channel="whatsapp",
                        reply_to=button_payload or button_text,
                        raw=msg,
                    ))

                elif msg_type == "interactive":
                    # Interactive list/reply
                    interactive = msg.get("interactive", {})
                    list_reply = interactive.get("list_reply", {})
                    button_reply = interactive.get("button_reply", {})
                    reply_text = (list_reply or button_reply).get("title", "")
                    reply_id = (list_reply or button_reply).get("id", "")
                    messages.append(ChannelMessage(
                        user_id=from_number,
                        text=reply_text,
                        channel="whatsapp",
                        reply_to=reply_id or reply_text,
                        raw=msg,
                    ))

                else:
                    logger.warning("wa_unsupported_msg_type type=%s from=%s", msg_type, from_number)

        return messages

    # ─── download_media ────────────────────────────────────────────────────

    async def download_media(self, media_url: str) -> bytes:
        """Download media from WA Cloud API using media ID.

        WA returns a media ID, not a URL. We first retrieve the download URL
        via GET /{media_id}, then download the content.
        """
        # Step 1: Get download URL from media ID
        url_resp = await self._http.get(
            f"{self._base_url}/{media_url}",
            headers={"Authorization": f"Bearer {self.access_token}"},
        )
        url_resp.raise_for_status()
        download_url = url_resp.json().get("url")
        if not download_url:
            raise ChannelSendError("whatsapp", media_url,
                                    RuntimeError("No download URL in media response"))

        # Step 2: Download the actual media content
        content_resp = await self._http.get(
            download_url,
            headers={"Authorization": f"Bearer {self.access_token}"},
        )
        content_resp.raise_for_status()
        return content_resp.content

    # ─── send_reply ─────────────────────────────────────────────────────────

    async def send_reply(
        self,
        user_id: str,
        text: str,
        buttons: list[ChannelButton] | None = None,
    ) -> dict:
        """Send a text message via WhatsApp Cloud API.

        Max 3 quick-reply buttons per message (WA limitation).
        Raises ChannelSendError on failure.
        """
        # Post-process text for WhatsApp formatting
        text = format_text_for_channel(text, "whatsapp")

        payload: dict = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": user_id,
            "type": "text",
            "text": {"body": text},
        }

        # Quick-reply buttons (max 3)
        if buttons and len(buttons) <= 3:
            payload["type"] = "interactive"
            payload["interactive"] = {
                "type": "button",
                "body": {"text": text[:1024]},  # WA body limit
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {"id": b.callback_data or b.label, "title": b.label[:20]},
                        }
                        for b in buttons[:3]
                    ]
                },
            }
            del payload["text"]

        try:
            resp = await self._http.post(
                f"{self._base_url}/{self.phone_number_id}/messages",
                headers={"Authorization": f"Bearer {self.access_token}"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise ChannelSendError("whatsapp", user_id, e) from e

    # ─── send_chat_action ───────────────────────────────────────────────────

    async def send_chat_action(self, user_id: str) -> None:
        """Send 'typing' indicator via WhatsApp (best-effort)."""
        try:
            await self._http.post(
                f"{self._base_url}/{self.phone_number_id}/messages",
                headers={"Authorization": f"Bearer {self.access_token}"},
                json={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": user_id,
                    "type": "typing",
                },
            )
        except Exception:
            pass  # Best-effort, don't fail on typing indicator

    # ─── verify_webhook ─────────────────────────────────────────────────────

    def verify_webhook(self, request: Request, body: bytes = b"") -> bool:
        """Verify WhatsApp webhook signature using HMAC-SHA256.

        Callers should read the request body first (await request.body())
        and pass it as the `body` parameter. If `body` is empty, falls back
        to request._body (populated after body() is called).
        """
        try:
            if body:
                raw = body
            elif hasattr(request, "_body") and request._body:
                raw = request._body
            else:
                return False

            body_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not body_str:
                return False

            signature = request.headers.get("X-Hub-Signature-256", "")
            if not signature or not signature.startswith("sha256="):
                return False

            expected = hmac.new(
                self.app_secret.encode("utf-8"),
                body_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            received = signature.removeprefix("sha256=")
            return hmac.compare_digest(expected, received)
        except Exception as e:
            logger.warning("wa_webhook_verify_error: %s", e)
            return False

    # ─── handle_verification ────────────────────────────────────────────────

    async def handle_verification(self, request: Request) -> Response | None:
        """Handle WA webhook GET verification (hub.challenge).

        Returns Response if this is a verification request, None otherwise.
        """
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")

        if mode == "subscribe" and hmac.compare_digest(token or "", self.verify_token):
            logger.info("wa_webhook_verified")
            return Response(content=challenge, status_code=200)
        return None

    # ─── format_text ───────────────────────────────────────────────────────

    def format_text(self, text: str) -> str:
        """Post-process LLM output for WhatsApp display rules."""
        return format_text_for_channel(text, "whatsapp")


# ─── 24-hour service window ────────────────────────────────────────────────────

async def check_wa_service_window(
    db,  # AsyncSession
    tenant_id: int,
    user_id: str,
) -> bool:
    """Check if the 24-hour service window is still open for this user.

    Returns True if within window (can send free-form message).
    Returns False if outside window (must use template or log + skip).
    """
    from sqlalchemy import select
    from db import WaServiceWindow

    result = await db.execute(
        select(WaServiceWindow)
        .where(
            WaServiceWindow.tenant_id == tenant_id,
            WaServiceWindow.user_id == user_id,
        )
    )
    window = result.scalar_one_or_none()

    if window is None:
        return False

    elapsed = (datetime.now(timezone.utc) - window.last_user_message_at).total_seconds()
    return elapsed < 86400  # 24 hours in seconds


async def update_wa_service_window(
    db,  # AsyncSession
    tenant_id: int,
    user_id: str,
) -> None:
    """UPSERT: update last_user_message_at for the 24h window tracker."""
    from sqlalchemy import select
    from db import WaServiceWindow

    result = await db.execute(
        select(WaServiceWindow)
        .where(
            WaServiceWindow.tenant_id == tenant_id,
            WaServiceWindow.user_id == user_id,
        )
    )
    window = result.scalar_one_or_none()

    if window:
        window.last_user_message_at = datetime.now(timezone.utc)
    else:
        db.add(WaServiceWindow(
            tenant_id=tenant_id,
            user_id=user_id,
            last_user_message_at=datetime.now(timezone.utc),
        ))
    await db.commit()


async def send_wa_template(
    adapter: WhatsAppAdapter,
    user_id: str,
    template_name: str,
) -> dict | None:
    """Send a pre-approved template message outside the 24h window.

    Returns API response dict or None if template_name is empty.
    """
    if not template_name:
        logger.warning("wa_no_template user=%s — cannot send outside 24h window", user_id)
        return None

    try:
        resp = await adapter._http.post(
            f"{adapter._base_url}/{adapter.phone_number_id}/messages",
            headers={"Authorization": f"Bearer {adapter.access_token}"},
            json={
                "messaging_product": "whatsapp",
                "to": user_id,
                "type": "template",
                "template": {"name": template_name, "language": {"code": "es_AR"}},
            },
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        raise ChannelSendError("whatsapp", user_id, e) from e