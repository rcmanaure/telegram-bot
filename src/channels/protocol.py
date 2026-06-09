"""
Channel-agnostic protocol for multi-tenant messaging.

Every channel (Telegram, WhatsApp, etc.) implements ChannelAdapter.
The channel-agnostic handler calls parse_incoming → process → format_text → send_reply
without knowing which platform produced the message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from starlette.requests import Request
from starlette.responses import Response


# ─── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ChannelButton:
    """Channel-agnostic button. Adapters translate to native format.

    - url: opens a link (WhatsApp: URL button, Telegram: inline URL button)
    - callback_data: returns payload to bot (WhatsApp: quick-reply, Telegram: callback_data)
    """

    label: str
    url: str | None = None
    callback_data: str | None = None


@dataclass
class ChannelMessage:
    """Parsed message from any channel."""

    user_id: str  # Channel-specific identifier (TG: user_id str, WA: E.164 phone)
    text: str | None = None  # Message text (None for media-only messages)
    media_url: str | None = None  # Download URL for voice/image/document
    media_type: str | None = None  # "voice" | "image" | "document" | None
    channel: str = "telegram"  # "telegram" | "whatsapp"
    media_group_id: str | None = None  # Telegram album grouping ID
    reply_to: str | None = None  # Message ID being replied to (for button callbacks)
    raw: dict = field(default_factory=dict)  # Original webhook payload


class ChannelSendError(Exception):
    """Raised when a channel adapter fails to send a message."""

    def __init__(self, channel: str, user_id: str, original_error: Exception):
        self.channel = channel
        self.user_id = user_id
        self.original_error = original_error
        super().__init__(f"[{channel}] Failed to send to {user_id}: {original_error}")


# ─── Per-channel formatting specs ──────────────────────────────────────────────

@dataclass(frozen=True)
class ChannelFormatting:
    """Per-channel formatting rules for LLM system prompts and output post-processing."""

    # System prompt instructions (inserted into _build_system_prompt)
    format_instructions: str

    # Max message length before truncation
    max_length: int

    # Whether markdown tables are supported
    supports_markdown_tables: bool

    # Bold format: TG="*text*", WA="*text*" (same, but could differ)
    bold_format: str = "*text*"

    # Italic format
    italic_format: str = "_text_"

    # Code format
    code_format: str = "`text`"


TELEGRAM_FORMATTING = ChannelFormatting(
    format_instructions="""Formato para Telegram (OBLIGATORIO):
- NUNCA uses tablas Markdown (| col | col |) — Telegram no las renderiza, se ven como texto crudo.
- Para comparar opciones o listar ítems con atributos usa listas con viñetas y negrita:
  *Plan Basic* — $29/mes · acceso 6am–10pm · 2 clases/mes
  *Plan Pro* — $59/mes · acceso 24/7 · clases ilimitadas
- Para listas simples usa guiones o números.
- Para horarios o datos tabulares usa formato vertical:
  📅 Yoga: Lun/Mié/Vie — 7am y 6pm
  📅 HIIT: Mar/Jue — 6am y 7pm
- Negrita con *asteriscos* para títulos o datos clave.
- Código con `backticks` solo para datos técnicos exactos (precios, horarios puntuales).""",
    max_length=4096,
    supports_markdown_tables=False,
)

WHATSAPP_FORMATTING = ChannelFormatting(
    format_instructions="""Formato para WhatsApp (OBLIGATORIO):
- Formato soportado: *negrita*, _cursiva_, ~tachado~, ```monoespacio```.
- NUNCA uses tablas Markdown (| col | col |) — WhatsApp no las renderiza.
- NUNCA uses `backticks sueltos` para código inline — usa ```bloque monoespacio``` para datos técnicos exactos, o texto plano.
- NUNCA uses encabezados con # — WhatsApp no los renderiza. Usa *NEGRITA* para títulos.
- Para comparar opciones usa listas con viñetas (usa - o *):
  *Plan Basic* — $29/mes · acceso 6am–10pm · 2 clases/mes
  *Plan Pro* — $59/mes · acceso 24/7 · clases ilimitadas
- Para listas simples usa guiones o números.
- Para horarios usa formato vertical:
  📅 Yoga: Lun/Mié/Vie — 7am y 6pm
  📅 HIIT: Mar/Jue — 6am y 7pm
- Usá > para citas o destacados:
  > "La mejor atención" — Cliente satisfecho
- Emojis con moderación (máx ~10 por mensaje) como señales visuales, no decoración.
- Mensajes cortos y escaneables: idealmente menos de 450 caracteres para plantillas.
- Separá secciones con una línea en blanco (máx 2 saltos consecutivos).
- Pone la información más importante al principio (la preview muestra las primeras 2 líneas).
- Negrita con *asteriscos* para títulos o datos clave.
- Cursiva con _guiones bajos_ para énfasis secundario o nombres.
- Tachado con ~virgulillas~ para precios anteriores o info obsoleta.""",
    max_length=4096,
    supports_markdown_tables=False,
    code_format="text",  # No backtick support — plain text only
)

CHANNEL_FORMATTING: dict[str, ChannelFormatting] = {
    "telegram": TELEGRAM_FORMATTING,
    "whatsapp": WHATSAPP_FORMATTING,
}


# ─── Adapter protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class ChannelAdapter(Protocol):
    """Interface every messaging channel must implement."""

    async def parse_incoming(self, webhook_data: dict) -> list[ChannelMessage]:
        """Parse raw webhook payload into ChannelMessage list.

        Returns [] for non-message events (status callbacks, verification).
        Raises ValueError for malformed payloads that cannot be parsed.
        """

    async def download_media(self, media_url: str) -> bytes:
        """Download media from channel-specific URL. Returns raw bytes."""

    async def send_reply(
        self,
        user_id: str,
        text: str,
        buttons: list[ChannelButton] | None = None,
    ) -> dict:
        """Send reply. Buttons translated to channel-native format.
        Returns API response dict. Raises ChannelSendError on failure."""

    async def send_chat_action(self, user_id: str) -> None:
        """Send 'typing' indicator (best-effort)."""

    def verify_webhook(self, request: Request) -> bool:
        """Verify webhook signature. Returns True if valid."""

    async def handle_verification(self, request: Request) -> Response | None:
        """Handle channel verification (GET for WA hub.challenge).
        Returns Response if verification request, None otherwise."""

    def format_text(self, text: str) -> str:
        """Post-process LLM output for channel-specific display rules.
        Strips or converts unsupported formatting, truncates to max_length."""


# ─── Post-processing helpers ──────────────────────────────────────────────────


def format_text_for_channel(text: str, channel: str, max_length: int | None = None) -> str:
    """Apply channel-specific post-processing to LLM output.

    Universal normalization (both channels):
      - **bold** → *bold* (LLMs often emit standard markdown bold)
      - __italic__ → _italic_ (standard markdown italic)
    WhatsApp-specific:
      - Strip markdown tables, headers, horizontal rules
      - Convert ~~strikethrough~~ → ~strikethrough~
      - Strip code block language hints and inline backticks
      - Collapse multiple blank lines
    Telegram:
      - After universal normalization, formatting is already native (*bold*, _italic_)
    """
    import re
    from channels.protocol import CHANNEL_FORMATTING

    fmt = CHANNEL_FORMATTING.get(channel, CHANNEL_FORMATTING["telegram"])

    # ── Universal markdown normalization (both channels) ──────────────────
    # Convert **bold** → *bold* (standard markdown → Telegram/WA native)
    # Match **text** but not ***text*** (bold+italic combo — rare, strip both)
    text = re.sub(r"\*\*(?!\*)(.+?)(?<!\*)\*\*", r"*\1*", text)
    # Convert __italic__ → _italic_ (standard markdown)
    # Avoid matching __ within URLs or snake_case identifiers
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"_\1_", text)

    if channel == "whatsapp":
        # Strip markdown tables: | col | col | → remove entire line
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
        # Strip markdown headers: # Title → *Title*
        text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)
        # Strip horizontal rules (---, ***, ___) → blank line
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
        # Convert markdown strikethrough ~~text~~ → WhatsApp ~text~
        text = re.sub(r"~~([^~]+)~~", r"~\1~", text)
        # Strip code block language hints: ```python → ```
        text = re.sub(r"```\w+\n", "```\n", text)
        # Strip inline backtick code: `text` → text
        # Only match single-backtick inline code (not triple-backtick blocks)
        text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", r"\1", text)
        # Collapse multiple blank lines (max 2 consecutive newlines for readability)
        text = re.sub(r"\n{3,}", "\n\n", text)

    # Truncate to channel max length
    limit = max_length or fmt.max_length
    if len(text) > limit:
        text = text[: limit - 3] + "..."

    return text


# ─── Utility functions ──────────────────────────────────────────────────────────


def normalize_phone(user_id: str) -> str:
    """Normalize a phone number for consistent DB lookups and rate limiting.

    Strips '+' prefix and 'whatsapp:' prefix, returning clean E.164 without '+'.
    Examples:
        +549111234567 → 549111234567
        whatsapp:+549111234567 → 549111234567
        549111234567 → 549111234567
    """
    phone = user_id.strip()
    # Strip 'whatsapp:' prefix (Twilio/Meta formats)
    if phone.startswith("whatsapp:"):
        phone = phone[len("whatsapp:"):]
    # Strip leading '+'
    phone = phone.lstrip("+")
    return phone