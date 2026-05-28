"""
One-time script to seed the first tenant.
Run inside the api container:
    docker compose exec api python scripts/create_tenant.py

Or locally (with DATABASE_URL pointing to a reachable postgres):
    python scripts/create_tenant.py
"""
import asyncio
import hashlib
import os
import secrets
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def main():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    # Accept both asyncpg and plain postgres URLs
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # Read bot token — still in .env even though removed from Settings
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        bot_token = input("Enter TELEGRAM_BOT_TOKEN: ").strip()

    slug = input("Enter tenant slug (e.g. 'mi-empresa'): ").strip() or "demo"
    expertise_area = input(
        "Enter expertise area (e.g. 'membership plans, group classes, personal training for ACME Gym'): "
    ).strip()

    # Generate credentials
    raw_api_key = secrets.token_urlsafe(32)
    api_key_hash = hashlib.sha256(raw_api_key.encode()).hexdigest()
    webhook_secret = secrets.token_urlsafe(32)

    engine = create_async_engine(database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    from db import Base, Tenant

    async with async_session() as db:
        # Check if slug already exists
        result = await db.execute(select(Tenant).where(Tenant.slug == slug))
        existing = result.scalar_one_or_none()
        if existing:
            print(f"\nTenant '{slug}' already exists (id={existing.id}). Aborting.")
            await engine.dispose()
            return

        tenant = Tenant(
            slug=slug,
            api_key_hash=api_key_hash,
            webhook_secret=webhook_secret,
            bot_token=bot_token,
            expertise_area=expertise_area or None,
            plan="free",
            active=True,
        )
        db.add(tenant)
        await db.commit()
        await db.refresh(tenant)

    await engine.dispose()

    print("\n" + "=" * 60)
    print(f"Tenant created: id={tenant.id}, slug='{slug}'")
    print(f"\nAPI Key (save this — shown only once):")
    print(f"  {raw_api_key}")
    print(f"\nWebhook secret (stored in DB, used by Telegram):")
    print(f"  {webhook_secret}")
    print("\nNext steps:")
    print(f"  1. Set APP_DOMAIN in .env to your public HTTPS URL (e.g. ngrok)")
    print(f"  2. Restart the API: docker compose restart api")
    print(f"  3. Upload a PDF: curl -X POST https://YOUR_DOMAIN/upload \\")
    print(f"       -H 'X-API-Key: {raw_api_key}' \\")
    print(f"       -F 'file=@your_doc.pdf'")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
