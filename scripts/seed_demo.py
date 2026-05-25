"""
Seed script — creates a realistic demo document and indexes it.
Run this to have something to show in your Upwork portfolio video.

Usage:
    python scripts/seed_demo.py

This creates a fake "Acme Gym" FAQ document and indexes it,
so you can immediately demo the bot answering gym-related questions.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import init_db, AsyncSessionLocal
from rag import chunk_text, index_chunks

# ─── Fake company document ────────────────────────────────────────────────────
# This simulates what a real client would upload.
# Change this to any business domain for different demos.

DEMO_DOCUMENT = """
ACME FITNESS CENTER — MEMBER FAQ & POLICIES
Last updated: January 2025

MEMBERSHIP PLANS
================

Basic Plan — $29/month
- Access to gym floor (6am–10pm)
- Locker room access
- 2 group classes per month
- No contract, cancel anytime

Pro Plan — $59/month
- Unlimited gym access (24/7)
- Unlimited group classes
- 1 personal training session per month
- Nutrition consultation (monthly)
- Guest pass (2 per month)

Elite Plan — $99/month
- All Pro benefits
- 4 personal training sessions per month
- Bi-weekly nutrition coaching
- Priority class booking
- Dedicated locker

CANCELLATION POLICY
===================
Members can cancel their membership at any time with 30 days written notice.
Cancellations must be submitted via email to cancel@acmefitness.com or in person at the front desk.
No refunds are issued for partial months. Annual contracts have a $150 early termination fee.

PERSONAL TRAINING
=================
Personal training sessions are 60 minutes each.
Sessions can be booked via the ACME app or by calling 555-0100.
Cancellations must be made at least 24 hours in advance to avoid being charged.
Late cancellations or no-shows are charged the full session fee.
Personal trainers are available Monday through Saturday, 7am–8pm.

GROUP CLASSES
=============
We offer the following group classes:
- Yoga (Mon/Wed/Fri at 7am and 6pm)
- HIIT (Tue/Thu at 6am and 7pm)
- Cycling (Mon/Wed/Sat at 8am)
- Pilates (Tue/Thu/Sat at 9am)
- Zumba (Friday at 7pm and Saturday at 10am)
- CrossFit (Mon/Wed/Fri at 5:30am and 5:30pm)

Classes must be booked in advance through the app. Walk-ins are allowed if space is available.
Class capacity is limited to 20 participants for safety reasons.

NUTRITION SERVICES
==================
Our certified nutritionists offer:
- Initial consultation (60 min): $80
- Follow-up sessions (30 min): $45
- Monthly nutrition plan: $120
- Meal prep coaching workshop (group): $35/person

Nutrition services are included in Pro and Elite memberships at no extra cost.

FACILITIES
==========
- Free weights: dumbbells 5–100 lbs, barbells, kettlebells
- Cardio equipment: 20 treadmills, 15 ellipticals, 10 rowing machines, 12 bikes
- Strength machines: full circuit of 30+ machines
- Stretching/recovery area with foam rollers and resistance bands
- Sauna and steam room (Pro and Elite members only)
- Outdoor training area (open April–October)
- Childcare: available Tue/Thu/Sat 9am–12pm, $10/hr per child

PARKING & LOCATION
==================
Free parking available in our dedicated lot (50 spaces).
We are located at 1234 Fitness Blvd, Suite 100.
Nearest subway: Central Station (5 min walk).
Bike racks available outside the main entrance.

CONTACT & HOURS
===============
Phone: 555-0100
Email: info@acmefitness.com
Emergency: 555-0911

Monday–Friday: 5:30am–11pm
Saturday: 7am–9pm
Sunday: 8am–8pm
Holidays: 8am–4pm (check app for specific holiday hours)

GUEST POLICY
============
Pro and Elite members may bring up to 2 guests per visit.
Guests must sign a waiver at the front desk and pay a $15 day pass fee.
The same guest may not visit more than 3 times per month.
Basic members can purchase day passes for guests at $20 each.
"""


async def seed():
    print("🌱 Seeding demo database...")
    await init_db()

    chunks = chunk_text(DEMO_DOCUMENT, source="acme_fitness_faq.txt", page=1)
    print(f"   Created {len(chunks)} chunks from demo document")

    async with AsyncSessionLocal() as db:
        stored = await index_chunks(db, chunks, namespace="demo")

    print(f"✅ Indexed {stored} chunks into namespace 'demo'")
    print()
    print("🤖 Try asking the bot:")
    print("   - 'How much does the Pro plan cost?'")
    print("   - 'Can I cancel my membership?'")
    print("   - 'What group classes do you offer?'")
    print("   - 'Is there parking available?'")
    print("   - 'What are the gym hours on Sunday?'")


if __name__ == "__main__":
    asyncio.run(seed())
