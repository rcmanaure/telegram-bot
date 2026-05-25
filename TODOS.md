# TODOS

## P3 — Post-demo

### Conversation history TTL / cleanup
**What:** `conversations` table grows forever. Add per-user row cap (e.g. keep last 50 turns)
or a periodic DELETE for rows older than 7 days.
**Why:** After days of testing, table accumulates garbage. Affects retrieval history accuracy.
**Effort:** S (human ~30 min / CC ~10 min)
**Files:** `src/rag.py` (get_history), `src/bot.py` (/clear resets only one user)
