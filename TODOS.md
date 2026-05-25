# TODOS

## P2 — Code quality

### Migrate Pydantic config to v2 style
**What:** `src/config.py:28` uses deprecated `class Config`. Replace with `model_config = ConfigDict(env_file=".env")`.
**Why:** Removes Pydantic deprecation warning from test output. Required before Pydantic v3.
**Effort:** S (human ~10 min / CC ~2 min)
**Files:** `src/config.py`

### Migrate FastAPI startup to lifespan context manager
**What:** `src/main.py:24` uses `@app.on_event("startup")` which is deprecated.
Migrate to `@asynccontextmanager` lifespan pattern.
**Why:** Avoids deprecation warning in future FastAPI versions. Cleaner shutdown handling.
**Effort:** S (human ~10 min / CC ~5 min)
**Files:** `src/main.py`

## P3 — Post-demo

### Conversation history TTL / cleanup
**What:** `conversations` table grows forever. Add per-user row cap (e.g. keep last 50 turns)
or a periodic DELETE for rows older than 7 days.
**Why:** After days of testing, table accumulates garbage. Affects retrieval history accuracy.
**Effort:** S (human ~30 min / CC ~10 min)
**Files:** `src/rag.py` (get_history), `src/bot.py` (/clear resets only one user)

### Reuse httpx AsyncClient for LLM calls
**What:** `src/rag.py:201` creates a new `httpx.AsyncClient` per LLM call.
Lift it to module level (or inject as dependency) to reuse the connection pool.
**Why:** Saves ~50ms per call (TCP handshake + TLS negotiation). Minor for demo, meaningful under load.
**Effort:** S (human ~15 min / CC ~5 min)
**Files:** `src/rag.py`
