"""
TwinVEIL — Module 2, Step 4.5: Redis-backed twin-state store.

Key format is fixed by the project constraints: twin:{vehicle_id}.
State is stored as JSON with a 1-hour TTL (ex=3600).

The spec targets a real Redis (docker run -d -p 6379:6379 redis:alpine). If a
real Redis is not reachable on localhost:6379, we transparently fall back to an
in-process fakeredis so the service still runs and can be tested anywhere. The
key format, TTL and API are identical either way.
"""
import json
import redis

_REAL = redis.Redis(host="localhost", port=6379, decode_responses=True,
                    socket_connect_timeout=0.5)
try:
    _REAL.ping()
    _r = _REAL
    BACKEND = "redis://localhost:6379"
except Exception:
    import fakeredis
    _r = fakeredis.FakeRedis(decode_responses=True)
    BACKEND = "fakeredis (in-process; real Redis not reachable)"


def save_twin_state(vehicle_id: str, state: dict) -> None:
    _r.set(f"twin:{vehicle_id}", json.dumps(state), ex=3600)


def get_twin_state(vehicle_id: str):
    raw = _r.get(f"twin:{vehicle_id}")
    return json.loads(raw) if raw else None
