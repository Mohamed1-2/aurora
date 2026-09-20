"""Bounded, process-local request traces. ContextVar isolates concurrent requests."""
from collections import deque
from contextvars import ContextVar
from contextlib import contextmanager
from time import perf_counter, time
from uuid import uuid4

traces = deque(maxlen=100)
current = ContextVar("assistant_trace", default=None)

@contextmanager
def span(name, **details):
    trace = current.get()
    started = perf_counter()
    item = {"name": name, "status": "ok", **details}
    try:
        yield item
    except BaseException:
        item["status"] = "error"
        raise
    finally:
        item["duration_ms"] = round((perf_counter()-started)*1000, 3)
        if trace is not None:
            trace["stages"].append(item)

async def traced(pipeline, req):
    row = {"id": uuid4().hex, "at": time(), "question": req.question[:240], "stages": [], "status": "running"}
    token = current.set(row)
    start = perf_counter()
    try:
        response = await pipeline._ask(req)
        response.trace_id = row["id"]
        row.update(status="refused" if response.refused else "ok", provider=response.provider,
                   model=response.model, answer=response.answer[:1500], grounding=response.grounding.model_dump(),
                   release=row.get("release"))
        return response
    except BaseException as exc:
        row.update(status="error", error=type(exc).__name__)
        raise
    finally:
        row["duration_ms"] = round((perf_counter()-start)*1000, 3)
        traces.appendleft(row)
        current.reset(token)
