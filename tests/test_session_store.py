from __future__ import annotations

from infragraph.session_store import SessionStore


class _FakeRedis:
    def __init__(self) -> None:
        self._kv: dict[str, object] = {}
        self._streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self._sets: dict[str, set[str]] = {}
        self._counters: dict[str, int] = {}
        self._stream_seq = 0

    def incr(self, key: str) -> int:
        value = self._counters.get(key, 0) + 1
        self._counters[key] = value
        return value

    def xadd(self, key: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = False) -> str:
        del maxlen, approximate
        self._stream_seq += 1
        stream_id = "{0}-0".format(self._stream_seq)
        self._streams.setdefault(key, []).append((stream_id, dict(fields)))
        return stream_id

    def xrange(self, key: str, min: str = "-", max: str = "+", count: int | None = None):
        rows = list(self._streams.get(key, []))
        if min != "-":
            rows = [row for row in rows if row[0] >= min]
        if max != "+":
            rows = [row for row in rows if row[0] <= max]
        if count is not None:
            rows = rows[:count]
        return rows

    def xrevrange(self, key: str, max: str = "+", min: str = "-", count: int | None = None):
        rows = list(reversed(self._streams.get(key, [])))
        if max != "+":
            rows = [row for row in rows if row[0] <= max]
        if min != "-":
            rows = [row for row in rows if row[0] >= min]
        if count is not None:
            rows = rows[:count]
        return rows

    def set(self, key: str, value: object) -> None:
        self._kv[key] = value

    def get(self, key: str):
        return self._kv.get(key)

    def sadd(self, key: str, *values: str) -> None:
        self._sets.setdefault(key, set()).update(values)

    def smembers(self, key: str):
        return self._sets.get(key, set())


def test_append_event_and_list_events_preserve_order() -> None:
    store = SessionStore(redis_client=_FakeRedis())

    first = store.append_event(session_id="sess-1", event_type="progress", summary="worker started")
    second = store.append_event(
        session_id="sess-1",
        event_type="progress",
        summary="worker finished",
        body={"ok": True},
    )

    events = store.list_events("sess-1")

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert [event["summary"] for event in events] == ["worker started", "worker finished"]
    assert events[1]["body"] == {"ok": True}


def test_checkpoint_and_artifact_round_trip() -> None:
    store = SessionStore(redis_client=_FakeRedis())

    checkpoint = store.put_checkpoint(session_id="sess-1", content="resume from here", checkpoint_id="ckpt-1")
    artifact = store.put_artifact(
        session_id="sess-1",
        artifact_id="patch.diff",
        content="diff --git",
        kind="patch",
    )

    assert store.get_checkpoint(session_id="sess-1", checkpoint_id="ckpt-1") == checkpoint
    assert store.get_checkpoint(session_id="sess-1") == checkpoint
    assert store.get_artifact(session_id="sess-1", artifact_id="patch.diff") == artifact
    assert store.list_artifact_refs("sess-1") == ["patch.diff"]


def test_resume_packet_includes_latest_checkpoint_events_and_artifacts() -> None:
    store = SessionStore(redis_client=_FakeRedis())

    store.append_event(session_id="sess-1", event_type="progress", summary="started")
    store.put_checkpoint(session_id="sess-1", content="checkpoint body", checkpoint_id="ckpt-1")
    store.put_artifact(session_id="sess-1", artifact_id="summary.txt", content="done")
    store.append_event(session_id="sess-1", event_type="done", summary="finished")

    packet = store.get_resume_packet(session_id="sess-1", event_limit=10)

    assert packet["session_id"] == "sess-1"
    assert packet["checkpoint"]["checkpoint_id"] == "ckpt-1"
    assert [event["type"] for event in packet["recent_events"]] == ["progress", "done"]
    assert packet["artifact_refs"] == ["summary.txt"]
