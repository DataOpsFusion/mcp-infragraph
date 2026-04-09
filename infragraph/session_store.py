from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis as redis_lib

from infragraph.config import get_settings


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SessionStore:
    def __init__(self, redis_client: Any | None = None, prefix: str = "infragraph:sessions") -> None:
        self._prefix = prefix.rstrip(":")
        if redis_client is not None:
            self._redis = redis_client
            return

        cfg = get_settings()
        self._redis = redis_lib.from_url(
            cfg.redis_url,
            db=cfg.context_cache_db,
            decode_responses=True,
        )

    def _session_prefix(self, session_id: str) -> str:
        return "{0}:{1}".format(self._prefix, session_id)

    def _seq_key(self, session_id: str) -> str:
        return "{0}:seq".format(self._session_prefix(session_id))

    def _event_stream_key(self, session_id: str) -> str:
        return "{0}:events".format(self._session_prefix(session_id))

    def _checkpoint_key(self, session_id: str, checkpoint_id: str) -> str:
        return "{0}:checkpoint:{1}".format(self._session_prefix(session_id), checkpoint_id)

    def _checkpoint_latest_key(self, session_id: str) -> str:
        return "{0}:checkpoint:latest".format(self._session_prefix(session_id))

    def _artifact_key(self, session_id: str, artifact_id: str) -> str:
        return "{0}:artifact:{1}".format(self._session_prefix(session_id), artifact_id)

    def _artifact_index_key(self, session_id: str) -> str:
        return "{0}:artifacts".format(self._session_prefix(session_id))

    @staticmethod
    def _encode_stream_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value)

    @staticmethod
    def _decode_stream_value(value: str) -> Any:
        if value == "":
            return ""
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        if value.isdigit():
            try:
                return int(value)
            except ValueError:
                return value
        return value

    def append_event(
        self,
        *,
        session_id: str,
        event_type: str,
        summary: str,
        body: dict[str, Any] | list[Any] | str | None = None,
        job_id: str = "",
        worker_id: str = "",
        reply_to: str = "",
        checkpoint_ref: str = "",
        artifact_ref: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        seq = int(self._redis.incr(self._seq_key(session_id)))
        payload = {
            "event_id": event_id or "evt_{0}".format(uuid.uuid4().hex),
            "session_id": session_id,
            "job_id": job_id,
            "worker_id": worker_id,
            "seq": seq,
            "ts": timestamp or _utc_now(),
            "type": event_type,
            "summary": summary,
            "body": body if body is not None else {},
            "reply_to": reply_to,
            "checkpoint_ref": checkpoint_ref,
            "artifact_ref": artifact_ref,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        stream_id = self._redis.xadd(
            self._event_stream_key(session_id),
            {key: self._encode_stream_value(value) for key, value in payload.items()},
            maxlen=10000,
            approximate=True,
        )
        payload["stream_id"] = stream_id
        return payload

    def list_events(self, session_id: str, *, after: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        key = self._event_stream_key(session_id)
        if after:
            rows = self._redis.xrange(key, min=after, max="+", count=limit)
        else:
            rows = self._redis.xrevrange(key, max="+", min="-", count=limit)
            rows = list(reversed(rows))

        events: list[dict[str, Any]] = []
        for stream_id, fields in rows:
            event = {name: self._decode_stream_value(value) for name, value in fields.items()}
            event["stream_id"] = stream_id
            events.append(event)
        return events

    def put_checkpoint(
        self,
        *,
        session_id: str,
        content: str,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_event_id: str = "",
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "checkpoint_id": checkpoint_id or "ckpt_{0}".format(uuid.uuid4().hex),
            "session_id": session_id,
            "ts": timestamp or _utc_now(),
            "content": content,
            "metadata": metadata or {},
            "source_event_id": source_event_id,
        }
        self._redis.set(
            self._checkpoint_key(session_id, record["checkpoint_id"]),
            json.dumps(record, sort_keys=True),
        )
        self._redis.set(self._checkpoint_latest_key(session_id), record["checkpoint_id"])
        return record

    def get_checkpoint(self, *, session_id: str, checkpoint_id: str | None = None) -> dict[str, Any] | None:
        target_id = checkpoint_id or self._redis.get(self._checkpoint_latest_key(session_id))
        if not target_id:
            return None
        raw = self._redis.get(self._checkpoint_key(session_id, target_id))
        if not raw:
            return None
        return json.loads(raw)

    def put_artifact(
        self,
        *,
        session_id: str,
        artifact_id: str,
        content: str,
        kind: str = "text",
        metadata: dict[str, Any] | None = None,
        source_event_id: str = "",
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "artifact_id": artifact_id,
            "session_id": session_id,
            "kind": kind,
            "ts": timestamp or _utc_now(),
            "content": content,
            "metadata": metadata or {},
            "source_event_id": source_event_id,
        }
        self._redis.set(
            self._artifact_key(session_id, artifact_id),
            json.dumps(record, sort_keys=True),
        )
        self._redis.sadd(self._artifact_index_key(session_id), artifact_id)
        return record

    def get_artifact(self, *, session_id: str, artifact_id: str) -> dict[str, Any] | None:
        raw = self._redis.get(self._artifact_key(session_id, artifact_id))
        if not raw:
            return None
        return json.loads(raw)

    def list_artifact_refs(self, session_id: str) -> list[str]:
        refs = self._redis.smembers(self._artifact_index_key(session_id))
        return sorted(refs)

    def get_resume_packet(self, *, session_id: str, event_limit: int = 50) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "checkpoint": self.get_checkpoint(session_id=session_id),
            "recent_events": self.list_events(session_id, limit=event_limit),
            "artifact_refs": self.list_artifact_refs(session_id),
        }

    def close(self) -> None:
        close = getattr(self._redis, "close", None)
        if callable(close):
            close()
