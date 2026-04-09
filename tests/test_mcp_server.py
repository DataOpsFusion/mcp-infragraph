from __future__ import annotations

import json

import pytest

from infragraph.mcp import server


@pytest.mark.asyncio
async def test_list_tools_upload_and_session_tools() -> None:
    tools = await server.list_tools()
    names = {tool.name for tool in tools}

    assert "ocr_extract_upload" in names
    assert "ocr_ingest_upload" in names
    assert "ingest_file_upload" in names
    assert "session_event_append" in names
    assert "session_checkpoint_put" in names
    assert "session_checkpoint_get" in names
    assert "session_artifact_put" in names
    assert "session_artifact_get" in names
    assert "session_resume_packet" in names
    assert "ingest_path" not in names
    assert "ocr_ingest_folder" not in names


def test_decode_uploaded_file_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError):
        server._decode_uploaded_file("test.pdf", "not-base64")


def test_decode_uploaded_text_file_rejects_ocr_type() -> None:
    with pytest.raises(ValueError):
        server._decode_uploaded_text_file("scan.pdf", b"fake")


class _FakeSessionStore:
    def append_event(self, **kwargs) -> dict:
        return {"kind": "session_event", **kwargs, "stream_id": "1-0"}

    def put_checkpoint(self, **kwargs) -> dict:
        return {"kind": "session_checkpoint", **kwargs}

    def get_checkpoint(self, **kwargs) -> dict | None:
        if kwargs.get("session_id") == "missing":
            return None
        return {"kind": "session_checkpoint", **kwargs, "checkpoint_id": kwargs.get("checkpoint_id") or "ckpt-1"}

    def put_artifact(self, **kwargs) -> dict:
        return {"kind": "session_artifact", **kwargs}

    def get_artifact(self, **kwargs) -> dict | None:
        if kwargs.get("artifact_id") == "missing":
            return None
        return {"kind": "session_artifact", **kwargs}

    def get_resume_packet(self, **kwargs) -> dict:
        return {
            "session_id": kwargs["session_id"],
            "checkpoint": {"checkpoint_id": "ckpt-1"},
            "recent_events": [{"event_id": "evt-1"}],
            "artifact_refs": ["artifact-1"],
        }


@pytest.mark.asyncio
async def test_session_tools_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_get_session_store", lambda: _FakeSessionStore())

    event = await server.call_tool(
        "session_event_append",
        {"session_id": "s-1", "event_type": "progress", "summary": "hello"},
    )
    checkpoint = await server.call_tool(
        "session_checkpoint_put",
        {"session_id": "s-1", "content": "state", "checkpoint_id": "ckpt-1"},
    )
    checkpoint_get = await server.call_tool(
        "session_checkpoint_get",
        {"session_id": "s-1", "checkpoint_id": "ckpt-1"},
    )
    artifact = await server.call_tool(
        "session_artifact_put",
        {"session_id": "s-1", "artifact_id": "patch.diff", "content": "diff"},
    )
    artifact_get = await server.call_tool(
        "session_artifact_get",
        {"session_id": "s-1", "artifact_id": "patch.diff"},
    )
    resume = await server.call_tool(
        "session_resume_packet",
        {"session_id": "s-1", "event_limit": 5},
    )

    assert json.loads(event[0].text)["kind"] == "session_event"
    assert json.loads(event[0].text)["stream_id"] == "1-0"
    assert json.loads(checkpoint[0].text)["checkpoint_id"] == "ckpt-1"
    assert json.loads(checkpoint_get[0].text)["kind"] == "session_checkpoint"
    assert json.loads(artifact[0].text)["artifact_id"] == "patch.diff"
    assert json.loads(artifact_get[0].text)["kind"] == "session_artifact"
    assert json.loads(resume[0].text)["session_id"] == "s-1"
    assert json.loads(resume[0].text)["artifact_refs"] == ["artifact-1"]


@pytest.mark.asyncio
async def test_session_tool_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_get_session_store", lambda: _FakeSessionStore())

    missing_session = await server.call_tool("session_checkpoint_get", {})
    missing_summary = await server.call_tool(
        "session_event_append",
        {"session_id": "s-1", "event_type": "progress", "summary": "   "},
    )
    missing_content = await server.call_tool(
        "session_artifact_put",
        {"session_id": "s-1", "artifact_id": "patch.diff", "content": "   "},
    )
    missing_checkpoint = await server.call_tool(
        "session_checkpoint_get",
        {"session_id": "missing"},
    )

    assert missing_session[0].text == "session_checkpoint_get: 'session_id' must not be empty."
    assert missing_summary[0].text == "session_event_append: 'summary' must not be empty."
    assert missing_content[0].text == "session_artifact_put: 'content' must not be empty."
    assert missing_checkpoint[0].text == "null"
