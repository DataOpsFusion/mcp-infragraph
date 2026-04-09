from __future__ import annotations

from types import SimpleNamespace

from infragraph.storage.qdrant import QdrantClient


def _build_client(fake_client: object) -> QdrantClient:
    client = QdrantClient.__new__(QdrantClient)
    client._client = fake_client
    client._collection = "infragraph"
    client._dim = 1536
    return client


def test_search_uses_query_points_when_available() -> None:
    class FakeClient:
        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="abc",
                        score=0.91,
                        payload={"text": "hello"},
                    )
                ]
            )

    fake = FakeClient()
    client = _build_client(fake)

    results = client.search(vector=[0.1, 0.2], top_k=3, doc_type_filter="text", score_threshold=0.4)

    assert results == [{"score": 0.91, "payload": {"text": "hello"}, "id": "abc"}]
    assert fake.kwargs["collection_name"] == "infragraph"
    assert fake.kwargs["query"] == [0.1, 0.2]
    assert fake.kwargs["limit"] == 3
    assert fake.kwargs["score_threshold"] == 0.4
    assert fake.kwargs["query_filter"] is not None


def test_search_falls_back_to_legacy_search() -> None:
    class FakeClient:
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    id="xyz",
                    score=0.73,
                    payload={"text": "legacy"},
                )
            ]

    fake = FakeClient()
    client = _build_client(fake)

    results = client.search(vector=[0.4, 0.5], top_k=2)

    assert results == [{"score": 0.73, "payload": {"text": "legacy"}, "id": "xyz"}]
    assert fake.kwargs["collection_name"] == "infragraph"
    assert fake.kwargs["query_vector"] == [0.4, 0.5]
    assert fake.kwargs["limit"] == 2
