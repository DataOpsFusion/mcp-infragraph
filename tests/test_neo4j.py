from __future__ import annotations

from infragraph.storage.neo4j import Neo4jClient


class FakeNode(dict):
    def __init__(self, *args, labels: set[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.labels = labels


def test_node_to_dict_infers_type_from_labels() -> None:
    node = FakeNode({"id": "svc::grafana", "name": "grafana"}, labels={"Service"})

    payload = Neo4jClient._node_to_dict(node)

    assert payload["type"] == "Service"
    assert payload["name"] == "grafana"


def test_node_to_dict_preserves_existing_type() -> None:
    node = FakeNode(
        {"id": "svc::grafana", "name": "grafana", "type": "Service"},
        labels={"Service"},
    )

    payload = Neo4jClient._node_to_dict(node)

    assert payload["type"] == "Service"
