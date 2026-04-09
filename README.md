# mcp-infragraph

Infrastructure knowledge graph for the homelab. Hybrid BM25 + vector search over docs, runbooks, and operational data. Backed by Neo4j and Qdrant.

## Tools

| Tool | Description |
|------|-------------|
| `query` | Hybrid search across the knowledge graph with configurable `top_k` |
| `ingest` | Add documents or structured data to the graph |

## Usage

Used by agents to look up infrastructure context, runbooks, and previously ingested operational knowledge before taking action.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NEO4J_URI` | Neo4j bolt URI (e.g. `bolt://192.168.0.35:7687`) |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `QDRANT_URL` | Qdrant HTTP URL (e.g. `http://192.168.0.35:6333`) |

## CI/CD

Images are built on every push to `main` and pushed to:
- Harbor: `harbor.homeserverlocal.com/mcp/mcp-infragraph:latest`
- Docker Hub: `dataopsfusion/mcp-infragraph:latest`

