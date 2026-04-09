"""CLI: infragraph-query — interactive hybrid query against db-docker."""
from __future__ import annotations

import json
import logging

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Query the infragraph knowledge graph.")
console = Console()


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    top_k: int = typer.Option(8, help="Number of doc chunks to retrieve"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Answer a question using hybrid RAG (Qdrant + Neo4j + deepseek)."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    from infragraph.retrieval.hybrid import HybridRetriever

    console.print(Panel(f"[bold]{question}[/bold]", title="Query", border_style="blue"))

    retriever = HybridRetriever()
    result = retriever.answer(question, top_k=top_k)

    console.print(Markdown(result["answer"]))
    console.print(
        f"\n[dim]Sources ({len(result['sources'])}): "
        + ", ".join(result["sources"][:5])
        + "[/dim]"
    )
    console.print(
        f"[dim]Semantic hits: {result['semantic_hits']} | "
        f"Graph entities: {result['graph_entities']}[/dim]"
    )


@app.command("search")
def search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(10),
    doc_type: str = typer.Option(None, help="markdown | code | config | text"),
) -> None:
    """Semantic document search only (no LLM call)."""
    from infragraph.retrieval.semantic import SemanticRetriever

    retriever = SemanticRetriever()
    results = retriever.search(query, top_k=top_k, doc_type=doc_type)

    table = Table("Score", "Type", "Source", "Preview", show_lines=True)
    for r in results:
        p = r.get("payload", {})
        preview = (p.get("text", "")[:120] + "…") if len(p.get("text", "")) > 120 else p.get("text", "")
        table.add_row(
            f"{r['score']:.3f}",
            p.get("doc_type", "?"),
            p.get("source_path", "?")[-50:],
            preview,
        )
    console.print(table)


@app.command("entity")
def entity(
    name: str = typer.Argument(..., help="Entity name to look up"),
    depth: int = typer.Option(2, help="Graph traversal depth"),
) -> None:
    """Look up an entity and its neighbours in Neo4j."""
    from infragraph.retrieval.graph import GraphRetriever

    depth = max(1, depth)
    gr = GraphRetriever()
    matches = gr.find(name, limit=5)

    if not matches:
        console.print(f"[red]No entity found for '{name}'[/red]")
        raise typer.Exit(1)

    for match in matches:
        node = match.get("node", {})
        console.print(Panel(
            json.dumps(node, indent=2),
            title=f"{node.get('name')} [{node.get('type', '?')}] (score={match['score']:.2f})",
            border_style="green",
        ))
        eid = node.get("id")
        if eid:
            nbrs = gr.neighbours(eid, depth=depth)
            if nbrs:
                console.print(f"  [bold]Neighbours (depth={depth}):[/bold]")
                for group in nbrs[:5]:
                    for n in group.get("nodes", []):
                        if n.get("id") != eid:
                            console.print(f"    → {n.get('name')} ({n.get('type', '?')})")


@app.command("stats")
def stats() -> None:
    """Show entity and relation counts in the knowledge graph."""
    from infragraph.retrieval.graph import GraphRetriever
    from infragraph.storage.qdrant import QdrantClient

    gr = GraphRetriever()
    s = gr.summary_stats()

    console.print("\n[bold]Neo4j — Entities[/bold]")
    t = Table("Type", "Count")
    for row in s["entities"]:
        t.add_row(row.get("type", "?"), str(row.get("count", 0)))
    console.print(t)

    console.print("\n[bold]Neo4j — Relations[/bold]")
    t2 = Table("Relation", "Count")
    for row in s["relations"]:
        t2.add_row(row.get("rel", "?"), str(row.get("count", 0)))
    console.print(t2)

    qc = QdrantClient()
    info = qc.collection_info()
    console.print(f"\n[bold]Qdrant — {info.get('points_count', 0)} vectors[/bold]")


if __name__ == "__main__":
    app()
