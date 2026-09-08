# rag/app.py
# -*- coding: utf-8 -*-
"""YarnBall Conversational UI & Interactive Subgraph Visualizer.

Chainlit-powered financial intelligence chat interface providing:
1. Multi-turn conversational chat with streaming grounded responses.
2. Collapsible Thought & Step-Tracing accordions for Cypher, latency, and news provenance.
3. Interactive draggable PyVis force-directed graph visualizer embedded in the stream.
4. Snapshot hot-swapping admin controls (/snapshot list, /snapshot load, /eval run).
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Set
from pathlib import Path

import chainlit as cl
from pyvis.network import Network

from graph.snapshot_manager import SnapshotManager
from graph.memgraph_driver import get_memgraph_driver
from rag.text_to_cql import TextToCQL
from rag.hybrid_retriever import HybridRetriever, GroundedSynthesizer, HybridGraphRAGEngine
from rag.eval_harness import EvaluationHarness

logger = logging.getLogger(__name__)

# Node color palette for PyVis graph visualization
NODE_COLORS: Dict[str, str] = {
    "Company": "#3b82f6",     # Blue
    "Person": "#10b981",      # Emerald Green
    "Product": "#f59e0b",     # Amber Orange
    "Technology": "#8b5cf6",  # Purple
    "Sector": "#ec4899",      # Pink
    "Regulation": "#ef4444",  # Crimson Red
    "Location": "#06b6d4",    # Cyan
    "News": "#6b7280",        # Gray
    "Entity": "#3b82f6",      # Default Blue
}


def generate_subgraph_html(
    triples: List[Dict[str, Any]],
    height: str = "400px",
    width: str = "100%",
) -> str:
    """Generate a self-contained PyVis interactive HTML force-directed network graph."""
    if not triples:
        return "<p style='color: #9ca3af; font-style: italic;'>No subgraph connections discovered for this query.</p>"

    net = Network(
        height=height,
        width=width,
        bgcolor="#111827",
        font_color="#f3f4f6",
        directed=True,
    )
    net.toggle_physics(True)

    added_nodes: Set[str] = set()

    for t in triples:
        src = str(t.get("source", "")).strip()
        tgt = str(t.get("target", "")).strip()
        rel = str(t.get("relation", "RELATED_TO")).strip()
        props = t.get("properties", {}) or {}

        if not src or not tgt:
            continue

        # Add Source Node
        if src not in added_nodes:
            color = NODE_COLORS.get("Company", "#3b82f6")
            net.add_node(
                src,
                label=src,
                title=f"Entity: {src}",
                color=color,
                size=22,
                font={"color": "#f9fafb", "size": 14, "face": "Inter, system-ui"},
            )
            added_nodes.add(src)

        # Add Target Node
        if tgt not in added_nodes:
            color = NODE_COLORS.get("Company", "#3b82f6")
            net.add_node(
                tgt,
                label=tgt,
                title=f"Entity: {tgt}",
                color=color,
                size=20,
                font={"color": "#f9fafb", "size": 14, "face": "Inter, system-ui"},
            )
            added_nodes.add(tgt)

        # Add Relationship Edge
        edge_title = f"Relation: {rel}"
        if props.get("context"):
            edge_title += f"\nContext: {props['context']}"
        if props.get("date"):
            edge_title += f"\nDate: {props['date']}"

        net.add_edge(
            src,
            tgt,
            label=rel,
            title=edge_title,
            color="#9ca3af",
            font={"color": "#9ca3af", "size": 10, "align": "middle"},
            arrows="to",
        )

    # Configure physics layout options
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)

    return net.generate_html()


# ----------------------------------------------------------------------
# Chainlit Lifecycle Handlers
# ----------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    """Initialize the YarnBall GraphRAG engine and send welcome greeting."""
    snapshot_mgr = SnapshotManager()
    text_to_cql = TextToCQL()
    retriever = HybridRetriever(text_to_cql_engine=text_to_cql)
    synthesizer = GroundedSynthesizer()
    engine = HybridGraphRAGEngine(retriever=retriever, synthesizer=synthesizer)
    eval_harness = EvaluationHarness(snapshot_manager=snapshot_mgr)

    # Store engines in user session
    cl.user_session.set("engine", engine)
    cl.user_session.set("snapshot_mgr", snapshot_mgr)
    cl.user_session.set("eval_harness", eval_harness)
    cl.user_session.set("active_snapshot", "G_resolved")

    # Fetch active snapshot details
    active_snap = snapshot_mgr.get_snapshot("G_resolved") or {}
    node_cnt = active_snap.get("node_count", 0)
    edge_cnt = active_snap.get("edge_count", 0)

    welcome_content = f"""# YarnBall Financial Intelligence 🧶

**Untangling financial tall tales into grounded, multi-hop knowledge graphs.**

- **Active Graph State:** `G_resolved` ({node_cnt} entities, {edge_cnt} relationships)
- **Model:** Ollama `qwen3:8b` (Local, Read-Only Guarded)
- **Retriever:** Dense `pgvector` + Memgraph Multi-Hop Expansion

### Quick Start Questions:
- *"Which semiconductor manufacturers supply AI chips or foundry capacity to Apple and Microsoft?"*
- *"What AI startups share investment backing from Microsoft and Amazon?"*
- *"Which companies compete with NVIDIA in data center accelerators?"*

*Type `/help` to view snapshot hot-swapping and benchmark commands.*"""

    await cl.Message(content=welcome_content).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages, process slash commands, and execute hybrid GraphRAG."""
    user_input = message.content.strip()

    engine: HybridGraphRAGEngine = cl.user_session.get("engine")
    snapshot_mgr: SnapshotManager = cl.user_session.get("snapshot_mgr")
    eval_harness: EvaluationHarness = cl.user_session.get("eval_harness")
    active_snapshot: str = cl.user_session.get("active_snapshot", "G_resolved")

    # ------------------------------------------------------------------
    # 1. Slash Commands Handler
    # ------------------------------------------------------------------
    if user_input.startswith("/"):
        tokens = user_input.split()
        cmd = tokens[0].lower()

        if cmd == "/help":
            help_text = """### YarnBall Admin Commands:
- `/snapshot list` — Display all registered graph snapshots and metrics.
- `/snapshot load <snapshot_id>` — Hot-swap Memgraph to a saved graph state (e.g., `G_raw` or `G_resolved`).
- `/eval run` — Run the 25 golden multi-hop benchmark harness and generate the A/B scorecard.
- `/help` — Display this command reference."""
            await cl.Message(content=help_text).send()
            return

        elif cmd == "/snapshot":
            subcmd = tokens[1].lower() if len(tokens) > 1 else "list"

            if subcmd == "list":
                snapshots = snapshot_mgr.list_snapshots()
                if not snapshots:
                    await cl.Message(content="No snapshots currently registered in catalog.").send()
                    return

                table_lines = [
                    "| Snapshot ID | Tag | Nodes | Edges | Created At |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
                for s in snapshots:
                    table_lines.append(
                        f"| `{s['snapshot_id']}` | {s['tag']} | {s['node_count']} | {s['edge_count']} | {s['created_at'][:19]} |"
                    )
                await cl.Message(content="\n".join(table_lines)).send()
                return

            elif subcmd == "load":
                if len(tokens) < 3:
                    await cl.Message(content="Usage: `/snapshot load <snapshot_id>` (e.g., `/snapshot load G_raw`)").send()
                    return
                target_id = tokens[2]
                try:
                    res = snapshot_mgr.restore_snapshot(target_id)
                    cl.user_session.set("active_snapshot", target_id)
                    await cl.Message(
                        content=f"Successfully hot-swapped active Memgraph graph to **`{target_id}`** ({res['node_count']} nodes, {res['edge_count']} edges)."
                    ).send()
                except Exception as exc:
                    await cl.Message(content=f"Error restoring snapshot `{target_id}`: {exc}").send()
                return

        elif cmd == "/eval" and len(tokens) > 1 and tokens[1].lower() == "run":
            async with cl.Step(name="Running 25 Multi-Hop Benchmark Harness...") as step:
                step.output = "Executing queries across G_raw and G_resolved..."
                ab_summary = eval_harness.run_ab_benchmark(
                    raw_snapshot_id="G_raw",
                    resolved_snapshot_id="G_resolved",
                    dry_run=False,
                )
                report_md = eval_harness.generate_markdown_report(ab_summary)
            await cl.Message(content=report_md).send()
            return

    # ------------------------------------------------------------------
    # 2. Hybrid GraphRAG Execution Pipeline
    # ------------------------------------------------------------------

    # Step 1: Guarded Text-to-CQL
    async with cl.Step(name="Guarded Cypher Query Execution") as cql_step:
        cql_res = engine.retriever.text_to_cql.execute_query(user_input)
        cypher = cql_res.get("cypher", "None generated")
        latency = cql_res.get("latency_ms", 0.0)
        cql_step.output = f"```cypher\n{cypher}\n```\n*Execution Latency: {latency}ms | Status: {cql_res.get('status')}*"

    # Step 2: Dense & Subgraph Neighborhood Expansion
    async with cl.Step(name="Dense Subgraph Expansion") as dense_step:
        context = engine.retriever.retrieve(user_input)
        triples = context.get("subgraph_triples", [])
        dense_entities = context.get("dense_entities", [])

        dense_lines = [f"- Seed Entities: {', '.join([e['node_id'] for e in dense_entities[:5]]) or 'None'}"]
        for t in triples[:5]:
            dense_lines.append(f"- ({t['source']}) -[{t['relation']}]-> ({t['target']})")
        dense_step.output = "\n".join(dense_lines) if dense_lines else "No neighborhood expansion required."

    # Step 3: News Article Provenance
    articles = context.get("articles", [])
    if articles:
        async with cl.Step(name="News Article Provenance") as news_step:
            art_lines = []
            for a in articles:
                art_lines.append(f"- [{a['title']}]({a['url']}) ({a.get('published_at', 'Recent')})")
            news_step.output = "\n".join(art_lines)

    # Step 4: Interactive PyVis Visualizer
    if triples:
        pyvis_html = generate_subgraph_html(triples)
        await cl.Message(
            content="### Retrieved Subgraph:",
            elements=[
                cl.Html(
                    name="subgraph_viewer",
                    content=f"<div style='border-radius: 8px; overflow: hidden; border: 1px solid #374151;'>{pyvis_html}</div>",
                    display="inline",
                )
            ],
        ).send()

    # Step 5: Grounded Answer Synthesis
    synthesis = engine.synthesizer.synthesize(user_input, context)
    answer = synthesis.get("answer", "")
    citations = synthesis.get("citations", [])

    final_msg = cl.Message(content=answer)
    await final_msg.send()


if __name__ == "__main__":
    # CLI test execution runner
    print("Run `chainlit run rag/app.py -w` to start the YarnBall UI.")
