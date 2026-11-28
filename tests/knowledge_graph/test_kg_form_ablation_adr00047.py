"""Unit & integration tests for Controlled KG Form-Ablation & Embedder/Reranker Study (ADR-00047)."""

from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embedding import embedder
from embedding.embedder import (
    InvalidEmbeddingError,
    embed_documents,
    get_embedder_model,
    reset_embedder_cache,
    truncate_and_renormalize_mrl,
)
from embedding.reranker import (
    DEFAULT_RERANKER_MODEL,
    DeterministicMockReranker,
    LocalCrossEncoderReranker,
    RerankerUnavailableError,
    get_reranker,
    reset_reranker_cache,
)
from knowledge_graph.ablation import (
    DEFAULT_EMBEDDER_ARMS,
    GoldQueryCase,
    RetrievalMetrics,
    bootstrap_ci_95,
    compute_extraction_fidelity,
    compute_lofo_delta,
    evaluate_arm_promotion_gate,
    evaluate_form_promotion_gate,
    paired_bootstrap_delta_ci_95,
    render_ablation_artifacts,
    run_kg_form_ablation_study,
)
from knowledge_graph.chunk_store import (
    DocumentChunkRecord,
    search_chunks_hybrid_rrf,
    store_document_chunks,
)
from retrieval.temporal import PiTContext
from tools.eval_kg_form_ablation import main as eval_ablation_main


class _RecordingCursor:
    def __init__(self, executed: List[Tuple[str, Any]], rows_queue: List[List[Tuple[Any, ...]]]) -> None:
        self.executed = executed
        self._rows_queue = rows_queue

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return self._rows_queue.pop(0) if self._rows_queue else []


class _RecordingConn:
    def __init__(self, rows_queue: Optional[List[List[Tuple[Any, ...]]]] = None) -> None:
        self.executed: List[Tuple[str, Any]] = []
        self._rows_queue = rows_queue or []
        self.committed = False

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.executed, self._rows_queue)

    def commit(self) -> None:
        self.committed = True


def _write_synthetic_multiform_snapshot(snapshot_dir: Path) -> Path:
    """Build a deterministic 3-cluster operational graph + a synthetic 13F-HR institutional hub."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    # 3 tightly knit operational supply clusters (12 issuers + 1 institutional 13F-HR hub)
    clusters = [
        ["C1_A", "C1_B", "C1_C", "C1_D"],
        ["C2_A", "C2_B", "C2_C", "C2_D"],
        ["C3_A", "C3_B", "C3_C", "C3_D"],
    ]
    nodes: List[Dict[str, Any]] = []
    for grp in clusters:
        for nid in grp:
            nodes.append({"id": nid, "label": "Company", "ticker": nid, "source_hash": None, "properties_json": "{}"})
    nodes.append(
        {
            "id": "VANGUARD_HUB",
            "label": "Company",
            "ticker": "VANG",
            "source_hash": None,
            "properties_json": "{}",
        }
    )

    edges: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []

    def _add_edge(
        src: str,
        tgt: str,
        rel: str,
        form_type: str,
        chunk_id: str,
        *,
        share: float = 0.25,
        buffer_days: float = 30.0,
        provenance: str = "EXTRACTED_QUANTITATIVE",
    ) -> None:
        props = {
            "form_type": form_type,
            "valid_from": "2024-01-10",
            "share": share,
            "buffer_days": buffer_days,
            "param_provenance": provenance,
            "evidence_chunk_ids": [chunk_id],
            "source_type": "sec_filing",
        }
        edges.append(
            {
                "source_id": src,
                "target_id": tgt,
                "rel_type": rel,
                "source_hash": chunk_id,
                "properties_json": json.dumps(props),
            }
        )

    # Dense intra-cluster operational edges from 10-K + 10-Q + 8-K
    for idx, grp in enumerate(clusters, start=1):
        c10k = f"acc_10k_{idx}#item_1:c001"
        c8k = f"acc_8k_{idx}#item_8_01:c001"
        c10q = f"acc_10q_{idx}#item_7:c001"
        chunks.extend(
            [
                {
                    "chunk_id": c10k,
                    "doc_id": f"acc_10k_{idx}",
                    "source_type": "sec_filing",
                    "cik": f"000000000{idx}",
                    "section": "Item 1 - Business",
                    "char_start": 0,
                    "char_end": 120,
                    "text": f"Cluster {idx} primary wafer supplier {grp[0]} supplies silicon components to {grp[1]} and {grp[2]}.",
                    "valid_from": "2024-01-10",
                    "known_at": "2024-01-10T21:00:00+00:00",
                    "form_type": "10-K",
                    "sec_graph_layer": "DOMAIN1_OPERATIONAL",
                },
                {
                    "chunk_id": c8k,
                    "doc_id": f"acc_8k_{idx}",
                    "source_type": "sec_filing",
                    "cik": f"000000000{idx}",
                    "section": "Item 8.01 - Other Events",
                    "char_start": 0,
                    "char_end": 120,
                    "text": f"Form 8-K Exhibit 99.1 alert: facility fire halted {grp[1]} packaging output affecting {grp[3]}.",
                    "valid_from": "2024-02-15",
                    "known_at": "2024-02-15T21:00:00+00:00",
                    "form_type": "8-K",
                    "sec_graph_layer": "DOMAIN1_OPERATIONAL",
                },
                {
                    "chunk_id": c10q,
                    "doc_id": f"acc_10q_{idx}",
                    "source_type": "sec_filing",
                    "cik": f"000000000{idx}",
                    "section": "Item 7 - Management Discussion and Analysis",
                    "char_start": 0,
                    "char_end": 120,
                    "text": f"Quarterly MD&A update: {grp[2]} expanded buffer inventory to 45 days for {grp[3]} deliveries.",
                    "valid_from": "2024-05-05",
                    "known_at": "2024-05-05T21:00:00+00:00",
                    "form_type": "10-Q",
                    "sec_graph_layer": "DOMAIN1_OPERATIONAL",
                },
            ]
        )
        _add_edge(grp[0], grp[1], "SUPPLIES_TO", "10-K", c10k)
        _add_edge(grp[0], grp[2], "SUPPLIES_TO", "10-K", c10k)
        _add_edge(grp[1], grp[2], "SUPPLIES_TO", "10-K", c10k)
        _add_edge(grp[1], grp[3], "SUPPLIES_TO", "8-K", c8k)
        _add_edge(grp[2], grp[3], "SUPPLIES_TO", "10-Q", c10q)

    # Synthetic 13F-HR institutional hub connecting VANGUARD_HUB to EVERY issuer across all 3 clusters
    c13f = "acc_13f_001#holdings:c001"
    chunks.append(
        {
            "chunk_id": c13f,
            "doc_id": "acc_13f_001",
            "source_type": "sec_filing",
            "cik": "0000102909",
            "section": "Holdings",
            "char_start": 0,
            "char_end": 90,
            "text": "Quarterly 13F-HR institutional equity holdings table across passive S&P 500 constituents.",
            "valid_from": "2024-02-14",
            "known_at": "2024-02-14T21:00:00+00:00",
            "form_type": "13F-HR",
            "sec_graph_layer": "CAPITAL_OWNERSHIP",
        }
    )
    for grp in clusters:
        for nid in grp:
            _add_edge(
                "VANGUARD_HUB",
                nid,
                "INVESTED_IN",
                "13F-HR",
                c13f,
                share=1.0,
                buffer_days=0.0,
                provenance="DEFAULT_CONSERVATIVE_BOUND",
            )

    pq.write_table(pa.Table.from_pylist(nodes), snapshot_dir / "nodes.parquet")
    pq.write_table(pa.Table.from_pylist(edges), snapshot_dir / "edges.parquet")
    pq.write_table(pa.Table.from_pylist(chunks), snapshot_dir / "chunks.parquet")
    return snapshot_dir


def test_chunk_provenance_form_filtering_and_reranker_hook() -> None:
    """DocumentChunkRecord auto-derives sec_graph_layer; search_chunks_hybrid_rrf filters forms & invokes RerankerPort."""
    conn = _RecordingConn()
    rec_8k = DocumentChunkRecord(
        chunk_id="acc1#item_8_01:c001",
        doc_id="acc1",
        source_type="sec_filing",
        cik="320193",
        section="Item 8.01 - Other Events",
        char_start=0,
        char_end=45,
        text="Wafer fab fire halted assembly output.",
        valid_from="2024-03-01",
        form_type="Form 8-K",
    )
    rec_13f = DocumentChunkRecord(
        chunk_id="acc2#holdings:c001",
        doc_id="acc2",
        source_type="sec_filing",
        cik="102909",
        section="Holdings",
        char_start=0,
        char_end=35,
        text="Passive institutional holdings.",
        valid_from="2024-03-01",
        form_type="13F-HR",
    )
    inserted = store_document_chunks([rec_8k, rec_13f], conn=conn)
    assert inserted == 2
    assert conn.committed is True
    # Check canonical normalized form_type & derived sec_graph_layer in SQL params
    first_params = conn.executed[0][1]
    second_params = conn.executed[1][1]
    assert first_params[10] == "8-K"
    assert first_params[11] == "DOMAIN1_OPERATIONAL"
    assert second_params[10] == "13F-HR"
    assert second_params[11] == "CAPITAL_OWNERSHIP"

    # Verify form/layer SQL filtering and RerankerPort re-ordering in search_chunks_hybrid_rrf
    dense_rows = [
        ("c_generic", "d1", "sec_filing", "0000320193", "Item 1", "Generic annual corporate report.", "2024-01-10", None, 0.92, "10-K", "DOMAIN1_OPERATIONAL"),
        ("c_exact", "d2", "sec_filing", "0000320193", "Item 8.01", "Wafer fab fire halted assembly output.", "2024-02-10", None, 0.70, "8-K", "DOMAIN1_OPERATIONAL"),
    ]
    lexical_rows: List[Tuple[Any, ...]] = []
    search_conn = _RecordingConn(rows_queue=[dense_rows, lexical_rows])
    ctx = PiTContext(as_of=date(2024, 6, 1), mode="historical")
    reranker = DeterministicMockReranker()

    with patch("embedding.embedder.embed_query", return_value=[0.6, 0.8]):
        results = search_chunks_hybrid_rrf(
            "wafer fab fire assembly output",
            ctx=ctx,
            top_k=2,
            conn=search_conn,
            allowed_form_types=["10-K", "Form 8-K"],
            allowed_graph_layers=["DOMAIN1_OPERATIONAL"],
            reranker=reranker,
        )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c_exact"  # Cross-encoder mock promotes exact phrase match above generic dense hit
    assert "rerank_score" in results[0]
    assert any("form_type = ANY(%s)" in sql for sql, _ in search_conn.executed)
    assert any("sec_graph_layer = ANY(%s)" in sql for sql, _ in search_conn.executed)


def test_mrl_truncation_renormalization_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """MRL projects 2560-d vectors to unit-norm 1024-d, sets Ollama `dimensions`, and rejects degenerate prefixes."""
    reset_embedder_cache()
    raw_2560 = [3.0 if i < 1024 else 99.0 for i in range(2560)]
    projected = truncate_and_renormalize_mrl([raw_2560], target_dim=1024)
    assert len(projected) == 1
    assert len(projected[0]) == 1024
    norm = math.sqrt(sum(v * v for v in projected[0]))
    assert norm == pytest.approx(1.0, rel=1e-6)

    # Zero-norm prefix and undersized vector must fail fast via InvalidEmbeddingError
    with pytest.raises(InvalidEmbeddingError, match="Zero-norm MRL prefix"):
        truncate_and_renormalize_mrl([[0.0] * 1024 + [1.0] * 100], target_dim=1024)
    with pytest.raises(InvalidEmbeddingError, match="MRL truncation impossible"):
        truncate_and_renormalize_mrl([[1.0] * 512], target_dim=1024)

    # Verify Ollama payload carries `dimensions: 1024` and live CPU pin `num_gpu: 0`
    monkeypatch.delenv("YARNBALL_EMBEDDER_MOCK", raising=False)
    client = get_embedder_model(
        allow_mock=False,
        batch_gpu=False,
        model_name="qwen3-embedding:4b",
        mrl_dim=1024,
    )
    payload = client.build_request_payload(["SEC filing chunk"])
    assert payload["model"] == "qwen3-embedding:4b"
    assert payload["dimensions"] == 1024
    assert payload["options"] == {"num_gpu": 0}

    # Mock embed_documents with explicit mrl_dim emits exact unit-norm vectors of that dim
    vecs = embed_documents(["Apple relies on TSMC"], allow_mock=True, mrl_dim=256)
    assert len(vecs) == 1 and len(vecs[0]) == 256
    assert math.sqrt(sum(v * v for v in vecs[0])) == pytest.approx(1.0, rel=1e-6)
    reset_embedder_cache()


def test_reranker_port_mock_and_fail_fast_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """RerankerPort supports deterministic mock scoring and fails fast when CrossEncoder cannot load."""
    reset_reranker_cache()
    mock_rr = get_reranker(allow_mock=True)
    scores = mock_rr.score_pairs(
        "TSMC wafer fire",
        ["TSMC wafer fire halted foundry output", "Routine dividend declaration"],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]

    monkeypatch.delenv("YARNBALL_RERANKER_MOCK", raising=False)
    rr = LocalCrossEncoderReranker(model_name=DEFAULT_RERANKER_MODEL, device="cpu")
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(RerankerUnavailableError, match="Failed to load local CrossEncoder"):
            rr.score_pairs("query", ["passage"])
    reset_reranker_cache()


def test_ablation_detects_13f_hub_collapse_and_verifies_lofo_and_gates(tmp_path: Path) -> None:
    """FLAT 13F-HR ingestion collapses modularity Q and spikes degree skew; LAYERED restores operational health."""
    snap_dir = _write_synthetic_multiform_snapshot(tmp_path / "snap")
    rej_dir = tmp_path / "rejections"
    rej_dir.mkdir()
    (rej_dir / "rejections_20240601.jsonl").write_text(
        json.dumps({"task_id": "EXTRACT_SEC_GRAPH", "error_stage": "ONTOLOGY_PRUNING", "metadata": {"form_type": "13F-HR"}}) + "\n",
        encoding="utf-8",
    )

    queries = [
        GoldQueryCase("Q1", "Cluster 1 wafer supplier silicon components", ("acc_10k_1#item_1:c001",)),
        GoldQueryCase("Q2", "facility fire halted packaging output C2_B C2_D", ("acc_8k_2#item_8_01:c001",)),
        GoldQueryCase("Q3", "Quarterly MD&A buffer inventory 45 days C3_C C3_D", ("acc_10q_3#item_7:c001",)),
    ]

    scorecard = run_kg_form_ablation_study(
        snapshot_dir=snap_dir,
        as_of="2024-06-30",
        gold_queries=queries,
        rejection_dir=rej_dir,
        graph_isolations=("layered", "flat"),
        arm_ids=("E1", "E2", "E2b", "E3"),
        run_lofo=True,
        allow_mock=True,
        bootstrap_resamples=100,
        seed=42,
    )

    layered_topo = scorecard["cells"]["full__layered"]["topology"]
    flat_topo = scorecard["cells"]["full__flat"]["topology"]

    # 13F-HR hub in FLAT mode connects all 12 issuers -> collapses modularity Q and spikes max_degree & Gini
    assert flat_topo["max_degree"] == 12
    assert flat_topo["leiden_modularity_q"] < layered_topo["leiden_modularity_q"]
    assert flat_topo["degree_gini"] > layered_topo["degree_gini"]
    assert layered_topo["edge_count"] == 15  # Strictly operational edges (10-K + 8-K + 10-Q)
    assert flat_topo["edge_count"] == 27     # 15 operational + 12 13F-HR hub edges

    # Provenance grounding: operational layered edges are 100% EXTRACTED_FROM_FILING -> FILING_GROUNDED
    assert scorecard["cells"]["full__layered"]["provenance"]["filing_grounded_share"] == 1.0
    assert scorecard["cells"]["full__layered"]["provenance"]["resolved_evidence_link_share"] == 1.0

    # Exact LOFO identity: removing 10-K removes its 9 edges (3 clusters * 3 edges)
    assert scorecard["lofo_deltas"]["10-K"]["edge_count"] == 9.0
    assert scorecard["lofo_deltas"]["8-K"]["edge_count"] == 3.0
    assert scorecard["lofo_deltas"]["10-Q"]["edge_count"] == 3.0

    # Pre-registered gate checks: E2b (8B) must NEVER be eligible for live promotion
    assert DEFAULT_EMBEDDER_ARMS["E2b"].live_deployable is False
    assert scorecard["arm_promotion_verdicts"]["E2b"]["live_deployable"] is False
    assert scorecard["arm_promotion_verdicts"]["E2b"]["passed"] is False

    # Synthetic promotion gate boundary tests
    bad_form = evaluate_form_promotion_gate(
        "13F-HR",
        {"edges_per_node": 1.0, "leiden_modularity_q": -0.20, "mean_dropped_impact_share": 0.08},
        form_quarantine_rate=0.12,
    )
    assert bad_form.passed is False
    assert len(bad_form.reasons) >= 3

    good_form = evaluate_form_promotion_gate(
        "8-K",
        {"edges_per_node": 0.25, "leiden_modularity_q": 0.01, "mean_dropped_impact_share": 0.0, "recall_at_10": 0.05},
        form_quarantine_rate=0.01,
    )
    assert good_form.passed is True

    # Paired bootstrap CI sanity + ArmPromotionVerdict pass/fail
    lo, hi = paired_bootstrap_delta_ci_95([0.9, 0.95, 0.85, 0.92], [0.7, 0.72, 0.68, 0.71], n_resamples=200, seed=7)
    assert lo > 0.10 and hi > lo
    assert bootstrap_ci_95([0.8, 0.8, 0.8], n_resamples=100) == (0.8, 0.8)

    base_m = RetrievalMetrics(
        arm_id="E1", label="E1", live_deployable=True, query_count=4, chunk_count=10,
        recall_at_5=0.70, recall_at_10=0.75, mrr_at_10=0.65, ndcg_at_10=0.70,
        p50_query_latency_ms=20.0, p95_query_latency_ms=35.0, embed_throughput_chunks_per_sec=100.0,
        recall_at_10_ci95=(0.70, 0.80), ndcg_at_10_ci95=(0.65, 0.75),
    )
    promoted_m = RetrievalMetrics(
        arm_id="E3", label="E3", live_deployable=True, query_count=4, chunk_count=10,
        recall_at_5=0.80, recall_at_10=0.85, mrr_at_10=0.78, ndcg_at_10=0.82,
        p50_query_latency_ms=80.0, p95_query_latency_ms=140.0, embed_throughput_chunks_per_sec=100.0,
        recall_at_10_ci95=(0.80, 0.90), ndcg_at_10_ci95=(0.78, 0.86),
        delta_vs_e1_recall10_ci95=(0.05, 0.15), delta_vs_e1_ndcg10_ci95=(0.06, 0.18),
    )
    slow_m = RetrievalMetrics(
        arm_id="E2", label="E2", live_deployable=True, query_count=4, chunk_count=10,
        recall_at_5=0.80, recall_at_10=0.85, mrr_at_10=0.78, ndcg_at_10=0.82,
        p50_query_latency_ms=210.0, p95_query_latency_ms=320.0, embed_throughput_chunks_per_sec=25.0,
        recall_at_10_ci95=(0.80, 0.90), ndcg_at_10_ci95=(0.78, 0.86),
        delta_vs_e1_recall10_ci95=(0.05, 0.15), delta_vs_e1_ndcg10_ci95=(0.06, 0.18),
    )
    assert evaluate_arm_promotion_gate(promoted_m, base_m).passed is True
    assert evaluate_arm_promotion_gate(slow_m, base_m).passed is False


def test_cli_smoke_emits_scorecard_markdown_and_png_plots(tmp_path: Path) -> None:
    """tools/eval_kg_form_ablation.py --mock produces JSON scorecard, Markdown report, and all 3 PNG plots."""
    snap_dir = _write_synthetic_multiform_snapshot(tmp_path / "snap")
    queries_path = tmp_path / "queries_gold.jsonl"
    queries_path.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "Q1", "query": "Cluster 1 wafer supplier silicon", "relevant_chunk_ids": ["acc_10k_1#item_1:c001"]}),
                json.dumps({"query_id": "Q2", "query": "facility fire halted packaging output", "relevant_chunk_ids": ["acc_8k_2#item_8_01:c001"]}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "eval_out"
    rc = eval_ablation_main(
        [
            "--snapshot-dir", str(snap_dir),
            "--as-of", "2024-06-30",
            "--queries-jsonl", str(queries_path),
            "--output-dir", str(out_dir),
            "--arms", "E1,E2,E2b,E3",
            "--bootstrap-resamples", "50",
            "--mock",
        ]
    )
    assert rc == 0
    assert (out_dir / "ablation_scorecard.json").is_file()
    assert (out_dir / "ablation_report.md").is_file()
    assert (out_dir / "lofo_delta_heatmap.png").is_file()
    assert (out_dir / "modularity_vs_dropped_impact.png").is_file()
    assert (out_dir / "embedder_recall_vs_latency.png").is_file()

    loaded = json.loads((out_dir / "ablation_scorecard.json").read_text(encoding="utf-8"))
    assert loaded["adr_id"] == "ADR-00047"
    assert set(loaded["arm_promotion_verdicts"].keys()) == {"E1", "E2", "E2b", "E3"}
