"""ADR-00035 Text-Unit Chunk Layer, Qwen3-Embedding-0.6B, Targeted Gleaning & Evidence Invariant Tests."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq
import pytest

import embedding.embedder as embedder
from ingestion.sec.harvester import EdgarWorker
from ingestion.sec.parser import (
    extract_sec_relationships,
    rank_salient_chunks,
    should_glean_chunk,
)
from ingestion.segmentation.chunker import FilingMetadata, FinancialSectionSegmenter
from knowledge_graph.chunk_store import (
    DocumentChunkRecord,
    fetch_chunks_by_ids,
    make_chunk_id,
    search_chunks_hybrid_rrf,
    store_document_chunks,
)
from knowledge_graph.graph_store import (
    ExtractionGraphPayload,
    merge_chunk_extraction_payloads,
    store_graph_entities,
)
from knowledge_graph.snapshot_manager import SnapshotManager
from retrieval.pipeline import HopOrderedSubgraphVerbalizer
from retrieval.temporal import PiTContext
from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient
from synthesis.synthesizer import GroundedSynthesizer
from tools.reextract_sp500_chunks import run_sp500_chunk_reextraction
from universe.sp500 import SP500Constituent

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeCursor:
    def __init__(self, executed: List[Tuple[str, Any]], rows_queue: Optional[List[List[Tuple[Any, ...]]]] = None) -> None:
        self.executed = executed
        self._rows_queue = list(rows_queue or [])

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return self._rows_queue.pop(0) if self._rows_queue else []

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        if self._rows_queue:
            batch = self._rows_queue.pop(0)
            return batch[0] if batch else None
        return None

    def close(self) -> None:
        return None


class _FakePGConn:
    def __init__(self, rows_queue: Optional[List[List[Tuple[Any, ...]]]] = None) -> None:
        self.executed: List[Tuple[str, Any]] = []
        self._rows_queue = rows_queue or []
        self.committed = False

    def __enter__(self) -> "_FakePGConn":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.executed, self._rows_queue)

    def commit(self) -> None:
        self.committed = True


# ----------------------------------------------------------------------
# 1. Embedder Upgrade (Qwen3-Embedding-0.6B, 1024 dims, CPU pinning, RRF)
# ----------------------------------------------------------------------
def test_qwen3_embedding_defaults_cpu_pinning_and_asymmetric_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder.reset_embedder_cache()
    monkeypatch.delenv("EMBEDDER_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("YARNBALL_EMBEDDER_MOCK", raising=False)

    assert embedder._get_model_name() == "qwen3-embedding:0.6b"
    assert embedder.get_embedding_dim() == 1024

    live_client = embedder.get_embedder_model(allow_mock=False, batch_gpu=False)
    live_payload = live_client.build_request_payload(["hello"])
    assert live_payload["model"] == "qwen3-embedding:0.6b"
    assert live_payload["options"] == {"num_gpu": 0}

    gpu_client = embedder.get_embedder_model(allow_mock=False, batch_gpu=True)
    gpu_payload = gpu_client.build_request_payload(["hello"])
    assert "options" not in gpu_payload

    captured_inputs: List[List[str]] = []

    def _capture_embed(texts: List[str], **kwargs: Any) -> List[List[float]]:
        captured_inputs.append(list(texts))
        return [[0.5, 0.5] for _ in texts]

    with patch("embedding.embedder.embed_texts", side_effect=_capture_embed):
        q_vec = embedder.embed_query("TSMC wafer disruption")
        d_vecs = embedder.embed_documents(["Apple relies on TSMC for silicon."])

    assert len(q_vec) == 2
    assert len(d_vecs) == 1
    assert captured_inputs[0][0].startswith(embedder.QUERY_INSTRUCTION_PREFIX)
    assert captured_inputs[1][0] == "Apple relies on TSMC for silicon."


def test_search_chunks_hybrid_rrf_fuses_dense_and_lexical_ranks_with_pit() -> None:
    dense_rows = [
        ("acc#item_1:c001", "acc", "sec_filing", "0000320193", "Item 1", "Dense hit", "2024-01-02", None, 0.91),
        ("acc#item_1:c002", "acc", "sec_filing", "0000320193", "Item 1", "Shared hit", "2024-01-02", None, 0.85),
    ]
    lexical_rows = [
        ("acc#item_1:c002", "acc", "sec_filing", "0000320193", "Item 1", "Shared hit", "2024-01-02", None, 0.95),
        ("acc#item_1:c003", "acc", "sec_filing", "0000320193", "Item 1", "Lexical hit", "2024-01-02", None, 0.70),
    ]
    conn = _FakePGConn(rows_queue=[dense_rows, lexical_rows])
    ctx = PiTContext(as_of=date(2024, 5, 1), mode="historical")

    with patch("embedding.embedder.embed_query", return_value=[0.6, 0.8]):
        ranked = search_chunks_hybrid_rrf("TSMC Fab 18", ctx=ctx, top_k=3, conn=conn)

    # Chunk present in BOTH dense and lexical top lists must rank #1 under RRF
    assert ranked[0]["chunk_id"] == "acc#item_1:c002"
    assert ranked[0]["rrf_score"] > ranked[1]["rrf_score"]
    assert any("valid_from <= %s::date" in sql for sql, _ in conn.executed)
    assert any("plainto_tsquery('english'" in sql for sql, _ in conn.executed)


# ----------------------------------------------------------------------
# 2. Invariants #3 & #4: No Truncation + Chunk-Scoped Negation + Gleaning
# ----------------------------------------------------------------------
def test_adr00035_invariant4_no_text_dropped_and_invariant3_chunk_scoped_negation(
    tmp_path: Path,
) -> None:
    """Long SEC sections (>2500 chars) are segmented without truncation; hedging is scoped per chunk."""
    filler = "Routine general corporate governance commentary covering administrative board committees. " * 45
    long_section = (
        "ITEM 1A. RISK FACTORS\n\n"
        "Apple Inc. may from time to time consider partnering with VendorHedged as a backup supplier.\n\n"
        f"{filler}\n\n"
        f"{filler}\n\n"
        "In contrast, Apple Inc. relies on Taiwan Semiconductor Manufacturing as a sole-source wafer supplier "
        "representing 40% of procurement with 30 days of inventory buffer."
    )
    assert len(long_section) > 6000  # Well beyond legacy 2500-char ceiling

    seen_prompts: List[str] = []

    class _ChunkAwareClient(YarnBallModelClient):
        def generate(
            self,
            prompt: str,
            task: str = "EXTRACT_SEC_GRAPH",
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            enforce_think_prefix: bool = True,
            num_ctx: int = 8192,
        ) -> ModelResponse:
            seen_prompts.append(prompt)
            if "Pass-1 Already Extracted Triples" in prompt:
                raw = '{"nodes": [], "edges": []}'
            elif "VendorHedged" in prompt:
                raw = json.dumps(
                    {
                        "nodes": [
                            {"id": "Apple Inc.", "type": "Company"},
                            {"id": "VendorHedged", "type": "Company"},
                        ],
                        "edges": [
                            {
                                "source": "VendorHedged",
                                "target": "Apple Inc.",
                                "type": "SUPPLIES_TO",
                                "properties": {},
                            }
                        ],
                    }
                )
            elif "Taiwan Semiconductor Manufacturing" in prompt:
                raw = json.dumps(
                    {
                        "nodes": [
                            {"id": "Apple Inc.", "type": "Company"},
                            {"id": "Taiwan Semiconductor Manufacturing", "type": "Company"},
                        ],
                        "edges": [
                            {
                                "source": "Taiwan Semiconductor Manufacturing",
                                "target": "Apple Inc.",
                                "type": "SUPPLIES_TO",
                                "properties": {"share": 0.40, "buffer_days": 30},
                            }
                        ],
                    }
                )
            else:
                raw = '{"nodes": [], "edges": []}'
            return ModelResponse(
                raw_text=raw,
                think_trace=None,
                final_output=raw,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=100,
                generated_tokens=60,
            )

    res = extract_sec_relationships(
        long_section,
        model_client=_ChunkAwareClient(),
        rejection_queue=RuntimeRejectionQueue(base_dir=tmp_path / "rej"),
        doc_id="0000320193-24-000106",
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        enable_gleaning=False,
    )

    # All chunks were processed (zero truncation!) — late disclosure past char 6000 was reached
    assert len(res["chunk_ids"]) >= 2
    assert any("Taiwan Semiconductor Manufacturing" in p for p in seen_prompts)
    # Chunk 1 hedged VendorHedged edge was dropped without leaking hedge onto Chunk 2's Apple Inc. edge (Invariant #3)
    sources = [e["source"] for e in res["edges"]]
    assert "VendorHedged" not in sources
    assert "Taiwan Semiconductor Manufacturing" in sources
    # Extracted edge carries deterministic chunk_id pointer
    tsm_edge = next(e for e in res["edges"] if e["source"] == "Taiwan Semiconductor Manufacturing")
    assert len(tsm_edge["properties"]["evidence_chunk_ids"]) == 1
    assert tsm_edge["properties"]["evidence_chunk_ids"][0].startswith("0000320193-24-000106#item_1a_risk_factors:c")


def test_targeted_gleaning_rescues_pass2_edges_and_quarantines_pass2_errors_nondestructively(
    tmp_path: Path,
) -> None:
    """ADR-00035 Section 4.4: Targeted 1-pass gleaning rescues edges or fails non-destructively."""
    assert should_glean_chunk("Item 1 - Business", pass1_valid_edge_count=1, generated_tokens=100, runaway_detected=False, enabled=True) is True
    assert should_glean_chunk("Item 1A - Risk Factors", pass1_valid_edge_count=0, generated_tokens=450, runaway_detected=False, enabled=True) is True
    assert should_glean_chunk("Item 8.01 - Other Events", pass1_valid_edge_count=1, generated_tokens=80, runaway_detected=False, enabled=True) is True
    assert should_glean_chunk("Item 7 - Management Discussion and Analysis", pass1_valid_edge_count=2, generated_tokens=500, runaway_detected=False, enabled=True) is False
    assert should_glean_chunk("Item 1 - Business", pass1_valid_edge_count=2, generated_tokens=500, runaway_detected=True, enabled=True) is False
    assert should_glean_chunk("Item 1 - Business", pass1_valid_edge_count=2, generated_tokens=500, runaway_detected=False, enabled=False) is False

    section = (
        "ITEM 1. BUSINESS\n\n"
        "Apple Inc. relies on TSMC for wafer fabrication and Broadcom for wireless RF chips."
    )

    # Case A: Pass 2 rescues Broadcom -> Apple Inc. tagged extraction_pass=2
    class _GleaningRescueClient(YarnBallModelClient):
        def generate(
            self,
            prompt: str,
            task: str = "EXTRACT_SEC_GRAPH",
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            enforce_think_prefix: bool = True,
            num_ctx: int = 8192,
        ) -> ModelResponse:
            if "Pass-1 Already Extracted Triples" in prompt:
                raw = json.dumps(
                    {
                        "nodes": [{"id": "Broadcom", "type": "Company"}],
                        "edges": [
                            {"source": "Broadcom", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {"nature": "rf_chips"}}
                        ],
                    }
                )
            else:
                raw = json.dumps(
                    {
                        "nodes": [
                            {"id": "TSMC", "type": "Company"},
                            {"id": "Apple Inc.", "type": "Company"},
                        ],
                        "edges": [
                            {"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {"nature": "wafers"}}
                        ],
                    }
                )
            return ModelResponse(
                raw_text=raw,
                think_trace=None,
                final_output=raw,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=50,
                generated_tokens=80,
            )

    rescued = extract_sec_relationships(
        section,
        model_client=_GleaningRescueClient(),
        doc_id="0000320193-24-000106",
        enable_gleaning=True,
    )
    assert rescued["gleaning_new_edges_count"] == 1
    by_src = {e["source"]: e for e in rescued["edges"]}
    assert by_src["TSMC"]["properties"]["extraction_pass"] == 1
    assert by_src["Broadcom"]["properties"]["extraction_pass"] == 2

    # Case B: Pass 2 malformed completion quarantines GLEANING_PARSE_FAILED without discarding Pass 1
    rej_queue = RuntimeRejectionQueue(base_dir=tmp_path / "glean_rej")

    class _GleaningFailClient(YarnBallModelClient):
        def generate(
            self,
            prompt: str,
            task: str = "EXTRACT_SEC_GRAPH",
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            enforce_think_prefix: bool = True,
            num_ctx: int = 8192,
        ) -> ModelResponse:
            if "Pass-1 Already Extracted Triples" in prompt:
                raw = "MALFORMED_PASS2_NOT_JSON"
            else:
                raw = json.dumps(
                    {
                        "nodes": [{"id": "TSMC", "type": "Company"}, {"id": "Apple Inc.", "type": "Company"}],
                        "edges": [{"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}}],
                    }
                )
            return ModelResponse(
                raw_text=raw,
                think_trace=None,
                final_output=raw,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=50,
                generated_tokens=80,
            )

    preserved = extract_sec_relationships(
        section,
        model_client=_GleaningFailClient(),
        rejection_queue=rej_queue,
        doc_id="0000320193-24-000106",
        enable_gleaning=True,
    )
    assert len(preserved["edges"]) == 1
    assert preserved["edges"][0]["source"] == "TSMC"
    failures = rej_queue.load_records()
    assert len(failures) == 1
    assert failures[0].error_stage == ErrorStage.GLEANING_PARSE_FAILED


def test_cross_chunk_merge_unions_evidence_and_selects_highest_confidence_provenance() -> None:
    payload_default = {
        "chunk_id": "acc#item_1:c001",
        "nodes": [{"id": "TSMC", "type": "Company"}, {"id": "Apple Inc.", "type": "Company"}],
        "edges": [
            {
                "source": "TSMC",
                "target": "Apple Inc.",
                "type": "SUPPLIES_TO",
                "properties": {"nature": "wafers"},
            }
        ],
    }
    payload_quant = {
        "chunk_id": "acc#item_1a:c002",
        "nodes": [{"id": "TSMC", "type": "Company"}],
        "edges": [
            {
                "source": "TSMC",
                "target": "Apple Inc.",
                "type": "SUPPLIES_TO",
                "properties": {"share": 0.45, "buffer_days": 30.0},
            }
        ],
    }
    merged = merge_chunk_extraction_payloads([payload_default, payload_quant])
    assert len(merged["edges"]) == 1
    edge_props = merged["edges"][0]["properties"]
    assert edge_props["evidence_chunk_ids"] == ["acc#item_1:c001", "acc#item_1a:c002"]
    assert edge_props["share"] == 0.45
    assert edge_props["buffer_days"] == 30.0


# ----------------------------------------------------------------------
# 3. Invariants #1, #2, #5: Evidence Pointers, Thin Memgraph & PiT Inheritance
# ----------------------------------------------------------------------
def test_adr00035_invariants_1_2_5_thin_memgraph_and_chunk_pit_inheritance() -> None:
    """New SEC & news edges carry evidence_chunk_ids; Memgraph has zero :Chunk nodes; chunks inherit valid_from."""
    # Static check: no (:Chunk) node creation anywhere in runtime packages
    for pkg in ("ingestion", "knowledge_graph", "retrieval", "streaming", "tools"):
        for py_path in (REPO_ROOT / pkg).rglob("*.py"):
            content = py_path.read_text(encoding="utf-8")
            assert "(:Chunk" not in content and ":Chunk " not in content, (
                f"ADR-00035 Invariant #2 violation: (:Chunk) node reference in {py_path}"
            )

    pg_conn = _FakePGConn()
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # 1. News persistence
    with patch("knowledge_graph.embedding_store.store_node_embeddings"):
        store_graph_entities(
            source_hash="news_hash_42",
            text="TSMC supplies advanced wafers to Apple Inc.",
            graph_data={
                "nodes": [
                    {"id": "TSMC", "type": "Company"},
                    {"id": "Apple Inc.", "type": "Company"},
                ],
                "edges": [
                    {
                        "source": "TSMC",
                        "target": "Apple Inc.",
                        "type": "SUPPLIES_TO",
                        "properties": {"raw_text": "MUST_BE_STRIPPED_FROM_MEMGRAPH"},
                    }
                ],
            },
            driver=mock_driver,
            published_at="2024-05-01T18:00:00Z",  # 14:00 EDT -> same session 2024-05-01
            pg_conn=pg_conn,
        )

    # Postgres received document_chunks with exact inherited valid_from = 2024-05-01 (Invariant #5)
    chunk_inserts = [params for sql, params in pg_conn.executed if "INSERT INTO document_chunks" in sql]
    assert len(chunk_inserts) == 1
    assert chunk_inserts[0][0] == "news_hash_42#body:c001"
    assert chunk_inserts[0][8] == "2024-05-01"

    # Memgraph edge call received compact evidence_chunk_ids and stripped raw prose (Invariants #1 & #2)
    edge_calls = [c for c in mock_session.run.call_args_list if "MERGE (s)-[r:SUPPLIES_TO]->(t)" in c.args[0]]
    assert len(edge_calls) == 1
    kwargs = edge_calls[0].kwargs
    assert kwargs["evidence_chunk_ids"] == ["news_hash_42#body:c001"]
    assert kwargs["evidence_status"] == "chunked"
    assert "raw_text" not in kwargs["props"]
    assert kwargs["valid_from"] == "2024-05-01"

    # 2. ExtractionGraphPayload accepts chunk_id
    payload = ExtractionGraphPayload(chunk_id="news_hash_42#body:c001", nodes=[], edges=[])
    assert payload.chunk_id == "news_hash_42#body:c001"


# ----------------------------------------------------------------------
# 4. Snapshot chunks.parquet, Verbalizer Citation Handles & Synthesizer Gate
# ----------------------------------------------------------------------
def test_snapshot_chunks_parquet_verbalizer_handles_and_synthesizer_citation_gate(
    tmp_path: Path,
) -> None:
    """SnapshotManager exports chunks.parquet; verbalizer emits [E1 <- chunk_id]; uncited answers quarantine."""
    cid = make_chunk_id("0000320193-24-000106", "Item 1 - Business", 1)
    chunk_text = "Apple Inc. depends on TSMC as sole-source manufacturer for A-series silicon."

    mock_session = MagicMock()
    mock_session.run.side_effect = [
        [{"labels": ["Company"], "props": {"id": "TSMC", "ticker": "TSM"}}],
        [
            {
                "source_id": "TSMC",
                "target_id": "Apple Inc.",
                "rel_type": "SUPPLIES_TO",
                "source_hash": "0000320193-24-000106:Item 1",
                "props": {"evidence_chunk_ids": [cid], "valid_from": "2024-11-01"},
            }
        ],
    ]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mgr = SnapshotManager(base_dir=tmp_path / "snapshots")
    with patch("knowledge_graph.snapshot_manager.pg_connection", side_effect=RuntimeError("offline")):
        meta = mgr.export_snapshot(
            snapshot_id="G_adr35",
            driver=mock_driver,
            chunks=[
                DocumentChunkRecord(
                    chunk_id=cid,
                    doc_id="0000320193-24-000106",
                    source_type="sec_filing",
                    cik="0000320193",
                    section="Item 1 - Business",
                    char_start=0,
                    char_end=len(chunk_text),
                    text=chunk_text,
                    valid_from="2024-11-01",
                    known_at="2024-11-01T15:00:00+00:00",
                ).to_dict()
            ],
        )

    chunks_parquet = tmp_path / "snapshots" / "G_adr35" / "chunks.parquet"
    assert chunks_parquet.is_file()
    assert meta["chunk_count"] == 1
    assert pq.read_table(chunks_parquet).num_rows == 1

    # Single batched lookup via chunks.parquet
    fetched = fetch_chunks_by_ids([cid], chunks_parquet_path=chunks_parquet)
    assert fetched[cid]["text"] == chunk_text

    # Verbalizer emits [E1 <- chunk_id] and resolves passage ONLY for clamped edges
    verbalizer = HopOrderedSubgraphVerbalizer(max_edges=1, chunks_parquet_path=chunks_parquet)
    verbalized = verbalizer.verbalize(
        [
            {"source": "TSMC", "target": "AAPL", "rel_type": "SUPPLIES_TO", "hop": 1, "evidence_chunk_ids": [cid]},
            {"source": "AAPL", "target": "FOXCONN", "rel_type": "SUPPLIES_TO", "hop": 2, "evidence_chunk_ids": ["dropped#c001"]},
        ]
    )
    assert f"[E1 ← {cid}]" in verbalized
    assert "### Supporting Evidence Passages (ADR-00035):" in verbalized
    assert "sole-source manufacturer for A-series silicon" in verbalized
    assert "dropped#c001" not in verbalized

    # GroundedSynthesizer enforces citation handles and quarantines uncited claims
    rej_queue = RuntimeRejectionQueue(base_dir=tmp_path / "synth_rej")
    uncited_client = YarnBallModelClient(transport_fn=lambda _p: {"response": "TSMC supplies Apple silicon."})
    synth_bad = GroundedSynthesizer(model_client=uncited_client, rejection_queue=rej_queue)
    res_bad = synth_bad.synthesize(
        "Who supplies Apple?",
        {
            "subgraph_triples": [{"source": "TSMC", "rel_type": "SUPPLIES_TO", "target": "AAPL", "evidence_chunk_ids": [cid]}],
            "articles": [],
        },
    )
    assert res_bad["is_grounded"] is False
    assert len(res_bad["citation_violations"]) == 1
    assert len(rej_queue.load_records()) == 1

    cited_client = YarnBallModelClient(transport_fn=lambda _p: {"response": "TSMC supplies Apple silicon [E1]."})
    synth_good = GroundedSynthesizer(model_client=cited_client, rejection_queue=rej_queue)
    res_good = synth_good.synthesize(
        "Who supplies Apple?",
        {
            "subgraph_triples": [{"source": "TSMC", "rel_type": "SUPPLIES_TO", "target": "AAPL", "evidence_chunk_ids": [cid]}],
            "articles": [],
        },
    )
    assert res_good["is_grounded"] is True
    assert res_good["citation_violations"] == []


# ----------------------------------------------------------------------
# 5. Incremental Resumable PiT S&P 500 Batch Re-Extraction
# ----------------------------------------------------------------------
def test_resumable_sp500_chunk_reextraction_job_checkpoints_idempotently(tmp_path: Path) -> None:
    filings_dir = tmp_path / "sec_filings"
    bundle = filings_dir / "AAPL" / "0000320193-24-000106"
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "cik": "0000320193",
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "accession_number": "0000320193-24-000106",
                "filing_date": "2024-11-01",
                "acceptance_datetime": "2024-11-01T15:00:00Z",
                "form": "10-K",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "item_1_business.txt").write_text(
        "ITEM 1. BUSINESS\n\nApple Inc. relies on TSMC for leading-edge wafer fabrication.",
        encoding="utf-8",
    )

    mock_universe = MagicMock()
    mock_universe.get_constituents_at_date.return_value = [
        SP500Constituent(
            ticker="AAPL",
            cik="0000320193",
            company_name="Apple Inc.",
            gics_sector="Information Technology",
        )
    ]

    mock_client = YarnBallModelClient(
        transport_fn=lambda _p: {
            "response": json.dumps(
                {
                    "nodes": [{"id": "TSMC", "type": "Company"}, {"id": "Apple Inc.", "type": "Company"}],
                    "edges": [{"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}}],
                }
            )
        }
    )
    pg_conn = _FakePGConn()
    mock_driver = MagicMock()
    checkpoint = tmp_path / "checkpoints" / "reextract.jsonl"

    with patch("knowledge_graph.embedding_store.store_node_embeddings"):
        first = run_sp500_chunk_reextraction(
            as_of_date="2024-11-05",
            filings_dir=filings_dir,
            checkpoint_path=checkpoint,
            pg_conn=pg_conn,
            memgraph_driver=mock_driver,
            model_client=mock_client,
            universe_manager=mock_universe,
        )
        second = run_sp500_chunk_reextraction(
            as_of_date="2024-11-05",
            filings_dir=filings_dir,
            checkpoint_path=checkpoint,
            pg_conn=pg_conn,
            memgraph_driver=mock_driver,
            model_client=mock_client,
            universe_manager=mock_universe,
        )

    assert first["processed_bundles"] == 1
    assert first["skipped_checkpoints"] == 0
    assert first["chunks_persisted"] >= 1
    assert first["edges_persisted"] == 1

    assert second["processed_bundles"] == 0
    assert second["skipped_checkpoints"] == 1
