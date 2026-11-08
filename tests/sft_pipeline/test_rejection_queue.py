"""Unit tests for Stage 6.6: RuntimeRejectionQueue."""

import json
import threading
from pathlib import Path
import pytest
from sft_pipeline.rejection_queue import (
    RuntimeRejectionQueue,
    RejectionRecord,
    ErrorStage,
)


@pytest.fixture
def temp_queue(tmp_path: Path) -> RuntimeRejectionQueue:
    return RuntimeRejectionQueue(base_dir=tmp_path)


def test_record_failure_appends_to_daily_file(temp_queue: RuntimeRejectionQueue):
    path = temp_queue.record_failure(
        task_id="TEXT_TO_CYPHER",
        prompt="Find companies where ceo_age > 60",
        completion="MATCH (c:Company) WHERE c.ceo_age > 60 RETURN c",
        error_stage=ErrorStage.SCHEMA_AST_VIOLATION,
        error_message="Property 'ceo_age' does not exist in graph schema.",
        metadata={"invalid_properties": ["ceo_age"]},
    )
    assert path.exists()
    assert path.name.startswith("rejections_")
    assert path.name.endswith(".jsonl")

    records = temp_queue.load_records()
    assert len(records) == 1
    r = records[0]
    assert r.task_id == "TEXT_TO_CYPHER"
    assert r.error_stage == ErrorStage.SCHEMA_AST_VIOLATION
    assert r.metadata["invalid_properties"] == ["ceo_age"]
    assert "ceo_age" in r.prompt


def test_thread_safe_concurrent_writes(temp_queue: RuntimeRejectionQueue):
    n_threads = 5
    records_per_thread = 20

    def worker(worker_id: int):
        for i in range(records_per_thread):
            temp_queue.record_failure(
                task_id="CONTAGION_REASONING",
                prompt=f"Prompt from worker {worker_id} iteration {i}",
                completion="Unclosed reasoning...",
                error_stage=ErrorStage.RUNAWAY_LOOP,
                error_message=f"Worker {worker_id} tripped watchdog",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = temp_queue.load_records()
    assert len(records) == n_threads * records_per_thread


def test_get_summary_stats(temp_queue: RuntimeRejectionQueue):
    temp_queue.record_failure("TEXT_TO_CYPHER", "p1", "c1", ErrorStage.SCHEMA_AST_VIOLATION, "err1")
    temp_queue.record_failure("TEXT_TO_CYPHER", "p2", "c2", ErrorStage.SCHEMA_AST_VIOLATION, "err2")
    temp_queue.record_failure("CONTAGION_REASONING", "p3", "c3", ErrorStage.RUNAWAY_LOOP, "err3")
    temp_queue.record_failure("EXTRACT_SEC_GRAPH", "p4", "c4", ErrorStage.ONTOLOGY_PRUNING, "err4")

    stats = temp_queue.get_summary_stats()
    assert stats["total_rejections"] == 4
    assert stats["by_error_stage"][ErrorStage.SCHEMA_AST_VIOLATION] == 2
    assert stats["by_error_stage"][ErrorStage.RUNAWAY_LOOP] == 1
    assert stats["by_error_stage"][ErrorStage.ONTOLOGY_PRUNING] == 1
    assert stats["by_task_id"]["TEXT_TO_CYPHER"] == 2


def test_export_dpo_candidates(temp_queue: RuntimeRejectionQueue, tmp_path: Path):
    temp_queue.record_failure(
        task_id="TEXT_TO_CYPHER",
        prompt="Query with fake property",
        completion="MATCH (c:Company) WHERE c.fake_field = 10 RETURN c",
        error_stage=ErrorStage.SCHEMA_AST_VIOLATION,
        error_message="Unknown property",
        metadata={"invalid_properties": ["fake_field"]},
    )
    temp_queue.record_failure(
        task_id="EXTRACT_SEC_GRAPH",
        prompt="10-K section text",
        completion="(:Company)-[:COMPARES_TO]->(:Company)",
        error_stage=ErrorStage.ONTOLOGY_PRUNING,
        error_message="Non-schema predicate",
    )

    out_file = tmp_path / "exported_dpo.jsonl"
    candidates = temp_queue.export_dpo_candidates(output_path=out_file)

    assert len(candidates) == 2
    assert out_file.exists()

    lines = [json.loads(l) for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["task"] == "TEXT_TO_CYPHER"
    assert "fake_field" in lines[0]["chosen"]
    assert lines[0]["rejected"] == "MATCH (c:Company) WHERE c.fake_field = 10 RETURN c"
    assert lines[1]["chosen"] == "(none)"


def test_corrupt_line_handling_graceful(temp_queue: RuntimeRejectionQueue):
    log_path = temp_queue.record_failure("Task1", "p", "c", ErrorStage.UNKNOWN, "msg")

    # Manually append corrupt non-JSON line
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("{corrupt: json line -- invalid syntax\n")

    # Append valid record
    temp_queue.record_failure("Task2", "p2", "c2", ErrorStage.UNKNOWN, "msg2")

    records = temp_queue.load_records()
    assert len(records) == 2  # The corrupt line was gracefully skipped

