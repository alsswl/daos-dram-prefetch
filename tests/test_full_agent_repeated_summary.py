"""Repeated-agentic report tests: synthetic files only, no model/DAOS/GPU."""
from copy import deepcopy
import csv
import json
import statistics
import sys

import pytest

import summarize_full_agent_repeated as summary


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def repeated_folder(tmp_path):
    """One fill and ten client-restart workflows for each storage mode."""
    write_json(tmp_path / "manifest.json", {"args": {
        "restart_runs": 10, "model": "fixture-model", "chunk_size": 128,
        "gpu_buffer_gb": 10, "io_workers": 16, "meta_workers": 16,
    }})
    write_json(tmp_path / "status.json", {"status": "complete"})
    results = []
    for index in range(11):
        for mode in summary.MODES:
            run = f"run{index + 1}"
            tag = f"{mode}_{run}"
            path = tmp_path / tag
            path.mkdir()
            # Vary warm call counts to distinguish unweighted and weighted TTFT.
            n = 4 if index % 2 == 0 else 3
            tools, calls, requests = [], {}, []
            for call in range(n):
                prompt = 1597 + 128 * call
                hit = 0 if index == 0 and call == 0 else ((prompt - 1) // 128) * 128
                requests.append({"request_id": f"{tag}-{call}", "prompt_tokens": prompt,
                                 "lmcache_hit_tokens": hit, "load_tokens": hit})
                calls[f"call-{call}"] = {
                    "elapsed_seconds": .2,
                    "llm_output": {"token_usage": {"prompt_tokens": prompt, "completion_tokens": 25}},
                    "generations": [[{"text": f"same model output {index}-{call}"}]],
                }
                if call < n - 1:
                    tools.append({"code": f"print({call})", "output": str(call), "elapsed_seconds": .01})
            write_json(path / "tool_calls.json", tools)
            write_json(path / "llm_calls.json", calls)
            (tmp_path / f"{tag}_server.log").write_text(
                "INFO fixture model running\nWARNING fixture warning\n"
                "INFO [shutdown] fixture shutdown\nERROR fixture shutdown-only error\n")
            results.append({
                "status": "complete", "mode": mode, "run": run,
                "phase": "fill" if index == 0 else "restart_hit", "restart_index": index,
                "execution_sequence": len(results) + 1, "final_answer": f"final answer {index}",
                "workflow_seconds": 999.0 if index == 0 else float(index),
                "llm_calls": n, "python_calls": n - 1, "ttft_count": float(n),
                "mean_server_ttft_ms": 9999.0 if index == 0 else 10.0 * index,
                "cache_requests": requests,
            })
    write_json(tmp_path / "summary.json", results)
    return tmp_path


def test_collect_all_22_runs_and_compare_artifacts(repeated_folder):
    manifest, rows, artifacts, audits, validation = summary.collect(repeated_folder)
    assert validation["passed"], validation
    assert validation["expected_total_workflows"] == validation["observed_total_workflows"] == 22
    assert validation["restart_runs_requested_per_mode"] == 10
    assert len(rows) == len(artifacts) == len(audits) == 22
    assert all(row["execution_sequence"] for row in rows)
    assert all(not entry["errors_before_shutdown"] for entry in audits)
    assert all(len(entry["errors_after_shutdown"]) == 1 for entry in audits)
    checks = summary.equivalence(artifacts, 10)
    assert len(checks) == 49
    between_modes = [c for c in checks if c["comparison"] == "same_index_between_modes"]
    assert len(between_modes) == 11
    assert all(c["code_and_observations_equal"] and c["final_answers_equal"] for c in between_modes)
    assert any(not c["final_answers_equal"] for c in checks if c["comparison"] == "fill_vs_restart")
    assert manifest["args"]["restart_runs"] == 10


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_run_coverage_rejected(repeated_folder, change):
    path = repeated_folder / "summary.json"
    rows = summary.read_json(path)
    if change == "missing":
        rows.pop()
    else:
        rows.append(deepcopy(rows[-1]))
    write_json(path, rows)
    validation = summary.collect(repeated_folder)[-1]
    assert not validation["passed"]
    assert any("coverage" in error for error in validation["errors"])


@pytest.mark.parametrize(("field", "value", "expected_error"), [
    ("llm_calls", 99, "counts differ"),
    ("ttft_count", 99, "counts differ"),
    ("python_calls", 99, "tool-call count differs"),
    ("phase", "fill", "phase/restart_index"),
    ("restart_index", 8, "phase/restart_index"),
    ("workflow_seconds", float("nan"), "invalid workflow_seconds"),
    ("mean_server_ttft_ms", float("nan"), "invalid mean_server_ttft_ms"),
])
def test_invalid_counts_phase_or_latency_rejected(repeated_folder, field, value, expected_error):
    path = repeated_folder / "summary.json"
    rows = summary.read_json(path)
    rows[2][field] = value  # dfs_run2
    write_json(path, rows)
    validation = summary.collect(repeated_folder)[-1]
    assert not validation["passed"]
    assert any(expected_error in error for error in validation["errors"])


@pytest.mark.parametrize("change", ["cold_hit", "warm_miss", "not_loaded", "unaligned", "duplicate_id"])
def test_invalid_cache_evidence_rejected(repeated_folder, change):
    path = repeated_folder / "summary.json"
    rows = summary.read_json(path)
    request = rows[0 if change == "cold_hit" else 2]["cache_requests"][0]
    if change == "cold_hit":
        request.update(lmcache_hit_tokens=128, load_tokens=128)
    elif change == "warm_miss":
        request.update(lmcache_hit_tokens=0, load_tokens=0)
    elif change == "not_loaded":
        request["load_tokens"] = 0
    elif change == "unaligned":
        request.update(lmcache_hit_tokens=1535, load_tokens=1535)
    else:
        request["request_id"] = rows[2]["cache_requests"][1]["request_id"]
    write_json(path, rows)
    assert not summary.collect(repeated_folder)[-1]["passed"]


@pytest.mark.parametrize("change", ["incomplete", "runtime_error", "missing_shutdown", "usage_mismatch"])
def test_status_logs_and_callback_usage_rejected(repeated_folder, change):
    if change == "incomplete":
        write_json(repeated_folder / "status.json", {"status": "running"})
    elif change == "runtime_error":
        path = repeated_folder / "dfs_run2_server.log"
        path.write_text("ERROR runtime failure\nINFO [shutdown] stop\n")
    elif change == "missing_shutdown":
        (repeated_folder / "dfs_run2_server.log").write_text("INFO no shutdown evidence\n")
    else:
        path = repeated_folder / "dfs_run2" / "llm_calls.json"
        calls = summary.read_json(path)
        calls["call-0"]["llm_output"]["token_usage"]["prompt_tokens"] += 1
        write_json(path, calls)
    assert not summary.collect(repeated_folder)[-1]["passed"]


def test_fill_warm_statistics_separate_and_ttft_weighted(repeated_folder):
    _, rows, *_ = summary.collect(repeated_folder)
    stats = summary.aggregate(rows)
    assert len(stats) == 4
    for mode in summary.MODES:
        fill = next(s for s in stats if s["mode"] == mode and s["phase"] == "fill")
        warm = next(s for s in stats if s["mode"] == mode and s["phase"] == "restart_hit")
        assert fill["n"] == 1
        assert fill["workflow_seconds_mean"] == 999
        assert fill["workflow_seconds_stdev"] is None
        assert warm["n"] == 10
        assert warm["workflow_seconds_mean"] == warm["workflow_seconds_median"] == 5.5
        assert warm["workflow_seconds_min"] == 1
        assert warm["workflow_seconds_max"] == 10
        assert warm["workflow_seconds_stdev"] == pytest.approx(statistics.stdev(range(1, 11)))
        assert warm["mean_server_ttft_ms_mean"] == 55
        assert warm["server_ttft_call_weighted_mean_ms"] == pytest.approx(1950 / 35)
        assert warm["llm_calls_distribution"] == {"3": 5, "4": 5}
        assert warm["python_calls_distribution"] == {"2": 5, "3": 5}
        assert warm["first_request_hit_tokens_distribution"] == {"1536": 10}


def test_cli_artifacts_written_only_to_fixture(repeated_folder, monkeypatch):
    original = (repeated_folder / "summary.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["summarize_full_agent_repeated.py", str(repeated_folder)])
    summary.main()
    assert (repeated_folder / "summary.json").read_bytes() == original
    assert summary.read_json(repeated_folder / "repeated_validation.json")["passed"]
    with (repeated_folder / "full_agentic_repeated_runs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 22
    with (repeated_folder / "full_agentic_repeated_statistics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert len(summary.read_json(repeated_folder / "mode_equivalence.json")) == 49
    report = (repeated_folder / "REPEATED_AGENTIC_RESULT_KO.md").read_text()
    assert "오류 자동 재시도가 아니라" in report
    assert "호출수 가중 평균" in report
    assert "캐시가 반복마다 누적" in report


def test_cli_refuses_success_report_on_validation_failure(repeated_folder, monkeypatch):
    write_json(repeated_folder / "status.json", {"status": "running"})
    monkeypatch.setattr(sys, "argv", ["summarize_full_agent_repeated.py", str(repeated_folder)])
    with pytest.raises(ValueError, match="validation failed"):
        summary.main()
    assert not summary.read_json(repeated_folder / "repeated_validation.json")["passed"]
    assert (repeated_folder / "log_audit.json").exists()
    assert not (repeated_folder / "REPEATED_AGENTIC_RESULT_KO.md").exists()
    assert not (repeated_folder / "full_agentic_repeated_runs.csv").exists()
