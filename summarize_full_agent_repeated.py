"""Validate and summarize repeated full-agentic runs without merging fill/hit.

This reads existing results only; it neither starts models nor deletes caches.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics


MODES = ("dfs", "object")
PHASES = ("fill", "restart_hit")
ERROR_RE = re.compile(r"\bERROR\b|Double free|reference count.*negative|DaosGdsBackend.*failed", re.I)
WARNING_RE = re.compile(r"\bWARNING\b|\bUserWarning\b")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def describe(values):
    """Sample standard deviation is undefined for one observation, not zero."""
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def distribution(values):
    return dict(sorted(Counter(str(x) for x in values).items(), key=lambda x: int(x[0])))


def audit_log(folder, tag):
    text = (folder / f"{tag}_server.log").read_text(errors="replace")
    pre, separator, post = text.partition("[shutdown]")
    return {
        "run": tag,
        "shutdown_marker_found": bool(separator),
        "errors_before_shutdown": [s for s in pre.splitlines() if ERROR_RE.search(s)],
        "errors_after_shutdown": [s for s in post.splitlines() if ERROR_RE.search(s)],
        "warnings_before_shutdown": [s for s in pre.splitlines() if WARNING_RE.search(s)],
        "warnings_after_shutdown": [s for s in post.splitlines() if WARNING_RE.search(s)],
    }


def collect(folder):
    manifest = read_json(folder / "manifest.json")
    args = manifest["args"]
    restart_runs = int(args.get("restart_runs", 1))
    if restart_runs < 1:
        raise ValueError("manifest.args.restart_runs must be positive")
    chunk = int(args["chunk_size"])
    if chunk < 1:
        raise ValueError("chunk_size must be positive")
    results = read_json(folder / "summary.json")
    errors, audits, rows, artifacts = [], [], [], {}
    if read_json(folder / "status.json").get("status") != "complete":
        errors.append("The workflow status is not complete")
    expected = {(mode, f"run{i + 1}") for mode in MODES for i in range(restart_runs + 1)}
    found = Counter((r["mode"], r["run"]) for r in results)
    if set(found) != expected or any(n != 1 for n in found.values()):
        errors.append(f"Run coverage differs: expected {len(expected)} unique mode/run pairs, found {dict(found)}")
    for result in sorted(results, key=lambda r: (r["mode"], int(r["run"][3:]))):
        mode, run = result["mode"], result["run"]
        tag = f"{mode}_{run}"
        index = int(run[3:]) - 1
        phase = "fill" if index == 0 else "restart_hit"
        if result.get("phase", phase) != phase or result.get("restart_index", index) != index:
            errors.append(f"{tag}: phase/restart_index disagrees with run number")
        if result.get("status") != "complete" or not result.get("final_answer", "").strip():
            errors.append(f"{tag}: final-answer workflow did not complete")
        tools = read_json(folder / tag / "tool_calls.json")
        calls = list(read_json(folder / tag / "llm_calls.json").values())
        requests = result["cache_requests"]
        n = result["llm_calls"]
        if n <= 0 or len(calls) != n or len(requests) != n or result["ttft_count"] != n:
            errors.append(f"{tag}: callback/cache request/TTFT metric counts differ")
        if len(tools) != result["python_calls"]:
            errors.append(f"{tag}: Python tool-call count differs")
        if not requests:
            errors.append(f"{tag}: no cache request records")
            continue
        request_ids = [q["request_id"] for q in requests]
        if len(set(request_ids)) != len(request_ids):
            errors.append(f"{tag}: duplicated request IDs")
        for q in requests:
            prompt, hit, load = q["prompt_tokens"], q["lmcache_hit_tokens"], q["load_tokens"]
            if prompt <= 0 or not 0 <= load <= hit <= prompt or hit % chunk:
                errors.append(f"{tag}: invalid token/cache counts for {q['request_id']}")
        first = requests[0]
        expected_first_hit = ((first["prompt_tokens"] - 1) // chunk) * chunk
        if phase == "fill" and first["lmcache_hit_tokens"] != 0:
            errors.append(f"{tag}: fresh namespace's first request unexpectedly hit")
        if phase == "restart_hit" and (expected_first_hit <= 0 or
                first["lmcache_hit_tokens"] < expected_first_hit or first["load_tokens"] < expected_first_hit):
            errors.append(f"{tag}: restart did not load the expected persisted prefix ({expected_first_hit} tokens)")
        for field in ("workflow_seconds", "mean_server_ttft_ms"):
            if not math.isfinite(result[field]) or result[field] <= 0:
                errors.append(f"{tag}: invalid {field}")
        audit = audit_log(folder, tag)
        audits.append(audit)
        if audit["errors_before_shutdown"]:
            errors.append(f"{tag}: server errors before shutdown require review")
        if not audit["shutdown_marker_found"]:
            errors.append(f"{tag}: shutdown marker missing; run/shutdown errors cannot be separated")
        prompts = sum(q["prompt_tokens"] for q in requests)
        hits = sum(q["lmcache_hit_tokens"] for q in requests)
        usage = [c.get("llm_output", {}).get("token_usage", {}) for c in calls]
        if any("completion_tokens" not in u or "prompt_tokens" not in u for u in usage):
            errors.append(f"{tag}: callback token usage is missing")
        elif sum(u["prompt_tokens"] for u in usage) != prompts:
            errors.append(f"{tag}: callback and cache-log prompt token totals differ")
        row = {k: result[k] for k in ("mode", "run", "workflow_seconds", "llm_calls", "python_calls", "mean_server_ttft_ms", "ttft_count")}
        row.update(phase=phase, restart_index=index, prompt_tokens_sum=prompts,
                   completion_tokens_sum=sum(u.get("completion_tokens", 0) for u in usage),
                   lmcache_hit_tokens_sum=hits, load_tokens_sum=sum(q["load_tokens"] for q in requests),
                   hit_ratio=hits / prompts if prompts else 0,
                   first_request_hit_tokens=first["lmcache_hit_tokens"],
                   first_request_load_tokens=first["load_tokens"],
                   expected_first_request_min_hit_tokens=expected_first_hit if phase == "restart_hit" else 0,
                   python_seconds=sum(x["elapsed_seconds"] for x in tools),
                   llm_seconds=sum(x["elapsed_seconds"] for x in calls))
        row["execution_sequence"] = result.get("execution_sequence", "")
        rows.append(row)
        artifacts[tag] = {"tools": [(x["code"], x["output"]) for x in tools],
                          "final_answer": result["final_answer"],
                          "generated_text": [c.get("generations") for c in calls]}
    validation = {
        "passed": not errors, "errors": errors, "restart_runs_requested_per_mode": restart_runs,
        "expected_total_workflows": len(expected), "observed_total_workflows": len(results),
        "scope": "One fresh fill per mode, then repeated client restarts using retained DAOS cache; no scientific grading",
    }
    return manifest, rows, artifacts, audits, validation


def compare_artifacts(left_tag, right_tag, artifacts, comparison):
    left, right = artifacts[left_tag], artifacts[right_tag]
    return {"comparison": comparison, "left": left_tag, "right": right_tag,
            "code_and_observations_equal": left["tools"] == right["tools"],
            "final_answers_equal": left["final_answer"] == right["final_answer"],
            "all_model_generations_equal": left["generated_text"] == right["generated_text"]}


def equivalence(artifacts, restart_runs):
    comparisons = [compare_artifacts(f"dfs_run{i}", f"object_run{i}", artifacts, "same_index_between_modes")
                   for i in range(1, restart_runs + 2)]
    for mode in MODES:
        for i in range(2, restart_runs + 2):
            comparisons.append(compare_artifacts(f"{mode}_run1", f"{mode}_run{i}", artifacts, "fill_vs_restart"))
        for i in range(3, restart_runs + 2):
            comparisons.append(compare_artifacts(f"{mode}_run2", f"{mode}_run{i}", artifacts, "first_restart_vs_later_restart"))
    return comparisons


def aggregate(rows):
    statistics_rows = []
    for mode in MODES:
        for phase in PHASES:
            group = [r for r in rows if r["mode"] == mode and r["phase"] == phase]
            if not group:
                continue
            result = {"mode": mode, "phase": phase, "n": len(group)}
            for field in ("workflow_seconds", "mean_server_ttft_ms"):
                result.update({f"{field}_{key}": value for key, value in describe([r[field] for r in group]).items() if key != "n"})
            result["server_ttft_call_weighted_mean_ms"] = sum(r["mean_server_ttft_ms"] * r["ttft_count"] for r in group) / sum(r["ttft_count"] for r in group)
            result["llm_calls_distribution"] = distribution(r["llm_calls"] for r in group)
            result["python_calls_distribution"] = distribution(r["python_calls"] for r in group)
            result["first_request_hit_tokens_distribution"] = distribution(r["first_request_hit_tokens"] for r in group)
            result["completion_tokens_distribution"] = distribution(r["completion_tokens_sum"] for r in group)
            statistics_rows.append(result)
    return statistics_rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                         for key, value in row.items()} for row in rows)


def fmt_sd(value):
    return "—" if value is None else f"{value:.2f}"


def report(manifest, rows, statistics_rows, comparisons, audits):
    args = manifest["args"]
    restarts = int(args.get("restart_runs", 1))
    lines = ["# DiscoveryBench full agentic 반복 실행 결과", "",
             f"DFS와 object 각각 최초 캐시 채우기 1회와 vLLM 재시작 후 동일 문제 재풀이 {restarts}회를 수행했다. "
             f"총 {len(rows)}회의 에이전트 workflow이며, **{restarts}회는 오류 자동 재시도가 아니라 같은 문제를 다시 푸는 횟수**다. "
             "DiscoveryBench 전체 테스트셋이나 정답 채점 결과는 아니다.", "",
             "## 실행 조건과 집계 기준", "",
             f"- 모델 `{args['model']}`, 청크 {args['chunk_size']}토큰, GPU staging {args['gpu_buffer_gb']}GiB, "
             f"I/O·메타 작업자 {args['io_workers']}·{args['meta_workers']}개.",
             "- 실제 CSV를 읽고 Python 분석 코드를 실행한 뒤 최종 답변을 작성하는 동일 태스크다.",
             "- 모드별 새 namespace로 시작한다. 이후 vLLM만 매번 재시작하고 DAOS 데이터·서버 캐시는 유지한다.",
             ("- 같은 재시작 번호의 DFS/object를 교차 실행하고, 매 단계 두 모드의 선후 순서를 뒤집었다."
              if args.get("interleave_modes") else "- 모드별 순차 실행 순서는 manifest의 schedule 또는 원본 실행 기록을 참조한다."),
             "- 최초 `fill` 1회는 참고값이며, 아래 `restart_hit` 반복 통계에 포함하지 않는다.",
             "- 전체 작업 시간: 에이전트 시작부터 최종 답변까지. 모델·Python 컨테이너 기동은 제외하고 도구 실행·로깅은 포함한다.",
             "- TTFT: 각 실행 내 모델 호출들의 **서버 측 평균 TTFT**다. 그 실행별 평균들을 다시 비가중 평균·중앙값으로 집계한다. "
             "개별 호출의 p50/p95나 클라이언트 첫 텍스트 TTFT가 아니다.",
             "- 표준편차는 표본 표준편차다. 최초 1회의 표준편차는 계산할 수 없어 —로 표시한다.", "",
             "## 전체 작업 시간 (초)", "",
             "| 모드 | 단계 | 횟수 | 평균 | 중앙값 | 표준편차 | 최솟값 | 최댓값 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in statistics_rows:
        key = "workflow_seconds_"
        lines.append(f"| {item['mode']} | {item['phase']} | {item['n']} | {item[key+'mean']:.3f} | "
                     f"{item[key+'median']:.3f} | {fmt_sd(item[key+'stdev'])} | {item[key+'min']:.3f} | {item[key+'max']:.3f} |")
    lines += ["", "## 실행별 평균 서버 TTFT의 분포 (ms)", "",
              "| 모드 | 단계 | 횟수 | 평균 | 중앙값 | 표준편차 | 최솟값 | 최댓값 | 호출수 가중 평균 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in statistics_rows:
        key = "mean_server_ttft_ms_"
        lines.append(f"| {item['mode']} | {item['phase']} | {item['n']} | {item[key+'mean']:.2f} | "
                     f"{item[key+'median']:.2f} | {fmt_sd(item[key+'stdev'])} | {item[key+'min']:.2f} | "
                     f"{item[key+'max']:.2f} | {item['server_ttft_call_weighted_mean_ms']:.2f} |")
    lines += ["", "## 수행 작업 횟수 분포", "",
              "`호출 수: 실행 횟수` 형식이다. 예: `3: 10`은 모델을 3번 호출한 workflow가 10회라는 뜻이다.", "",
              "| 모드 | 단계 | 모델 호출 | Python 실행 | 첫 요청 hit 토큰 |",
              "|---|---|---|---|---|"]
    for item in statistics_rows:
        cells = [", ".join(f"{key}: {value}" for key, value in item[field].items()) for field in
                 ("llm_calls_distribution", "python_calls_distribution", "first_request_hit_tokens_distribution")]
        lines.append(f"| {item['mode']} | {item['phase']} | {' | '.join(cells)} |")
    lines += ["", "## 개별 실행", "",
              "| 모드 | 실행 | 단계 | 전체 시간 (s) | 평균 TTFT (ms) | 모델 호출 | Python 실행 | 생성 토큰 합계 | 첫 요청 hit |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        tag = f"{row['mode']}_{row['run']}"
        lines.append(f"| {row['mode']} | [{row['run']}]({tag}/llm_calls.json) | {row['phase']} | {row['workflow_seconds']:.3f} | "
                     f"{row['mean_server_ttft_ms']:.2f} | {row['llm_calls']} | {row['python_calls']} | "
                     f"{row['completion_tokens_sum']} | {row['first_request_hit_tokens']} |")
    between = [x for x in comparisons if x["comparison"] == "same_index_between_modes"]
    same_work = sum(x["code_and_observations_equal"] for x in between)
    same_answers = sum(x["final_answers_equal"] for x in between)
    warnings = sum(len(x["warnings_before_shutdown"]) for x in audits)
    shutdown_errors = sum(len(x["errors_after_shutdown"]) for x in audits)
    lines += ["", "## 검증과 해석 범위", "",
              "- 실행 횟수, 모델 콜백/캐시 로그/TTFT 카운트, Python 실행 횟수, 최초 miss와 재시작 후 첫 요청 prefix load를 검증했다.",
              f"- 동일 실행 번호의 DFS/object {len(between)}쌍 중 코드·실행 결과 동일 {same_work}쌍, 최종 답변 동일 {same_answers}쌍. "
              "단계 사이 동일성도 `mode_equivalence.json`에 기록했다. 출력 차이는 자동 실패로 처리하지 않았다.",
              f"- 종료 전 경고 {warnings}줄, 종료 후 오류 {shutdown_errors}줄을 별도로 기록했다. 종료 전 오류가 있으면 정상 성능 보고서 생성을 거부한다.",
              f"- 캐시가 반복마다 누적된다. {restarts}번의 독립적인 새 cold/warm 쌍 실험이 아니며, 뒤쪽 실행은 앞선 재풀이가 만든 추가 prefix도 재사용할 수 있다.",
              "- 최초 실행도 뒤쪽 모델 호출에서는 KV를 재사용할 수 있어 workflow 전체가 miss-only는 아니다.",
              "- 에이전트가 선택하는 분석 횟수·코드·출력 길이가 달라질 수 있다. 최초/재시작 또는 DFS/object의 전체 시간 차이를 모두 I/O나 캐시 효과로 해석하지 않는다.",
              "- 같은 문제를 여러 번 푼 결과이며 여러 태스크·동시 사용자 부하를 대표하지 않는다. workflow 완료는 과학적 답변 정확성의 검증이 아니다.",
              "- 문서의 태스크 데이터는 stress_tolerance가 일정해 관계 분석이 조기에 끝날 수 있다. 실제 수행한 코드와 최종 답변을 함께 확인해야 한다.", "",
              "[개별 실행 CSV](full_agentic_repeated_runs.csv) · [단계별 통계 CSV](full_agentic_repeated_statistics.csv) · "
              "[검증](repeated_validation.json) · [작업 동일성](mode_equivalence.json) · [로그 감사](log_audit.json) · [원자료](summary.json)", ""]
    order_flag = " --interleave-modes" if args.get("interleave_modes") else ""
    lines += ["## 재실행", "", "```bash", "cd /root/discos_minji",
              f"./venv/bin/python3 full_agent_bench.py --chunk-size {args['chunk_size']} "
              f"--restart-runs {restarts}{order_flag} \\",
              "  --output /root/discos_minji/full_agentic_새_결과_폴더",
              "./venv/bin/python3 summarize_full_agent_repeated.py \\",
              "  /root/discos_minji/full_agentic_새_결과_폴더", "```", "",
              "이미 존재하는 결과 폴더는 덮어쓰지 않는다. 새 실행마다 새 캐시 namespace가 생성되며, "
              "공유 DAOS 컨테이너나 이전 결과는 삭제하지 않는다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    manifest, rows, artifacts, audits, validation = collect(args.folder)
    write_json(args.folder / "log_audit.json", audits)
    write_json(args.folder / "repeated_validation.json", validation)
    if not validation["passed"]:
        raise ValueError("Repeated-agentic validation failed: " + "; ".join(validation["errors"]))
    comparisons = equivalence(artifacts, validation["restart_runs_requested_per_mode"])
    statistics_rows = aggregate(rows)
    write_json(args.folder / "mode_equivalence.json", comparisons)
    write_json(args.folder / "full_agentic_repeated_statistics.json", statistics_rows)
    write_csv(args.folder / "full_agentic_repeated_runs.csv", rows)
    write_csv(args.folder / "full_agentic_repeated_statistics.csv", statistics_rows)
    path = args.folder / "REPEATED_AGENTIC_RESULT_KO.md"
    path.write_text(report(manifest, rows, statistics_rows, comparisons, audits))
    print(f"Validated {len(rows)} full-agentic runs; report: {path}")


if __name__ == "__main__":
    main()
