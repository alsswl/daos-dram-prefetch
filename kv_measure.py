#!/usr/bin/env python3
"""
kv_measure.py — DiscoveryBench 태스크로 KV 재사용을 '정량 측정'한다.

kv_probe.py 의 확장판. 차이점:
  - 컬럼 수(프리픽스 길이)가 다른 여러 태스크를 자동으로 돌린다.
  - LMCache 로그를 파싱하는 대신, vLLM 출력의 토큰 수를 직접 집계해
    스텝별 hit / 프리픽스 재사용률 / 청크 경계 손실을 계산한다.
  - 결과를 콘솔 표 + CSV 로 저장한다 → "프리픽스 길수록 재사용 이득 큰가" 곡선용.

측정 방식(중요):
  vLLM 의 RequestOutput.num_cached_tokens 를 스텝마다 읽는다.
  이 값이 "재사용된(재계산 안 한) 토큰 수". step2+ 에서 이게 프리픽스만큼 나오면
  프리픽스 재사용이 확인된 것.

  * vLLM 내장 prefix caching 을 끄고(--no-prefix-cache) 돌리면
    순수 LMCache 효과만 본다. 켜면 GPU 캐시 + LMCache 합산.

실행:
  LMCACHE_CONFIG_FILE=~/kyu/lmcache_config.yaml PYTHONHASHSEED=0 \
    python kv_measure.py \
      --root ~/kyu/discoverybench/discoverybench/synth/train \
      --tasks 12 --steps 4 --out ~/kyu/kv_measure_results.csv
"""

import os
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("LMCACHE_CONFIG_FILE",
                      os.path.expanduser("~/kyu/lmcache_config.yaml"))
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"    # ← FlashInfer sampling 끄기

import json
import glob
import csv
import argparse


# ── 태스크 로딩 ─────────────────────────────────────────────────
def load_task(task_dir):
    meta_path = os.path.join(task_dir, "metadata_0.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    datasets = meta.get("datasets", [])
    if not datasets or not datasets[0].get("columns"):
        return None
    ds = datasets[0]
    lines = [f"Dataset: {ds.get('name','data.csv')}",
             f"Description: {ds.get('description','')}", "Columns:"]
    for c in ds["columns"]:
        lines.append(f"  - {c['name']}: {c.get('description','')}")
    schema_text = "\n".join(lines)
    queries = meta.get("queries", [])
    if not queries:
        return None
    return {"dir": task_dir, "domain": meta.get("domain", "?"),
            "goal": queries[0].get("question", ""),
            "schema": schema_text, "n_cols": len(ds["columns"])}


def make_pad_text(pad_tokens):
    """pad_tokens 개 만큼(근사치)의 고정 더미 텍스트를 만든다.
    task/step 무관하게 항상 동일한 문자열 → 프리픽스로 재사용 가능."""
    if pad_tokens <= 0:
        return ""
    unit = "CONTEXT: this is filler context used to pad the prefix for KV cache size benchmarking. "
    chars_per_token = 4  # 근사치, 정확할 필요 없음
    target_chars = pad_tokens * chars_per_token
    reps = target_chars // len(unit) + 1
    return (unit * reps)[:target_chars] + "\n\n"


def build_prompt(task, history, pad_text=""):
    system = ("You are a data-driven scientific discovery agent. "
              "You are given a dataset schema and a discovery goal. "
              "At each step you reason about what analysis to run next, "
              "and you may write a short snippet of Python. "
              "Think step by step and be concise.\n")
    prefix = (f"{pad_text}{system}\n=== DATASET SCHEMA ===\n{task['schema']}\n\n"
              f"=== DISCOVERY GOAL ===\n{task['goal']}\n\n=== WORK LOG ===\n")
    hist = ""
    for i, h in enumerate(history, 1):
        hist += f"[Step {i}] {h}\n"
    hist += f"[Step {len(history)+1}] Your next reasoning and code:\n"
    return prefix + hist


def get_cached(out):
    """RequestOutput 에서 재사용된 토큰 수를 최대한 견고하게 뽑는다.
    vLLM 버전에 따라 위치가 달라 여러 경로를 시도."""
    for attr in ("num_cached_tokens",):
        v = getattr(out, attr, None)
        if v is not None:
            return int(v)
    metrics = getattr(out, "metrics", None)
    if metrics is not None:
        v = getattr(metrics, "num_cached_tokens", None)
        if v is not None:
            return int(v)
    return None  # 못 찾으면 None (로그로 대체 확인)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="LMCache chunk_size (경계 손실 계산용, config와 일치시킬 것)")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="vLLM 내장 prefix caching 끄기 → 순수 LMCache 효과만")
    ap.add_argument("--pad-tokens", type=int, default=0,
                    help="프리픽스 맨 앞에 붙일 고정 더미 텍스트 크기(토큰 근사치). "
                         "0이면 더미 없음(기존과 동일). GDS 크기별 벤치마크용.")
    ap.add_argument("--out", default=os.path.expanduser("~/kyu/kv_measure_results.csv"))
    args = ap.parse_args()

    pad_text = make_pad_text(args.pad_tokens)

    root = os.path.expanduser(args.root)
    tasks = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        t = load_task(d)
        if t:
            tasks.append(t)
        if len(tasks) >= args.tasks:
            break
    if not tasks:
        print(f"[!] {root} 에서 유효 태스크 없음")
        return

    # 프리픽스 길이(컬럼 수) 순으로 정렬 → 곡선이 보기 좋게
    tasks.sort(key=lambda t: t["n_cols"])
    print(f"[i] 태스크 {len(tasks)}개 (컬럼 {tasks[0]['n_cols']}~{tasks[-1]['n_cols']})")

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm_kwargs = dict(model=args.model, max_model_len=args.max_model_len,
                      gpu_memory_utilization=0.75, enforce_eager=True,
                      kv_transfer_config=ktc)
    if args.no_prefix_cache:
        llm_kwargs["enable_prefix_caching"] = False
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    rows = []
    for t in tasks:
        history = []
        # step1 프롬프트 토큰 수 = 대략적 프리픽스 길이 지표
        for step in range(args.steps):
            prompt = build_prompt(t, history, pad_text)
            out = llm.generate([prompt], sp)[0]
            prompt_tokens = len(out.prompt_token_ids)
            cached = get_cached(out)
            gen = out.outputs[0].text.strip().replace("\n", " ")
            history.append(gen[:120])
            rows.append({
                "domain": t["domain"],
                "n_cols": t["n_cols"],
                "step": step + 1,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached if cached is not None else -1,
                "reuse_ratio": round(cached / prompt_tokens, 3)
                               if (cached and prompt_tokens) else 0.0,
            })

    # ── CSV 저장 ────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── 콘솔 표 ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'domain':<26}{'cols':>5}{'step':>5}{'prompt_tok':>11}"
          f"{'cached':>8}{'reuse%':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['domain'][:25]:<26}{r['n_cols']:>5}{r['step']:>5}"
              f"{r['prompt_tokens']:>11}{r['cached_tokens']:>8}"
              f"{r['reuse_ratio']*100:>7.1f}")
    print("=" * 78)

    # ── 태스크별 요약 (step2+ 평균 재사용) ─────────────────────
    print("\n[프리픽스 길이 vs 재사용] step2+ 평균 cached_tokens:")
    print(f"{'cols':>5}  {'domain':<26}{'avg_cached(step2+)':>20}")
    by_task = {}
    for r in rows:
        by_task.setdefault((r["n_cols"], r["domain"]), []).append(r)
    for (ncol, dom), rs in sorted(by_task.items()):
        s2 = [x["cached_tokens"] for x in rs if x["step"] >= 2 and x["cached_tokens"] >= 0]
        avg = sum(s2) / len(s2) if s2 else 0
        print(f"{ncol:>5}  {dom[:25]:<26}{avg:>20.0f}")

    print(f"\n[i] CSV 저장: {args.out}")
    if any(r["cached_tokens"] < 0 for r in rows):
        print("[!] cached_tokens=-1 인 행이 있음 → 이 vLLM 버전은 num_cached_tokens를 "
              "RequestOutput에 노출하지 않음. LMCache 로그의 'hit tokens'로 대체 확인 필요.")


if __name__ == "__main__":
    main()

