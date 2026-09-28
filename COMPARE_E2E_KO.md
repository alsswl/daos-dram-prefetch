# DFS / 오브젝트 직접 접근 E2E 비교 실행

`run_compare.sh`는 두 저장 구현을 같은 고정 입력으로 번갈아 실행한다. 빌드 옵션을 바꾸는 실험이 아니라, 같은 빌드에서 실행별 YAML과 `DAOSGDS_TRANSPORT`를 바꾸는 실험이다. `/root/discos`와 기본 YAML은 수정하지 않는다.

## 실행

```bash
cd /root/discos_minji

# 설정·실행 계획만 생성: GPU와 DAOS에 접근하지 않음
./run_compare.sh --dry-run

# 짧은 동작 확인: 두 모드 각각 저장 및 재시작, 요청 2개씩
./run_compare.sh --repeats 1 --steps 2 --max-tokens 8

# 기본 비교: 두 모드 × 3회 반복, 패스별 요청 6개, 출력 64토큰
./run_compare.sh --repeats 3 --steps 6 --max-tokens 64

# 글루시스 문서 Part A 길이에 맞춘 비교: --steps 대신 고정 길이 입력 사용
./run_compare.sh --repeats 3 --context-tokens 8192 16384 31744 \
  --chunk-size 256 --max-model-len 32768 --max-tokens 64 \
  --gpu-buffer-gb 10 --io-workers 16 --meta-workers 16
```

실제 실행은 GPU를 사용하고 DAOS에 실험용 캐시를 저장한다. 다른 GPU 작업이 없는 상태에서 실행한다. 기본 HTTP 포트는 `127.0.0.1:8017`이다. 포트가 사용 중이면 기존 서버를 중단하지 않고 실패한다. 필요하면 `--port 8018`을 지정한다. 오브젝트 shim은 빌드되어 있어야 한다(`make -B all && make check`). 스크립트는 서버·클라이언트를 재빌드하지 않는다.

결과 위치는 시작 시 `Results: ...`로 표시한다. 기본은 `/root/discos_minji/comparison_<날짜-시각-UUID>/`이며 `--output /root/discos_minji/원하는_새_디렉터리`로 지정할 수 있다. 기존 디렉터리는 덮어쓰지 않는다.

## 자동 실행 순서

각 반복에서 모드별로 다음을 실행한다. 홀수 반복은 DFS → object, 짝수 반복은 object → DFS다.

1. 해당 모드·반복 전용의 새 DFS 디렉터리 또는 오브젝트 dkey 접두사를 만든다.
2. vLLM을 시작하고 측정 입력과 다른 prefix로 모델·DAOS 워밍업을 한다. 워밍업 캐시 hit까지 확인한다.
3. `fill`: 측정 입력을 재생하며 캐시를 채운다. 순차 실행의 첫 요청은 miss여야 한다. 뒤 요청은 공통 prefix를 재사용할 수 있다.
4. 측정 밖에서 읽기 요청으로 캐시 준비를 확인하고 해당 vLLM 프로세스를 종료한다.
5. 같은 캐시 영역으로 새 vLLM 프로세스를 시작하고 공통 워밍업을 한다.
6. `restart_hit`: 같은 측정 입력을 재생하여 프로세스 재시작 후 DAOS 캐시 재사용을 측정한다.
7. `same_process_hit`: 같은 프로세스에서 다시 재생한다. vLLM 내장 prefix caching은 꺼져 있으므로 이때도 DAOS 경로를 사용한다.

`fill` 전체를 cold 결과로 부르면 안 된다. 공유 prefix 때문에 miss와 hit가 섞인다. 순차 실행의 순수 첫 miss는 요청별 CSV의 첫 행을 본다. `restart_hit`도 DAOS 서버 캐시를 비운 disk-cold 실험이 아니다. 새 클라이언트에서 서버에 남아 있는 KV를 읽는 실험이다.

단, `--context-tokens` 모드는 첫 청크부터 서로 다른 고정 입력을 사용하므로 fill의 모든 요청이 miss인지 검사한다. 8K·16K·31K를 한 번씩 보내는 패스가 생성되며 `--steps`는 사용하지 않는다. 입력 내용은 합성 반복 문장이므로 원문 벤치마크와 정확히 같은 텍스트는 아니다.

## 맞추는 조건과 확인 범위

- 동일 모델(Qwen3-14B 기본), 고정 토큰 ID 입력, 생성 길이, seed, eager 실행, GPU 메모리 비율 0.75.
- 동일 청크(2048), GPU staging(8GiB), I/O·메타데이터 작업자(각 8개), 비동기 로딩, multi-prefetch 활성화.
- 동일 `/opt/daos-gds-gpu` 및 `/opt/ofi-cuda` 라이브러리. 워커의 실제 로드 경로도 확인·저장한다.
- 동일 pool/container. POSIX·HEALTHY·`rd_fac=0`인지 조회하고 아니면 중단한다.
- DFS 신규 데이터 파일도 `OC_SX`로 생성해 직접 오브젝트의 클래스와 맞춘다. DFS 디렉터리 메타데이터 구조나 실제 물리 배치까지 같다는 뜻은 아니다.
- DFS 전용 기동 프로브(`DAOS_PROBE_CHUNKS=0`)와 object 전용 C 타이밍 출력(`DAOSGDR_TIMING=0`)을 끈다.

실험 목표는 **DFS 기반 구현과 직접 오브젝트 기반 구현 전체 비교**다. 파일/키/메타데이터 구조, API 호출 수, 데이터 분할 등의 차이는 남는다. 서버의 다른 부하나 물리 배치는 이 스크립트가 통제하지 않는다.

## 결과 읽기

| 파일 | 내용 |
|---|---|
| `status.json` | 전체 완료/실패, 캐시 검사와 출력 텍스트 일치 여부 |
| `summary.csv`, `summary.json` | 반복·모드·패스별 TTFT/E2E 평균·중앙값·p95·표준편차, 요청/s, 출력 토큰/s |
| `r1_dfs_restart_hit.csv` 등 | 요청별 입력/출력 토큰 수, cached tokens, TTFT, E2E, 입력·출력 해시 |
| `*_responses.json` | 생성된 텍스트 포함 상세 결과 |
| `*_server.log` | vLLM/LMCache 서버 로그 |
| `*_metrics_before.txt`, `*_metrics_after.txt` | 측정 패스 전후 서버 metrics 원문 |
| `*_native_maps.json` | 서버·워커에서 실제 로드한 DAOS/Mercury/libfabric 경로 |
| `manifest.json`, `workload.json`, `*.yaml` | 실행 순서·명령·소스 해시, 실제 입력 토큰 ID, 개별 설정 |
| `container_properties.txt` | 실행 전 컨테이너 속성 |
| `output_text_checks.json` | 같은 반복의 DFS fill 출력과 각 모드·패스 출력의 일치 여부 |

TTFT는 HTTP 요청 시작부터 **첫 비어 있지 않은 응답 텍스트** 도착까지다. 내부의 정확한 첫 토큰 생성 시각과는 다를 수 있다. E2E는 요청 시작부터 스트림 완료까지이며 로컬 HTTP 오버헤드를 포함한다. 모델 기동, 사전 워밍업, 저장 확인 요청은 측정에서 제외한다. 출력 토큰/s는 패스의 총 출력 토큰 수 / 전체 패스 시간으로, 순수 decode 속도는 아니다.

후속 버전은 `return_token_ids`도 요청해 `first_token_ms`(첫 토큰 ID가 도착하기까지)와 출력 토큰 ID를 기록한다. `status.json`의 `all_output_tokens_equal`로 모드·패스 간 토큰 단위 일치를 확인한다. 이는 텍스트 일치보다 직접적인 검사지만 GPU KV 바이트 전체 검증을 대신하지 않는다.

길이가 여러 개인 실험에서는 `summary.csv`의 전체 중앙값을 문서의 길이별 수치와 비교하면 안 된다. 실행 완료 후 아래처럼 **문맥 길이별** 보고서를 생성한다.

```bash
./venv/bin/python3 summarize_reference.py /root/discos_minji/실제_결과_디렉터리
```

생성되는 `by_context.csv`와 `REFERENCE_COMPARISON_KO.md`에서 문서 §6.2 표 2의 GDS cold-hit TTFT(8K 76~149ms, 16K 118~132ms, 31K 198~218ms)와 비교한다. 원문 TTFT를 우리 전체 응답 완료 시간과 직접 비교하지 않는다. vLLM 버전·입력 내용·원본 launcher 부재 등 남은 차이도 보고서에 명시한다.

`restart_hit`와 `same_process_hit`는 모든 요청에서 예상한 완전한 청크 prefix 이상이 cached로 보고되어야 한다. 토큰 수가 바뀌거나 hit가 부족하면 실패로 기록하고 중단한다. 생성 텍스트 일치는 별도 진단이며 GPU KV 전체 바이트 동일성을 검증하는 테스트는 아니다. 출력 불일치가 있으면 지연시간만 보고 결론 내리지 말고 원인을 확인한다.

반복별 결과가 따로 저장된다. 같은 패스·같은 요청 조건끼리 비교하고 반복 간 편차도 확인한다. 기본 요청 6개의 p95는 표본이 작으며, 1회 축약 실행은 스크립트 검증이지 최종 성능 결론이 아니다.

## 동시 요청과 보관

`--concurrency 2` 등으로 겹친 요청을 보낼 수 있다. 기본은 1이며 순차 실행이다. 동시 실행 시 fill의 첫 제출 요청이 반드시 첫 miss는 아니므로, 적어도 하나의 miss를 확인한다. 같은 prefix를 공유하는 고정 입력이므로 다양한 독립 사용자 부하를 대표하지 않는다. 높은 동시성에서 GPU staging 부족·취소·다중 요청 경합은 별도 검증 대상이다.

각 실행의 실험용 캐시는 **자동 삭제하지 않는다**. 공유 컨테이너 전체를 지우는 명령도 실행하지 않는다. 새 namespace는 논리적 cold 상태만 만들고 서버 캐시나 장치를 초기화하지 않는다. 반복 실행 시 저장 공간이 누적되며 정확한 실험 영역은 생성된 YAML에서 확인할 수 있다.

이 워크로드는 DiscoveryBench 스키마와 고정된 단계별 히스토리를 사용한다. 생성 코드 실행·도구 호출·과학적 정답 채점은 하지 않으므로 full-agentic DiscoveryBench가 아니라 **LLM 요청 단위 E2E 비교**다.

## 스크립트 검증 기록

2026-09-16, `--repeats 1 --steps 2 --max-tokens 8`로 Qwen3-14B 실제 추론을 실행했다. 두 모드의 fill·restart_hit·same_process_hit가 모두 완료됐고, 재시작 후 두 요청 각각 2048토큰을 재사용했다. 모드·패스 간 생성 텍스트가 일치했다. [검증 상태](comparison_smoke_20260916_v1/status.json)와 [요약 CSV](comparison_smoke_20260916_v1/summary.csv)를 보관했다. 이는 단일 짧은 실행이며 기본 3회 반복 성능 실험이나 동시 요청 검증은 아니다.

단위 테스트는 기존 17개와 드라이버 테스트 6개를 합쳐 23개 통과했다. 서버 종료 로그에는 vLLM 프로세스 정리 및 semaphore 누수 경고가 남았으며, 종료 후 GPU 사용량이 기동 전의 4MiB로 돌아온 것을 확인했다.

후속 Part A 길이별 3회 비교도 완료했다. [결과 보고서](gluesys_reference_20260916_v1/REFERENCE_COMPARISON_KO.md)에 TTFT·전체 응답 시간·문서와의 차이를 기록했다. 총 54개 측정 요청의 캐시 및 출력 토큰 ID 검증을 통과했고, 단위 테스트는 24개 통과했다. 실행 당시 코드는 결과 디렉터리의 `compare_e2e_executed.py`에 보관했다. 측정 이후 현재 드라이버의 청크 경계 hit 검사를 더 엄격하게 보완했으며, 완료 결과도 같은 엄격한 기준으로 `summarize_reference.py`에서 재검증했다. 워밍업용 중간 토큰열의 불필요한 길이 경고도 줄였다. 실제 전송한 워밍업 입력은 동일하다.
