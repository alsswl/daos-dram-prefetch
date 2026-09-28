# DiscoveryBench 장시간 staging 관측 방법

본 실행: `discovery_staging_long_20260926_v2`. 실행 중 자료와 완료 결과를 구분한다.

## 목적과 범위

이전 16요청 고정 KV 재사용 실험과 달리, 실제 데이터 분석 문제의 모델 호출·Python 실행·누적 대화·신규 KV 저장을 포함한다. 과학적 답변의 정답 채점이나 DiscoveryBench 전체 테스트셋 실행은 아니다.

- 로컬 DiscoveryBench real 메타데이터 158개 중 seed 926으로 분야별 순환 선정한 32개에서 각각 첫 query를 순환 실행한다. train/test를 함께 포함하며 생물학·경제학 각 6개, 공학·인문학·메타과학·사회학 각 5개다. 문제 목록과 입력 파일 해시는 `tasks.json`에 보존한다.
- 모델은 Qwen3-14B BF16, H100 NVL 한 개, vLLM 한 프로세스다. 컨텍스트 상한 32768, 호출당 출력 상한 2048, 에이전트당 최대 25단계, temperature 0이다. 기존 로컬 에이전트 프롬프트를 사용하고 padding하지 않는다. 로컬 `react_agent.py`에는 `/no_think`와 단계별 분석 지시가 이미 포함되어 있으므로 공식 upstream의 원본 프롬프트와 완전히 동일한 재현이라고 주장하지 않는다. 실행 계획의 original prompts는 이 기존 로컬 프롬프트를 뜻한다.
- `chat_template_kwargs.enable_thinking=false`를 명시한다. 원래 예시에 포함된 `FINAL_ANSWER:`도 종료 답변으로 인정하는 옵션을 추가했다. 기존 `run_full_agent.py`의 기본 동작은 유지하고 이 실행에만 옵션을 전달한다.
- 원래 ReAct 에이전트와 Python 도구를 사용한다. 도구는 네트워크 차단·읽기 전용 루트·권한 제거 컨테이너 안에서 동작한다. 데이터는 읽기 전용이며 각 작업 디렉터리만 쓰기 가능하다. 누락된 matplotlib/seaborn 등의 패키지는 실험 전용 폴더를 읽기 전용으로 추가 마운트했다. 원래 가상환경은 수정하지 않았다.
- object GPU-direct 모드, chunk 128, I/O/meta workers 각 16, async loading ON, vLLM prefix caching OFF.
- CPU allocator/캐시 8GiB, GPU staging 10GiB, DRAM 프리페치 ON의 기존 공유 풀 입장 기준 5GiB. 동적 예산 정책을 새로 적용한 실험은 아니다.

## DRAM 크기 선택

수정된 예비 실행 `discovery_staging_pilot_20260926_v2`의 조회 청크 기준 비중은 DRAM 22.76%, DAOS 68.68%, miss 8.56%였다. 이를 보고 8GiB를 선택했다. 이 값은 초기와 후반을 합친 비율로, 시간이 지나도 혼합 비율이 유지된다는 뜻은 아니다. 적정 운영 용량이나 최적값을 찾은 것도 아니다.

Qwen3-14B BF16 KV는 토큰당 160KiB이므로 128토큰 청크의 payload는 20MiB다. DRAM 8GiB는 약 409청크(52,352토큰 상당), GPU staging 10GiB는 512청크(65,536토큰 상당)의 payload 규모다. 이는 여러 요청이 공유하는 공간이며 모델의 1요청 컨텍스트 한도와는 별개다. 실행 중 DRAM에는 최대 409청크가 관측되었다.

예비 v1은 thinking 태그로 인한 파싱 실패가 있었고, 예비 v2도 FINAL_ANSWER 표기 불일치가 있었다. 둘 모두 본 성능 비교에서 제외한다. 본 long v1은 포트 재사용 사전 점검에서 중단됐으며 모델 부하는 실행하지 않았다. 포트 점검에 SO_REUSEADDR를 적용한 long v2가 본 실행이다. 모든 실패 기록은 보존한다.

## 부하

프리페치 OFF와 ON을 새 프로세스·새 namespace에서 각각 실행한다. 각 조건에서 캐시는 단계 사이에 유지한다.

1. 동시 에이전트 4개: 새 작업 투입 5분.
2. 동시 에이전트 8개: 새 작업 투입 5분.
3. 동시 에이전트 16개: 새 작업 투입 10분.

각 단계 뒤에는 진행 중인 작업을 최대 약 90초 마무리한다. 개별 작업은 300초 상한이다. 종료를 위한 추가 시간이 있어 전체 벽시계 시간은 조건당 20분보다 길 수 있다. 타임아웃·파싱 실패·문맥 길이 초과 등 실패도 기록하며, 성공한 작업만 선택해 지연 통계를 만들지 않는다.

32개 문제를 동일 순서로 투입하지만 완료 속도와 생성 내용이 달라 OFF/ON의 실제 호출 목록·도착 시점·문제별 반복 수는 다를 수 있다. 엄밀한 동일 요청 재생 성능 비교가 아니라, 장기 실행 중 캐시와 staging을 관찰하는 진단 실험이다.

## 측정 해석

- staging은 실제 allocator 활성 바이트를 allocation/free 이벤트와 명목 20ms 주기 관측으로 기록한다. GPU가 예약한 10GiB와 활성 KV 바이트는 다르다.
- 이번 staging 점유에는 읽기 프리페치뿐 아니라 새 KV를 DAOS로 저장하는 GPU 버퍼도 포함한다. 초기 DRAM hit가 많은 구간도 저장 때문에 staging 점유가 생길 수 있다.
- DRAM/DAOS/miss 비율의 분모는 비동기 prefix lookup 대상 청크 수다. 두 계층에 모두 저장된 키는 먼저 선택된 계층에만 계상한다. 전체 입력 토큰 비율이나 조건부 DAOS 조회 성공률과 구분한다.
- 모델 호출 동시성은 에이전트의 호출 시작·종료 콜백 기준이다. 클라이언트 대기·토큰화 등이 포함되며 GPU 계산이 동시에 실행됐다는 뜻은 아니다. Python 실행 중인 에이전트는 모델 호출 중이 아닐 수 있다.
- 지연 수치에는 계측 오버헤드가 있다. 정답 채점·전체 KV 바이트 일치 검증·CUDA 타임라인 검증은 별도다.

## 관찰 중인 DRAM 재사용 한계

현재 설치본의 CPU 비동기 조회는 첫 누락 청크에서 멈춘다. 앞부분이 DRAM에서 없어지면 뒤쪽 청크가 DRAM에 남아 있어도 해당 suffix를 CPU에서 다시 찾지 않는다. 또한 비동기 DAOS 읽기 완료 콜백의 CPU write-back은 구현되지 않아, 읽은 앞부분이 자동으로 DRAM에 복구되지 않는다.

CPU의 최근 사용 순서 갱신은 `touch_cache()`가 담당하지만, 확인한 비동기 경로에서는 동기 lookup처럼 이를 호출하는 부분이 보이지 않는다. 이 동작과 앞부분 누락 후 복구 부재가 관측된 DRAM hit 소멸에 부합한다. 실행 중 실제 퇴거 키 전체를 기록한 것은 아니므로 정확한 최초 퇴거 사건까지 증명한 것은 아니다. 측정 도중 이 정책을 수정하지 않는다.

## 실행 및 분석

```bash
cd /root/discos_minji
./venv/bin/python3 discovery_staging_bench.py \
  --output /root/discos_minji/discovery_staging_next \
  --cpu-gb 8 --conditions off,on --phases 4:300,8:300,16:600 \
  --tool-site /root/discos_minji/discovery_tool_deps_20260926

./venv/bin/python3 analyze_discovery_staging.py /root/discos_minji/discovery_staging_next
```

공용 컨테이너나 기존 캐시는 삭제하지 않는다. 이번 namespace의 데이터도 유지한다. 설치된 LMCache/vLLM과 `/root/discos` 원본은 변경하지 않는다.
