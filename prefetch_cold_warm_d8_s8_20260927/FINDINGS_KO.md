# Cold/Warm 분리 실험: 확인된 결과와 해석

2026-09-27. 목표는 프리페치에 유리한 조건을 만드는 것이 아니라, 초기 캐시 생성 과정과 재사용 구간을 분리해 ON/OFF 차이를 확인하는 것이다.

## 실행한 방식

- Qwen3-14B BF16, object 경로, 청크128, DRAM 8GiB / staging 8GiB.
- 동시 요청 8/16 × DRAM 프리페치 OFF/ON, 네 조건.
- 각 조건은 빈 DRAM과 새 DAOS namespace에서 cold 256요청.
- 저장·비동기 복사 완료와 staging이 비었음을 확인한 뒤, **같은 프로세스·같은 DRAM/DAOS 캐시**로 동일 입력 256개를 warm 실행.
- 조건 사이에는 새 프로세스·새 namespace. 공용 DAOS 데이터는 삭제하지 않았다. 서버/OS 내부 캐시는 flush하지 않았다.
- rolling 투입, 저장·읽기 승격·DAOS 프리페치·용량 제한은 기존 그대로 유지. EOS/Observation 종료도 유지했다.
- 총 2,048요청 완료. 조건별 cold/warm 각 1회이며 통계적인 우열 확정 실험은 아니다.

## 결과: 평균 TTFT(ms)

| 동시 요청 | Cold OFF | Cold ON | Warm OFF | Warm ON | Warm 변화 |
|---:|---:|---:|---:|---:|---:|
| 8 | 218.35 | 212.00 | 130.06 | 119.82 | −7.87% |
| 16 | 429.15 | 409.00 | 179.94 | 162.86 | −9.49% |

이번 warm에서는 **8개와 16개 모두 ON의 평균 TTFT가 낮았다.** 따라서 “16개에서는 프리페치가 효과 없다”를 고정적인 특성으로 볼 수 없다. 하지만 한 번의 비교만으로 항상 빨라진다거나, 이전 모든 차이가 cold 때문이었다고 확정하지 않는다.

## 재사용 조건은 실제로 비슷해졌는가?

| 조건 | DRAM hit 청크 비율 | DAOS hit 청크 비율 | 미재사용 입력 토큰 합계 |
|---|---:|---:|---:|
| 8 OFF warm | 92.61% | 7.39% | 17,097 |
| 8 ON warm | 92.52% | 7.48% | 17,097 |
| 16 OFF warm | 92.44% | 7.56% | 17,097 |
| 16 ON warm | 92.42% | 7.58% | 17,097 |

8개·16개 각각 ON/OFF의 **256개 요청 모두 실제 재사용 토큰 수가 일치**했다. 남은 계산은 요청당 1~127토큰의 청크 끝부분이었다. 조회 후보 청크는 모두 찾았다. hit 비율의 분모는 전체 조회 후보 청크이며 HTTP 입력 토큰 비율과 같지 않다.

DRAM/DAOS tier별 hit까지 같은 요청은 8개에서 253/256, 16개에서 250/256이었다. 완전히 동일한 tier 배치를 강제한 실험은 아니다. 동일 tier hit 요청만의 보조 분석에서도 ON−OFF TTFT 평균은 각각 −10.28ms, −16.07ms였다. 이 선별 분석은 전체 지표를 대신하지 않는다.

## 시간이 줄어든 곳과 늘어난 곳

평균 HTTP TTFT를 실제 이벤트 경계로 나눴다. 순수 GPU 연산/네트워크 시간 분해가 아니다.

| Warm 비교 | retrieve 전 변화 | retrieve 자체 변화 | retrieve 이후 첫 텍스트까지 변화 | TTFT 변화 |
|---|---:|---:|---:|---:|
| 8개, ON−OFF | +7.71ms | −16.73ms | −1.21ms | −10.24ms |
| 16개, ON−OFF | +6.20ms | −16.62ms | −6.66ms | −17.08ms |

ON의 retrieve는 약 19ms에서 2.5~2.7ms로 줄었다. 대신 retrieve 이전 구간이 길어졌다. 프리페치가 복사를 없애는 게 아니라 더 일찍 수행하기 때문에, retrieve 시간 감소가 그대로 전체 TTFT 감소가 되는 것은 아니다.

코드에서도 CPU 프리페치 완료 후 스케줄러에 준비된 토큰 수를 알리는 구조를 확인했다. 다만 위 +6~8ms를 모두 복사나 lookup polling 때문이라고 확정하지는 않는다. 그 구간에는 서버 처리·조회·읽기·스케줄 대기가 모두 포함된다. 이후 구간 차이에도 모델 처리·스케줄·응답 전달이 포함된다.

## 공간 부족과 대기열

- 8 ON warm의 executor 평균 대기는 0.44ms, 16 ON warm은 1.67ms였다.
- 16 ON warm에서 staging peak는 7.988GiB였고, DRAM 프리페치 3건이 실제 할당 실패 후 DRAM 직접 읽기로 전환됐다. 요청 index 9, 13, 14다.
- 이 fallback은 캐시 miss/재계산이 아니다. **모든 조건의 DAOS 할당 실패에 따른 재계산 토큰은 0**이었다.
- 다른 ON 단계의 DRAM fallback은 0이었다. 임계치 정책을 새로 적용하지 않았고 기존 physical-capacity 정책을 유지했다.

## 한계와 다음 판단 기준

계산량이 같은 warm 구간에서 프리페치의 이득을 관측했다는 점은 이전 cold 시작 평균만 보던 것보다 해석하기 좋다. 그래도 각 조건 1회이고, 생성 길이·세부 tier 배치·실제 도착 시점은 완전히 같지 않다. 총 실행 시간의 차이를 프리페치만의 효과로 단정하면 안 된다.

우선 이 결과를 반복 검증할 수 있다. 만약 이후에도 warm에서 ON이 느린 실행이 나오면, 같은 계산량인지 확인한 뒤 retrieve 전 대기·복사 큐·실제 메모리 fallback·스케줄링 지연을 비교한다. lookup backoff의 존재는 코드로 확인했지만 요청별 지연 기여량은 아직 측정하지 않았다. 이번 작업에서는 그 값을 조절하지 않았다.

## 자료 및 검증

- [상세 표](RESULT_KO.md)
- [Warm TTFT 구간별 그래프](warm_ttft_intervals.png)
- [원시 지표 요약](summary.json)
- [요청별 재사용 일치 검사](paired_checks.json)
- [검증 결과](validation.json)
- [현재 제어 경로 코드 확인 메모](CONTROL_PATH_NOTES_KO.md)

관련 단위 테스트 39개 통과. 모든 조건에서 입력 hash·native library·설정 일치(프리페치와 namespace 제외), 시작 상태, phase 간 동일 worker PID, 요청별 hit/재사용량, 종료 staging/mirror drain을 확인했다. 실행 후 GPU compute process는 남아 있지 않았다.

재실행:

```bash
cd /root/discos_minji
./venv/bin/python3 -u cold_warm_prefetch.py --output /root/discos_minji/새_결과_디렉터리
./venv/bin/python3 report_cold_warm_prefetch.py /root/discos_minji/새_결과_디렉터리
```
