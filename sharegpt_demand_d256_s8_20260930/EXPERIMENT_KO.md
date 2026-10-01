# ShareGPT: 프리페치 완전 OFF

Qwen3-14B / DRAM 256GiB / staging 8GiB / 동시요청16 / 청크128.

기존 ShareGPT 고정 입력1,684개를 cold1회, warm4회 재생한다. 빈 DRAM·staging과 새 DAOS namespace로 시작하고 warm 사이에는 캐시를 유지한다.

enable_async_loading=false, dram_prefetch=false. lookup은 존재 확인만 수행하며 DAOS payload는 retrieve 안에서 동기 읽기한다. DRAM hit은 모델 KV로 직접 전달한다. staging은 DAOS 요청 시 읽기와 새 KV 저장에 사용한다. 청크별 창 방식 전송을 새로 구현한 것은 아니다.

GPU-direct 저장 및 DAOS 읽기 후 비동기 DRAM 보관은 유지한다. 기본 동기 경로가 GPU MemoryObj를 CPU cache에 그대로 등록하지 않도록 막고, 기존 bounded D2H mirror가 복사 완료한 CPU 객체만 보관한다.

기존 DRAM-prefetch-OFF 결과는 DAOS-prefetch-ON이므로 이번 조건과 다르다. lookup 프로토콜과 읽기 시점도 바뀌므로 순수 DRAM 프리페치 효과만을 분리하는 비교는 아니다. 원본 대화 이력·출력상한·rolling 투입·GPU 설정은 기존256GiB 실험과 동일하다. OS/DAOS 서버 캐시는 강제 초기화하지 않는다. 완전 종료 후 이번 UUID KV만 삭제한다.
