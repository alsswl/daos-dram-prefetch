# DAOS 실험 namespace 정리 후보 — 삭제하지 않음

설정과 실행 로그의 존재를 기준으로 만든 검토 목록이다. DAOS 키 열거·삭제는 하지 않았다. 현재 서버에 남아 있는 키 수, 실제 저장량, 회수 가능한 물리 용량, 다른 프로세스의 사용 여부는 미확인이다.

## 우선 검토: 최근 재생 실험

`discospool/kvcache`의 최근 실험 전용 namespace **26개**. 결과 로그·그래프를 보존한 채 KV 데이터만 정리하는 후보다. 삭제하면 해당 namespace의 warm/restart 재사용은 불가능해지고 다시 fill해야 한다. 새 namespace로 시작하는 다음 rolling 실험에는 과거 캐시가 필요하지 않다.

| 실험 | namespace 수 |
|---|---:|
| discovery_capacity_matrix_20260927 | 6 |
| discovery_capacity_matrix_off_20260927 | 6 |
| discovery_fixed_replay_20260926 | 6 |
| discovery_replay_s8_d4_c16_20260927 | 2 |
| prefetch_d8_s8_c8_repeat3_20260927 | 6 |

### 정확한 namespace와 설정 근거

| 번호 | 실행 설정 | namespace 전체 문자열 |
|---:|---|---|
| 1 | [discovery_capacity_matrix_20260927/d4_s4/config.yaml](/root/discos_minji/discovery_capacity_matrix_20260927/d4_s4/config.yaml) | `minji-fixed-replay-04360371d13f4329a671580b00fd3e63:` |
| 2 | [discovery_capacity_matrix_20260927/d2_s4/config.yaml](/root/discos_minji/discovery_capacity_matrix_20260927/d2_s4/config.yaml) | `minji-fixed-replay-2ddb5e26c06b430bb0c1f4ad6f9b6067:` |
| 3 | [discovery_capacity_matrix_20260927/d4_s8/config.yaml](/root/discos_minji/discovery_capacity_matrix_20260927/d4_s8/config.yaml) | `minji-fixed-replay-653a35f8e6a54e3283e18ce54fa80525:` |
| 4 | [discovery_capacity_matrix_20260927/d8_s4/config.yaml](/root/discos_minji/discovery_capacity_matrix_20260927/d8_s4/config.yaml) | `minji-fixed-replay-695f06203a824fafb92e42935c7b96ef:` |
| 5 | [discovery_capacity_matrix_20260927/d8_s8/config.yaml](/root/discos_minji/discovery_capacity_matrix_20260927/d8_s8/config.yaml) | `minji-fixed-replay-6fb0c72290bd458085edff90376228ba:` |
| 6 | [discovery_capacity_matrix_20260927/d2_s8/config.yaml](/root/discos_minji/discovery_capacity_matrix_20260927/d2_s8/config.yaml) | `minji-fixed-replay-87955396d9a6488c8b04c39942d83ba5:` |
| 7 | [discovery_capacity_matrix_off_20260927/d8_s4/config.yaml](/root/discos_minji/discovery_capacity_matrix_off_20260927/d8_s4/config.yaml) | `minji-fixed-replay-0f3381dc142d4fe489e79681ada76566:` |
| 8 | [discovery_capacity_matrix_off_20260927/d2_s4/config.yaml](/root/discos_minji/discovery_capacity_matrix_off_20260927/d2_s4/config.yaml) | `minji-fixed-replay-208a7010cc6542dcb9325dde338be364:` |
| 9 | [discovery_capacity_matrix_off_20260927/d8_s8/config.yaml](/root/discos_minji/discovery_capacity_matrix_off_20260927/d8_s8/config.yaml) | `minji-fixed-replay-a1842d75930044b3979dd89a1f5e1ecf:` |
| 10 | [discovery_capacity_matrix_off_20260927/d2_s8/config.yaml](/root/discos_minji/discovery_capacity_matrix_off_20260927/d2_s8/config.yaml) | `minji-fixed-replay-a4a2176dc9bd4299993443001da16501:` |
| 11 | [discovery_capacity_matrix_off_20260927/d4_s4/config.yaml](/root/discos_minji/discovery_capacity_matrix_off_20260927/d4_s4/config.yaml) | `minji-fixed-replay-ac9d59be31ce4d749ec35ba472681f52:` |
| 12 | [discovery_capacity_matrix_off_20260927/d4_s8/config.yaml](/root/discos_minji/discovery_capacity_matrix_off_20260927/d4_s8/config.yaml) | `minji-fixed-replay-c06af06fda054261abe4536b4c34f12a:` |
| 13 | [discovery_fixed_replay_20260926/c08_off/config.yaml](/root/discos_minji/discovery_fixed_replay_20260926/c08_off/config.yaml) | `minji-fixed-replay-0454929d7c814b33b35d04bb236e8309:` |
| 14 | [discovery_fixed_replay_20260926/c04_on/config.yaml](/root/discos_minji/discovery_fixed_replay_20260926/c04_on/config.yaml) | `minji-fixed-replay-726d582f41e84d369a3d97a3a93d9044:` |
| 15 | [discovery_fixed_replay_20260926/c16_on/config.yaml](/root/discos_minji/discovery_fixed_replay_20260926/c16_on/config.yaml) | `minji-fixed-replay-7b0dfa02a99b41b9a0c609f2146883f8:` |
| 16 | [discovery_fixed_replay_20260926/c08_on/config.yaml](/root/discos_minji/discovery_fixed_replay_20260926/c08_on/config.yaml) | `minji-fixed-replay-7f2f00ec569d410ca7445c943e95f2dd:` |
| 17 | [discovery_fixed_replay_20260926/c04_off/config.yaml](/root/discos_minji/discovery_fixed_replay_20260926/c04_off/config.yaml) | `minji-fixed-replay-8f9ac7396e8945798c6d5540fc581eda:` |
| 18 | [discovery_fixed_replay_20260926/c16_off/config.yaml](/root/discos_minji/discovery_fixed_replay_20260926/c16_off/config.yaml) | `minji-fixed-replay-e97d72f8dddd480ebf1b6f82d2a127ad:` |
| 19 | [discovery_replay_s8_d4_c16_20260927/c16_on/config.yaml](/root/discos_minji/discovery_replay_s8_d4_c16_20260927/c16_on/config.yaml) | `minji-fixed-replay-02179e44f2c34c129f5565af710015b0:` |
| 20 | [discovery_replay_s8_d4_c16_20260927/c16_off/config.yaml](/root/discos_minji/discovery_replay_s8_d4_c16_20260927/c16_off/config.yaml) | `minji-fixed-replay-ab05ae9c2b2f417dae8e363a1209d230:` |
| 21 | [prefetch_d8_s8_c8_repeat3_20260927/r2_on/config.yaml](/root/discos_minji/prefetch_d8_s8_c8_repeat3_20260927/r2_on/config.yaml) | `minji-fixed-replay-2517e91cff044b8baf027d53c47db751:` |
| 22 | [prefetch_d8_s8_c8_repeat3_20260927/r3_on/config.yaml](/root/discos_minji/prefetch_d8_s8_c8_repeat3_20260927/r3_on/config.yaml) | `minji-fixed-replay-45be1ae38c2f4c05b814093b6aa5e0bd:` |
| 23 | [prefetch_d8_s8_c8_repeat3_20260927/r2_off/config.yaml](/root/discos_minji/prefetch_d8_s8_c8_repeat3_20260927/r2_off/config.yaml) | `minji-fixed-replay-872d173c4cb5494f861b6bc4bb115ecf:` |
| 24 | [prefetch_d8_s8_c8_repeat3_20260927/r3_off/config.yaml](/root/discos_minji/prefetch_d8_s8_c8_repeat3_20260927/r3_off/config.yaml) | `minji-fixed-replay-b1cef83336304ec58cfc0eb7c881276d:` |
| 25 | [prefetch_d8_s8_c8_repeat3_20260927/r1_off/config.yaml](/root/discos_minji/prefetch_d8_s8_c8_repeat3_20260927/r1_off/config.yaml) | `minji-fixed-replay-d7321993a1834d58a10d43a0a16c3867:` |
| 26 | [prefetch_d8_s8_c8_repeat3_20260927/r1_on/config.yaml](/root/discos_minji/prefetch_d8_s8_c8_repeat3_20260927/r1_on/config.yaml) | `minji-fixed-replay-ed5cfdf83a634d8094f157d6d19235cc:` |

## 기본 설정 — 우선 정리에서 제외

- `minji-async-dram:`: lmcache_config_daosgds_async_dram.yaml
- `minji-gpu-store:`: lmcache_config_daosgds_gpu_store.yaml
- `minji-v2:`: lmcache_config_daosgds_unified.yaml

## 그 밖의 과거 실험 — 별도 검토

실행 실패·부분 저장·중복 설정이 있을 수 있다. 로그가 있다는 사실만으로 현재 데이터 잔존이나 삭제 안전성을 보장하지 않는다.

| 구분 | 실험 | namespace |
|---|---|---|
| EXCLUDE_DRYRUN | staging_dryrun_20260921_v1 | `minji-staging-5ab2b29382c84d4d8af55734687e0dd1:` |
| OLDER_REVIEW | async_dram_inference_20260926 | `minji-async-dram-inference-3258de7a97b64476a12fbc7986f90488:` |
| OLDER_REVIEW | async_dram_mixed_20260926 | `minji-async-dram-inference-25a1db14321b4d81b3fb48ece14a9b2f:` |
| OLDER_REVIEW | chunk128_20260917_v1 | `minji-compare-20260917-143803-d0026824-r1-object:` |
| OLDER_REVIEW | chunk128_20260917_v1 | `minji-compare-20260917-143803-d0026824-r2-object:` |
| OLDER_REVIEW | chunk128_20260917_v1 | `minji-compare-20260917-143803-d0026824-r3-object:` |
| OLDER_REVIEW | chunk128_20260917_v2 | `minji-compare-20260917-144516-3bc7488e-r1-object:` |
| OLDER_REVIEW | chunk128_20260917_v2 | `minji-compare-20260917-144516-3bc7488e-r2-object:` |
| OLDER_REVIEW | chunk128_20260917_v2 | `minji-compare-20260917-144516-3bc7488e-r3-object:` |
| OLDER_REVIEW | comparison_smoke_20260916_v1 | `minji-compare-20260916-195329-2081dff8-r1-object:` |
| OLDER_REVIEW | discovery_async_dram_20260926 | `minji-discovery-live-624dbd232a0b498ebe17e89d3cb9424b:` |
| OLDER_REVIEW | discovery_async_dram_prefetch_20260926 | `minji-discovery-live-ddd02cc2e90e4f78ac750ff89da8c004:` |
| OLDER_REVIEW | discovery_staging_long_20260926_v2 | `minji-discovery-live-3f75502420b046c1a5b994d477b1230b:` |
| OLDER_REVIEW | discovery_staging_long_20260926_v2 | `minji-discovery-live-9ab71f3a473946d98725a9d6fa4a92d8:` |
| OLDER_REVIEW | discovery_staging_pilot_20260926_v1 | `minji-discovery-live-687cb9edd9124ecc9b107bc0de8fc71c:` |
| OLDER_REVIEW | discovery_staging_pilot_20260926_v2 | `minji-discovery-live-adc45669772441ec9f376be1cc12cc58:` |
| OLDER_REVIEW | dram_baseline_20260922_v1 | `minji-dram-6a590045a8d94bc88fad74385f74c863:` |
| OLDER_REVIEW | dram_gpu_prefetch_off_20260922_r1 | `minji-dram-8fb70a4a6a784e3f8cde5f7650b27cad:` |
| OLDER_REVIEW | dram_gpu_prefetch_off_20260922_r2 | `minji-dram-e04fa5c2ca284c62bb8c3a99f2a95546:` |
| OLDER_REVIEW | dram_gpu_prefetch_on_20260922_r1 | `minji-dram-d5aa19d2b6c5478091b8b0c15bb3e9b6:` |
| OLDER_REVIEW | dram_gpu_prefetch_on_20260922_r2 | `minji-dram-be9e06def16742dfb4481006f55f33b5:` |
| OLDER_REVIEW | dram_off_20260922_r1 | `minji-dram-61d6aa16cf554f6697c8c96fb38230a1:` |
| OLDER_REVIEW | dram_off_20260922_r2 | `minji-dram-d1bf7e09b6bb4ec09bc0805e0c4439e6:` |
| OLDER_REVIEW | dram_on_20260922_r1 | `minji-dram-6f546e3ad00640c9b374fe8e94c81b87:` |
| OLDER_REVIEW | dram_on_20260922_r2 | `minji-dram-977dfcd7136a46f580f44ceb2ac12bb8:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-158a9e0d69134fae8b0b39bf5c1c800a:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-1784422a8570465fba30fed2298d7656:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-2b0dbc72574d48f8bc17fee213c56a86:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-3a02dd9ec00f41359b0140dc4e32bf96:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-3ad2964149a94d67ae22a4dbfe431f8e:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-417ac7c831384f04bd59b4db28649d99:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-5ddac937815842399efcd92ca6cb3cf4:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-742ac8c233bb40988197bbd3e31c1bfb:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-7df492d3287542b4a9911988375a9099:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-9f8841ddbeae4d01921e35ea2efe4e2d:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-bd0698f3386140c58436068c3b491e91:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-cd7b7dfa4c2b442e8aedd3cf4cfeb62c:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-d170aad6e0584be0baa82b56fcfa47e8:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-ea14297485e74ab0bc265cbaeaedd595:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-ed72a1b818ed4ba9a502ea4d22484413:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-f11a52889e0649ef90f68155b12be9f3:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-f61dec0115a24ecd94743de81150b0e3:` |
| OLDER_REVIEW | dram_threeway_20260922_v1 | `minji-cold-feb6da32f4b54791aca3bc611e43890c:` |
| OLDER_REVIEW | full_agentic_20260917_v1 | `minji-compare-agent-20260917-163106-39f72700-r1-object:` |
| OLDER_REVIEW | full_agentic_20260917_v2 | `minji-compare-agent-20260917-164106-57f521e8-r1-object:` |
| OLDER_REVIEW | full_agentic_restart10_20260917_v1 | `minji-compare-agent-20260917-225519-f458a935-r1-object:` |
| OLDER_REVIEW | gluesys_reference_20260916_v1 | `minji-compare-20260916-200146-dcf439f4-r1-object:` |
| OLDER_REVIEW | gluesys_reference_20260916_v1 | `minji-compare-20260916-200146-dcf439f4-r2-object:` |
| OLDER_REVIEW | gluesys_reference_20260916_v1 | `minji-compare-20260916-200146-dcf439f4-r3-object:` |
| OLDER_REVIEW | gpu_store_ab_20260926 | `minji-store-ab-2c17728c94d5487d943fb92c04182bde:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v2 | `minji-store-ab-0437e345584247afb5199dcf48f61d28:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v2 | `minji-store-ab-b115699a653f470586b61e6c009b27ca:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v3 | `minji-store-ab-02d623756259445ca42bc3d50962b852:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v3 | `minji-store-ab-4a82a5a1fbd44afa9c48ea32ea2ead55:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v3 | `minji-store-ab-57b99bb9a07b4472b5c5e1f2e2117269:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v3 | `minji-store-ab-75f2b71f385d452d9afec4377909d402:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v3 | `minji-store-ab-a9993486b7704b72a4aa6d84a776ef60:` |
| OLDER_REVIEW | gpu_store_ab_20260926_v3 | `minji-store-ab-d7c5ce35b9d24a30bdf3531fc5f2fe2b:` |
| OLDER_REVIEW | prefetch_cold_warm_d8_s8_20260927 | `minji-cold-warm-1e2ef720e49a411e9bb602f38c71f847:` |
| OLDER_REVIEW | prefetch_cold_warm_d8_s8_20260927 | `minji-cold-warm-74264beef5f5470297a1dfc85be6f1eb:` |
| OLDER_REVIEW | prefetch_cold_warm_d8_s8_20260927 | `minji-cold-warm-76586c1c8812440c8537165118fa8e1d:` |
| OLDER_REVIEW | prefetch_cold_warm_d8_s8_20260927 | `minji-cold-warm-8c6222646ff04895b251db7bbd7a0a3c:` |
| OLDER_REVIEW | prefetch_d8_s8_c16_rolling_repeat3_20260927 | `minji-rolling-replay-33a22cb969924d3c88f436263207c0be:` |
| OLDER_REVIEW | prefetch_d8_s8_c16_rolling_repeat3_20260927 | `minji-rolling-replay-33a80e20c26c431383cb4c0f7c5b1fc6:` |
| OLDER_REVIEW | prefetch_d8_s8_c16_rolling_repeat3_20260927 | `minji-rolling-replay-73bc404d06094a919e4fd93f3cc8d2d0:` |
| OLDER_REVIEW | prefetch_d8_s8_c16_rolling_repeat3_20260927 | `minji-rolling-replay-7dcd0708669b4d6185e4f5df255b4841:` |
| OLDER_REVIEW | prefetch_d8_s8_c16_rolling_repeat3_20260927 | `minji-rolling-replay-8e504f0410244092ac2400afd613b8ac:` |
| OLDER_REVIEW | prefetch_d8_s8_c16_rolling_repeat3_20260927 | `minji-rolling-replay-f7cb90d3c1f04bb99fd9c3fcb995e218:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-0701f2e3434a40db970b8e0af5204b3e:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-0851e1a3af5743f989422a0e0fd62b14:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-0a4b6612da76456fa62bbcc2cdffcbf4:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-17d1088fb8b54f83a166373d3c26e867:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-56a5d22ab1c74acda286971849098915:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-56dcfdce023c4faaae4cf1792d1a99ce:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-5d473c70a50d4cf4a2e831b226031a80:` |
| OLDER_REVIEW | prefetch_timing_d8_s8_20260927 | `minji-rolling-replay-80cdd2d2a26549fc88fcff91ba1fd9bb:` |
| OLDER_REVIEW | queued_prefetch_experiment_20260927 | `minji-cold-warm-1ac5105ee3cc429d9d35fddaf5a44b8d:` |
| OLDER_REVIEW | queued_prefetch_experiment_20260927 | `minji-cold-warm-23486761fbc4471f9d9f23250c44dba1:` |
| OLDER_REVIEW | queued_prefetch_experiment_20260927 | `minji-cold-warm-55214d1b6240455db1fc95dd45aa7baa:` |
| OLDER_REVIEW | queued_prefetch_experiment_20260927 | `minji-cold-warm-75eee94bb9d04b9ba239e629e1731c13:` |
| OLDER_REVIEW | queued_prefetch_experiment_20260927 | `minji-cold-warm-c32d714d4d3b4c23b1215a583632527e:` |
| OLDER_REVIEW | queued_prefetch_experiment_20260927 | `minji-cold-warm-d4de8d9370134b578f0a3e7aa68c0854:` |
| OLDER_REVIEW | read_promotion_inference_20260926 | `minji-read-promotion-inference-1ab1486aab5d461bae5744ba16050c13:` |
| OLDER_REVIEW | staging_mixed_20260923_v1 | `minji-mixed-pressure-a14d2c0deb9d4ca4b08c5c5c1da5fb97prefetch_off:` |
| OLDER_REVIEW | staging_mixed_20260923_v2 | `minji-mixed-pressure-fd344fdd65b946c6892600f96867b306prefetch_off:` |
| OLDER_REVIEW | staging_mixed_20260923_v2 | `minji-mixed-pressure-fd344fdd65b946c6892600f96867b306prefetch_on:` |
| OLDER_REVIEW | staging_pressure_20260921_v1 | `minji-staging-9cd9107365744f0a8376f65297e0b05a:` |
| REVIEW_NO_LOG | comparison_20260916-195322-ce592af3 | `minji-compare-20260916-195322-ce592af3-r1-object:` |
| REVIEW_NO_LOG | discovery_staging_long_20260926_v1 | `minji-discovery-live-dc079eeb4fa249798f7a702db0f714be:` |

## 안전 범위

- 이 목록은 삭제 승인이나 자동 삭제 스크립트가 아니다. broad prefix `minji-` 전체 삭제 금지.
- 삭제 승인 후에도 pool/container/OID와 위의 정확한 namespace를 대조해 dkey 목록을 먼저 확인해야 한다.
- 풀·컨테이너 전체 및 서버 NVMe 파일을 직접 삭제하지 않는다.
- DFS 구성에 적힌 object_namespace는 실제 object 저장을 의미하지 않는다. DFS 설정은 별도 제외 목록으로 inventory.json에 보존했다.
- dry-run과 실행 로그가 없는 설정은 우선 후보에서 제외했다. 실행 소스 스냅샷 안의 기본 설정도 중복 집계하지 않았다.
- 로컬 로그·CSV·그래프·소스는 지우지 않는다. KV 제거는 백업이 없다면 되돌릴 수 없고 다시 추론/저장해야 한다.
- 예상 확보 용량은 미산정이다. 논리 키 삭제 후 물리 공간 반영까지도 별도로 확인해야 한다.

[전체 JSON](inventory.json) · [CSV](inventory.csv) · [우선 후보 JSON](priority_review.json)
