Chunk placement comparison (client only)

Purpose
  Compare the existing hash-derived dkey / fixed kv akey layout with
  absolute-chunk-index-based shard selection / full KV key akey layout.
  Uses native DAOS GPU fetch/update, not a patched live LMCache backend.

Server work
  None: existing DAOS pool, container, credentials and GPU-aware stack suffice.
  Uses discospool/kvcache, creates random experiment OIDs, and punches only those
  OIDs in finally blocks. Production OID 1000 is never opened.

Addressing
  baseline: dkey = namespace + full CacheEngineKey; akey = kv.
  balanced: dkey = persisted placement_keys[absolute_chunk_index % shard_count];
            akey = namespace + full CacheEngineKey.
  A placement key is chosen using installed Murmur64(seed=5731) + Jump Hash.
  Every prediction is checked against DAOS key2anchor and object layout.
  This predictor/anchor interpretation is specific to the installed DAOS ABI,
  MULTI_HASHED object type and nonreplicated S class, not a stable generic API.
  The descriptor is versioned and saved in pair_N/addressing.json.
  Never replace its frozen shard count with the current pool target count.

Controlled comparison
  Default: 96 chunks x 18 MiB, 2 GiB CUDA slab, workers 1 and 16.
  Both layouts use a single isolated OID per pair, with disjoint key spaces,
  so they have exactly the same shard-to-target map.
  Three fresh object/key layouts, five repeats plus warmup, randomized run order.
  One fetch per chunk in original chunk order; no grouped fetch or scheduling
  based on target. One-worker condition controls for placement-independent cost.
  This benchmark matches the earlier grouping experiment, not full CXS serving.
  There is no 500 MiB window, paged KV scatter, CPU payload cache or LLM compute.
  The dkey/akey hierarchy also changes; any gain is not solely hash scheduling.
  Server caches are not flushed. Bytes are verified outside timing on every read.
  Stored-size/missing-akey probes and suffix reads are additionally checked.
  Concurrency metrics describe outstanding client calls including registration
  and progress; they do not measure time actually spent serving I/O on targets.

Run
  bash /root/discos_minji/experiments/chunk_placement/run.sh \
    --out /root/discos_minji/experiments/chunk_placement/results_UNIQUE \
    --layouts 3 --repeats 5 --workers 1 16

For LMCache integration later
  The current backend API receives a CacheEngineKey without absolute position.
  Propagate stable prefix-relative chunk positions through lookup, store, get,
  removal and restart handling, or persist a location index. Never use a
  window-local or cache-miss-list-local index. Preserve full model/dtype/rank/tag
  identity, add per-chunk metadata akeys, and keep a distinct versioned namespace
  for old/new layouts. This experiment does not make that production change.

LMCache/CXS integration added
  lmcache_daos/placement_backend.py provides an opt-in PlacementBackend.
  A process-local full-key position index is populated by ChunkedTokenDatabase;
  the immutable CacheEngineKey type itself is not changed. Stored positions
  are also persisted to placement.positions.sqlite for key-only operations.
  This is client-local experimental metadata and must be copied along with the
  manifest for a fresh client; it is not a distributed location service.
  KV+metadata use one mixed-memory update. Get fetches one chunk. Removal punches
  the two akeys only, preserving sibling chunks under the shared dkey.

  realqa_placement_comparison.py prepares and runs full mixed CXS comparisons.
  Commands: prepare, gates, run --output /root/discos_minji/realqa_placement_20261002
  The 9GiB gate passed for both modes with 65536 restored tokens, no store
  allocation failures, full GPU value verification, key-only lookup using the
  persisted position index and metadata lookup after reopening the DAOS object.
  Full 480+480 serving runs completed: realqa_placement_20261002/placement_results.json.
  Mean retrieve 330.8 -> 300.0 ms; mean TTFT 1820.7 -> 1643.3 ms; p95 TTFT
  6678.5 -> 8356.2 ms. One run each; generated histories differ. See limitations.
