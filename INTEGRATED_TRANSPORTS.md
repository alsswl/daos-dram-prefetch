# Unified DFS / object-API GPU backend

한국어 구조·빌드·실측 결과·재실행 안내:
[DAOS DFS / DFS 우회 GPU I/O 통합 및 검증 자료](DFS_OBJECT_GUIDE_KO.md).

`lmcache_daos.gds_backend.DaosGdsBackend` supports two data paths selected by
one configuration value:

```yaml
extra_config:
  daosgds.transport: dfs      # libdfs / dfs_*_gpu
  # daosgds.transport: object # daos_obj_*_gpu through libdaosgdr.so
```

Both modes share the same LMCache plugin, GPU staging allocator, asynchronous
lookup/prefetch implementation, multi-prefetch patch, I/O and metadata thread
pools, and statistics.  Only the DAOS storage mapping and calls differ.

Both transports now use the original GPU-direct stack: `/opt/daos-gds-gpu`
for DAOS/Mercury and `/opt/ofi-cuda/lib64` for libfabric. This DAOS bundle
exports both DFS and object GPU APIs. The Makefile builds `libdaosgdr.so`
against the same bundle. Historical benchmark results used a different
object-mode bundle and must not be relabeled as measurements of this setup.

| mode | mapping | metadata | GPU payload call |
|---|---|---|---|
| `dfs` | one DFS file per hashed LMCache key | 4 KiB v2 header | `dfs_read_gpu` / `dfs_write_gpu` |
| `object` | one fixed multi-hashed object; namespace + LMCache key = dkey | `meta` akey | `daos_obj_fetch_gpu` / `daos_obj_update_gpu` on `kv` akey |

The checked-in comparison config uses the healthy POSIX container
`discospool/kvcache` for both modes. DFS files live below `/minji-v2`, while
object dkeys begin with `minji-v2:`. These separate namespaces avoid mixing
this checkout's entries with existing cache data. The old `gdrcont` is an
object-only container (`layout_type: unknown`) and therefore cannot be used
for the DFS side of the same comparison.

## Build the object shim

```bash
cd /root/discos_minji
make
make check
```

The target shim adds device-aware entry points while preserving the original
device-0 ABI.  `/root/discos` is not read or modified by the build.

## Select and run

When using `run_vllm.sh`, explicitly set `DAOSGDS_TRANSPORT`: the wrapper
defaults this environment variable to `dfs` and selects the matching client
bundle. This overrides `daosgds.transport` in the YAML. Direct backend users
without the environment variable can select the transport in the YAML.

```bash
DAOSGDS_TRANSPORT=dfs /root/discos_minji/run_vllm.sh <command...>
DAOSGDS_TRANSPORT=object /root/discos_minji/run_vllm.sh <command...>
```

The log line beginning `DaosGdsBackend: transport=` confirms the selected
path. Restart the process between A/B runs because the transport, DAOS
handles, and LMCache monkey patch are initialized once per process.

For a meaningful comparison, keep pool, container, GPU buffer size,
`io_workers`, `meta_workers`, prompt set, concurrency, and
`DAOS_GDS_MULTI_PREFETCH` identical.
The two modes use different DAOS namespaces and on-media formats, so data
written in one mode is not expected to be readable by the other.
