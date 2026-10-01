/*
 * libdaosgdr.c
 *
 * Thin C shim around DAOS GPUDirect object I/O (daos_obj_update_gpu /
 * daos_obj_fetch_gpu), built on top of the exact struct-assembly pattern
 * verified empirically in ~/gdr_test.c and ~/mixed_test.c:
 *
 *   - mixed_test.c confirmed: mem_attrs[i] maps 1:1 to sgls[i]/iods[i].
 *     A single nr=2 update_gpu/fetch_gpu call can mix a GPU-resident sgl
 *     (mem_attrs[i]={DAOS_MEM_TYPE_CUDA,...}) and a host-resident sgl
 *     (mem_attrs[i]={DAOS_MEM_TYPE_HOST,...}) in one RPC. KV VERIFY and
 *     META VERIFY both PASS.
 *   - mixed_test.c also confirmed: sizeof(daos_mem_attr_t)=16,
 *     offsetof(ma_mem_type)=0, offsetof(ma_device_id)=8 (4B padding in
 *     between) -- useful for a future ctypes.Structure mirror, though this
 *     shim itself never needs that layout in Python since all struct
 *     assembly happens here in C.
 *
 * shim_test.c then ran this shim's actual public API end-to-end and settled
 * the two questions this file used to mark UNVERIFIED:
 *
 *   - "not found" is rc=0 with the meta akey's fetched size == 0. Observed
 *     identically for (a) a key that was never put at all, and (b) a key
 *     that existed and was then removed via daosgdr_remove() ->
 *     daos_obj_punch_dkeys(). Existence must be checked via the reported
 *     meta length being > 0, not via rc being nonzero.
 *   - the "kv size-only" query (an empty sgl: sg_nr=0, sg_iovs=NULL, with
 *     iod_size=DAOS_REC_ANY) does NOT work: daos_obj_fetch returns rc=-2013
 *     == -DER_REC2BIG ("Record is too large") -- see daos_errno.h:126-152
 *     for where DER_REC2BIG resolves to DAOS_BASE(2000)+13=2013 (D_FOREACH_
 *     ERR_RANGE at daos_errno.h:222-224 defines DER_ERR_DAOS_BASE=2000).
 *     This matches daos_obj_fetch's own documented return-code list, which
 *     names -DER_REC2BIG for exactly this case ("Record is too large and
 *     can't be fit into output buffer") -- a zero-capacity sgl is treated
 *     as "buffer too small for the record", not "just tell me the size".
 *     There is no data-less way to learn the kv akey's size through
 *     daos_obj_fetch with this API. Consequence: the kv byte size is no
 *     longer something this shim can report on its own -- it must be
 *     encoded inside the opaque "meta" payload by the Python side, which
 *     already has to know the shape/dtype to decode "meta" in the first
 *     place. daosgdr_stat() therefore no longer takes a `size` out-param;
 *     see its doc comment below.
 *
 * Purpose: expose a small set of flat, ctypes-friendly functions to
 * LMCache's Python DaosGdrBackend. All DAOS struct assembly (d_iov_t,
 * d_sg_list_t, daos_iod_t, daos_mem_attr_t, daos_key_t) is done in this
 * file; Python only ever passes plain pointers/sizes/strings.
 *
 * Object/key mapping (same as mixed_test.c):
 *   - One fixed DAOS object (oid below) shared by all keys. DAOS_OT_MULTI_HASHED
 *     already distributes dkeys across the pool internally, so (unlike
 *     LMCache's GdsBackend, which hand-shards into l1/l2 directories because
 *     POSIX filesystems have no such distribution) no manual sharding is
 *     needed here.
 *   - dkey = the LMCache cache key string (as given by the caller), NUL
 *     excluded.
 *   - akey "kv"   = KV tensor bytes, DAOS_IOD_SINGLE, GPU memory
 *                   (mem_attr = {DAOS_MEM_TYPE_CUDA, ma_device_id=0}).
 *   - akey "meta" = opaque metadata bytes chosen entirely by the Python
 *                   side (shape/dtype/fmt/kv-size encoding, etc.) --
 *                   this shim never interprets meta's contents. Host
 *                   memory, DAOS_IOD_SINGLE, mem_attr = HOST (or plain
 *                   daos_obj_fetch/update with no mem_attrs at all).
 *
 * NOT modified: nothing under /opt/daos-gdr is touched. This file only
 * includes headers from there and links against its libdaos at build time.
 *
 * Compile (.so):
 *   gcc -shared -fPIC -O2 -o ~/libdaosgdr.so ~/libdaosgdr.c \
 *       -I/opt/daos-gdr/include \
 *       -L/opt/daos-gdr/lib64 -ldaos -ldaos_common -lgurt -lcart \
 *       -Wl,-rpath,/opt/daos-gdr/lib64
 *
 * NOTE: no CUDA runtime header/library is needed here -- gpu_ptr is a raw
 * CUDA device pointer handed in by the Python/PyTorch side; this shim never
 * calls any cudaMalloc/cudaMemcpy itself, it only passes the pointer value
 * through to DAOS (which performs the actual GPUDirect RDMA).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <daos.h>

/* ---------------------------------------------------------------------
 * Shim-local error codes.
 *
 * All *DAOS* return codes are passed through to the caller verbatim
 * (per the caller's request: "모든 rc를 그대로 반환"). These two codes
 * below are NOT DAOS return codes -- they signal a failure that happened
 * inside this shim before any DAOS call was made (e.g. bad arguments, or
 * a post-fetch sanity check).
 *
 * Confirmed non-collision (daos_errno.h:222-224, D_FOREACH_ERR_RANGE):
 *   ACTION(GURT, 1000)  ACTION(DAOS, 2000)
 * i.e. DER_ERR_GURT_BASE=1000 and DER_ERR_DAOS_BASE=2000, and the two
 * D_FOREACH_*_ERR lists (daos_errno.h:31-124 for GURT, :126-217 for DAOS)
 * each only have on the order of dozens of entries, so every actually
 * assigned -DER_* code returned by this build of libdaos has magnitude
 * in roughly [1000, 2100). (DER_UNKNOWN = DER_ERR_GURT_BASE + 500000 =
 * 501000 is a deliberate reserved sentinel for "out of range", not an
 * assigned code -- it's the only value in the whole daos_errno enum with
 * a magnitude anywhere near -100000/-100001, and it is never returned by
 * a real DAOS call.) -100000/-100001 are therefore confirmed to not
 * collide with anything this library can actually return.
 *
 * shim_test.c independently observed rc=-2013 == -DER_REC2BIG from a real
 * daos_obj_fetch() call (see the top-of-file comment) -- 2000 + 13, where
 * REC2BIG is the 13th entry in D_FOREACH_DAOS_ERR (daos_errno.h:126-152:
 * IO, FREE_MEM, ENOENT, NOTYPE, NOSCHEMA, NOLOCAL, STALE, NOTLEADER,
 * TGT_CREATE, EP_RO, EP_OLD, KEY2BIG, REC2BIG = position 13). This is
 * consistent with the base/offset arithmetic above.
 * --------------------------------------------------------------------- */
#define DAOSGDR_ERR_INVAL          (-100000) /* bad argument, no DAOS call made */
#define DAOSGDR_ERR_SIZE_MISMATCH  (-100001) /* fetched size != caller's expected size */

/* Fixed object id shared by all keys. Distinct from gdr_test.c (.lo=1) and
 * mixed_test.c (.lo=2) so this shim's data never collides with leftover
 * test data from either. */
#define DAOSGDR_OID_HI 0
#define DAOSGDR_OID_LO 1000

#define AKEY_KV   "kv"
#define AKEY_KV_LEN 2
#define AKEY_META "meta"
#define AKEY_META_LEN 4

typedef struct {
    daos_handle_t poh;   /* pool handle */
    daos_handle_t coh;   /* container handle */
    daos_handle_t oh;    /* object handle (single fixed object, all keys as dkeys) */
} daosgdr_ctx_t;

static void
daosgdr_log(const char *what, int rc)
{
    if (rc)
        fprintf(stderr, "[libdaosgdr] %s rc=%d\n", what, rc);
}

/* ---------------------------------------------------------------------
 * Measurement-only timing instrumentation, gated by DAOSGDR_TIMING=1.
 *
 * NOT for optimization -- this session's task is purely to find out where
 * daosgdr_put/stat/get spend their time (struct assembly vs the actual
 * DAOS call), because the Python DaosGdrBackend path was observed to be
 * ~2x slower than daosgdr_poc.py's raw ctypes calls despite going through
 * the same shim. No behavior changes here, only stderr diagnostics.
 *
 * Deliberately no out-param added to any function signature (per the
 * request) -- the caller side (Python) cannot read these numbers back
 * structurally, only see them on stderr. Cheap to check when disabled:
 * daosgdr_timing_enabled() does getenv() once and caches the result in a
 * static, so steady-state cost when DAOSGDR_TIMING is unset is a single
 * integer comparison per call.
 * --------------------------------------------------------------------- */
static int
daosgdr_timing_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DAOSGDR_TIMING");
        cached = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return cached;
}

static double
daosgdr_elapsed_ms(const struct timespec *start, const struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1.0e6;
}

/* ---------------------------------------------------------------------
 * daosgdr_init
 *
 * daos_init() is called exactly once here (paired with exactly one
 * daos_fini() in daosgdr_fini). Connects pool -> opens container ->
 * opens the single fixed object used for all keys.
 *
 * Returns an opaque ctx pointer on success, NULL on failure (reason
 * logged to stderr via daosgdr_log; the specific DAOS rc that failed
 * is in that stderr line, not in the return value, since NULL can't
 * carry an rc).
 * --------------------------------------------------------------------- */
void *
daosgdr_init(const char *pool, const char *cont)
{
    int rc;

    if (!pool || !cont) {
        fprintf(stderr, "[libdaosgdr] daosgdr_init: pool/cont must not be NULL\n");
        return NULL;
    }

    daosgdr_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "[libdaosgdr] daosgdr_init: calloc failed\n");
        return NULL;
    }

    rc = daos_init();
    if (rc) {
        daosgdr_log("daos_init", rc);
        free(ctx);
        return NULL;
    }

    rc = daos_pool_connect(pool, NULL, DAOS_PC_RW, &ctx->poh, NULL, NULL);
    if (rc) {
        daosgdr_log("daos_pool_connect", rc);
        daos_fini();
        free(ctx);
        return NULL;
    }

    rc = daos_cont_open(ctx->poh, cont, DAOS_COO_RW, &ctx->coh, NULL, NULL);
    if (rc) {
        daosgdr_log("daos_cont_open", rc);
        daos_pool_disconnect(ctx->poh, NULL);
        daos_fini();
        free(ctx);
        return NULL;
    }

    daos_obj_id_t oid = {.hi = DAOSGDR_OID_HI, .lo = DAOSGDR_OID_LO};
    rc = daos_obj_generate_oid(ctx->coh, &oid, DAOS_OT_MULTI_HASHED, OC_SX, 0, 0);
    if (rc) {
        daosgdr_log("daos_obj_generate_oid", rc);
        daos_cont_close(ctx->coh, NULL);
        daos_pool_disconnect(ctx->poh, NULL);
        daos_fini();
        free(ctx);
        return NULL;
    }

    rc = daos_obj_open(ctx->coh, oid, DAOS_OO_RW, &ctx->oh, NULL);
    if (rc) {
        daosgdr_log("daos_obj_open", rc);
        daos_cont_close(ctx->coh, NULL);
        daos_pool_disconnect(ctx->poh, NULL);
        daos_fini();
        free(ctx);
        return NULL;
    }

    return ctx;
}

/* ---------------------------------------------------------------------
 * daosgdr_put
 *
 * One nr=2 daos_obj_update_gpu call: akey "kv" from gpu_ptr (GPU memory,
 * DAOS_MEM_TYPE_CUDA), akey "meta" from the caller's host buffer
 * (DAOS_MEM_TYPE_HOST). Exactly the pattern verified in mixed_test.c's
 * update_gpu(mixed) call.
 *
 * daosgdr_put_device takes the CUDA device ordinal explicitly.  The legacy
 * daosgdr_put symbol remains below as an ABI-compatible device-0 wrapper.
 *
 * Returns: DAOS rc from daos_obj_update_gpu, or DAOSGDR_ERR_INVAL if
 * ctx/key/gpu_ptr/meta are NULL (checked before any DAOS call).
 * --------------------------------------------------------------------- */
int
daosgdr_put_device(void *ctx_, const char *key,
                   void *gpu_ptr, size_t size,
                   const void *meta, size_t meta_len,
                   int device_id)
{
    daosgdr_ctx_t *ctx = (daosgdr_ctx_t *)ctx_;
    int timing = daosgdr_timing_enabled();
    struct timespec t_begin, t_assembled, t_end;

    if (!ctx || !key || !gpu_ptr || (meta_len > 0 && !meta) || device_id < 0) {
        fprintf(stderr, "[libdaosgdr] daosgdr_put_device: invalid argument(s)\n");
        return DAOSGDR_ERR_INVAL;
    }

    if (timing)
        clock_gettime(CLOCK_MONOTONIC, &t_begin);

    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));

    daos_iod_t iods[2];
    memset(iods, 0, sizeof(iods));

    d_iov_set(&iods[0].iod_name, AKEY_KV, AKEY_KV_LEN);
    iods[0].iod_type = DAOS_IOD_SINGLE;
    iods[0].iod_size = size;
    iods[0].iod_nr   = 1;

    d_iov_set(&iods[1].iod_name, AKEY_META, AKEY_META_LEN);
    iods[1].iod_type = DAOS_IOD_SINGLE;
    iods[1].iod_size = meta_len;
    iods[1].iod_nr   = 1;

    d_iov_t kv_iov, meta_iov;
    d_iov_set(&kv_iov, gpu_ptr, size);
    d_iov_set(&meta_iov, (void *)meta, meta_len);

    d_sg_list_t sgls[2] = {
        {.sg_nr = 1, .sg_iovs = &kv_iov},
        {.sg_nr = 1, .sg_iovs = &meta_iov},
    };

    daos_mem_attr_t mem_attrs[2] = {
        {.ma_mem_type = DAOS_MEM_TYPE_CUDA, .ma_device_id = device_id}, /* -> iods[0]/sgls[0] = kv */
        {.ma_mem_type = DAOS_MEM_TYPE_HOST, .ma_device_id = 0}, /* -> iods[1]/sgls[1] = meta */
    };

    if (timing)
        clock_gettime(CLOCK_MONOTONIC, &t_assembled);

    int rc = daos_obj_update_gpu(ctx->oh, DAOS_TX_NONE, 0, &dkey, 2, iods,
                                  sgls, mem_attrs, NULL);

    if (timing) {
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        fprintf(stderr,
                "[daosgdr_timing] daosgdr_put: assemble=%.3fms "
                "daos_obj_update_gpu=%.3fms total=%.3fms kv_size=%zu meta_len=%zu\n",
                daosgdr_elapsed_ms(&t_begin, &t_assembled),
                daosgdr_elapsed_ms(&t_assembled, &t_end),
                daosgdr_elapsed_ms(&t_begin, &t_end),
                size, meta_len);
    }

    daosgdr_log("daos_obj_update_gpu", rc);
    return rc;
}

/* Backward-compatible entry point used by the original /root/discos code. */
int
daosgdr_put(void *ctx_, const char *key,
            void *gpu_ptr, size_t size,
            const void *meta, size_t meta_len)
{
    return daosgdr_put_device(ctx_, key, gpu_ptr, size, meta, meta_len, 0);
}

/* ---------------------------------------------------------------------
 * daosgdr_stat
 *
 * Used by LMCache's get_blocking() *before* it has allocated a GPU
 * buffer: it needs to know (a) whether the key exists at all, and
 * (b) the opaque "meta" bytes (shape/dtype/fmt, and -- by design, see
 * below -- the kv byte size too, all decoded entirely on the Python
 * side; this shim never parses meta's contents).
 *
 * One plain (non-GPU) daos_obj_fetch call, host-side, no mem_attrs
 * needed: akey "meta", iod_size=DAOS_REC_ANY, with the caller's meta
 * buffer (capacity = *meta_len) as the sgl target. This is the
 * "unknown size, let DAOS report the real size back in iod_size"
 * pattern already proven in gdr_test.c (iod.iod_size=DAOS_REC_ANY
 * before fetch_gpu) and mixed_test.c (fetch_iods[].iod_size=
 * DAOS_REC_ANY) -- not a new assumption.
 *
 * There used to be a second step here that queried the "kv" akey's size
 * directly via an empty sgl (sg_nr=0, no buffer), to spare the caller
 * from having to encode the kv size itself. shim_test.c ran that path
 * for real and got rc=-2013 == -DER_REC2BIG ("Record is too large") --
 * DAOS treats a zero-capacity sgl as "too small for this record", not
 * "just tell me the size". There is no data-less way to learn an
 * akey's size through daos_obj_fetch with this API, so that step has
 * been removed entirely. CONSEQUENCE (confirmed design, not an open
 * question anymore): the kv byte size must be encoded by the Python
 * side inside the opaque "meta" payload -- e.g. alongside shape/dtype,
 * which Python already needs to decode "meta" in the first place.
 * daosgdr_stat() therefore has no `size` out-param.
 *
 * "존재하지 않음" vs "에러": CONFIRMED empirically by shim_test.c
 * (steps 2 and 8: a key that was never put, and a key that was put and
 * then removed via daosgdr_remove(), both produced rc=0 with the meta
 * fetch's reported size == 0 -- identically, not merely similarly).
 * A "not found" key is rc=0 with *meta_len==0 after the call, NOT a
 * nonzero rc. Callers must check *meta_len, not rc, to detect
 * existence. If a key exists, rc=0 and *meta_len>0.
 *
 * Returns: rc from daos_obj_fetch verbatim (0 = call succeeded --
 * check *meta_len to tell "found" from "not found"; nonzero = an
 * actual DAOS-level error, e.g. -DER_REC2BIG if the caller's meta
 * buffer is smaller than the stored meta -- *meta_len is still updated
 * to the real size in that case so the caller can retry with a bigger
 * buffer). DAOSGDR_ERR_INVAL if ctx/key/meta_len are NULL (meta may be
 * NULL only if *meta_len is 0, i.e. caller doesn't want meta bytes at
 * all -- checked before any DAOS call).
 * --------------------------------------------------------------------- */
int
daosgdr_stat(void *ctx_, const char *key,
             void *meta, size_t *meta_len)
{
    daosgdr_ctx_t *ctx = (daosgdr_ctx_t *)ctx_;
    int timing = daosgdr_timing_enabled();
    struct timespec t_begin, t_assembled, t_end;
    size_t capacity = meta_len ? *meta_len : 0;

    if (!ctx || !key || !meta_len || (*meta_len > 0 && !meta)) {
        fprintf(stderr, "[libdaosgdr] daosgdr_stat: invalid argument(s)\n");
        return DAOSGDR_ERR_INVAL;
    }

    if (timing)
        clock_gettime(CLOCK_MONOTONIC, &t_begin);

    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));

    daos_iod_t meta_iod;
    memset(&meta_iod, 0, sizeof(meta_iod));
    d_iov_set(&meta_iod.iod_name, AKEY_META, AKEY_META_LEN);
    meta_iod.iod_type = DAOS_IOD_SINGLE;
    meta_iod.iod_size = DAOS_REC_ANY; /* proven pattern, see gdr_test.c/mixed_test.c */
    meta_iod.iod_nr   = 1;

    d_iov_t meta_iov;
    d_iov_set(&meta_iov, meta, *meta_len); /* iov_buf_len = caller's capacity */
    d_sg_list_t meta_sgl = {.sg_nr = 1, .sg_iovs = &meta_iov};

    if (timing)
        clock_gettime(CLOCK_MONOTONIC, &t_assembled);

    int rc = daos_obj_fetch(ctx->oh, DAOS_TX_NONE, 0, &dkey, 1, &meta_iod,
                             &meta_sgl, NULL, NULL);

    if (timing) {
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        fprintf(stderr,
                "[daosgdr_timing] daosgdr_stat: assemble=%.3fms "
                "daos_obj_fetch=%.3fms total=%.3fms capacity=%zu\n",
                daosgdr_elapsed_ms(&t_begin, &t_assembled),
                daosgdr_elapsed_ms(&t_assembled, &t_end),
                daosgdr_elapsed_ms(&t_begin, &t_end),
                capacity);
    }

    daosgdr_log("daos_obj_fetch(meta)", rc);
    *meta_len = (size_t)meta_iod.iod_size; /* actual size; 0 means "not found" when rc==0 */
    return rc;
}

/* ---------------------------------------------------------------------
 * daosgdr_get
 *
 * One nr=1 daos_obj_fetch_gpu call: akey "kv" straight into gpu_ptr
 * (caller must have already allocated >= size bytes there. Since
 * daosgdr_stat() has no way to report the kv size itself -- see its
 * comment: querying an akey's size with no data transfer isn't
 * possible through this API -- the caller must get `size` by decoding
 * it out of the "meta" bytes daosgdr_stat() returned, which the Python
 * side encoded at put time for exactly this purpose). mem_attrs[0] =
 * CUDA. This is the fetch half of mixed_test.c's kv path, minus the
 * "meta" akey (meta is handled separately by daosgdr_stat).
 *
 * iod_size is set to DAOS_REC_ANY (not `size`) so a real mismatch between
 * what the caller expects and what's actually stored is caught explicitly
 * below rather than silently accepted or silently truncated -- if the
 * value DAOS reports doesn't equal the caller's `size`, this returns
 * DAOSGDR_ERR_SIZE_MISMATCH instead of the (successful) DAOS rc, even
 * though the DAOS-level fetch itself may have written data into gpu_ptr.
 *
 * Returns: DAOS rc from daos_obj_fetch_gpu if nonzero; DAOSGDR_ERR_INVAL
 * for bad arguments (checked before any DAOS call); DAOSGDR_ERR_SIZE_MISMATCH
 * if the fetch succeeded but returned a different size than expected;
 * 0 on success.
 * --------------------------------------------------------------------- */
int
daosgdr_get_device(void *ctx_, const char *key,
                   void *gpu_ptr, size_t size,
                   int device_id)
{
    daosgdr_ctx_t *ctx = (daosgdr_ctx_t *)ctx_;
    int timing = daosgdr_timing_enabled();
    struct timespec t_begin, t_assembled, t_end;

    if (!ctx || !key || !gpu_ptr || device_id < 0) {
        fprintf(stderr, "[libdaosgdr] daosgdr_get_device: invalid argument(s)\n");
        return DAOSGDR_ERR_INVAL;
    }

    if (timing)
        clock_gettime(CLOCK_MONOTONIC, &t_begin);

    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));

    daos_iod_t iod;
    memset(&iod, 0, sizeof(iod));
    d_iov_set(&iod.iod_name, AKEY_KV, AKEY_KV_LEN);
    iod.iod_type = DAOS_IOD_SINGLE;
    iod.iod_size = DAOS_REC_ANY;
    iod.iod_nr   = 1;

    d_iov_t kv_iov;
    d_iov_set(&kv_iov, gpu_ptr, size);
    d_sg_list_t sgl = {.sg_nr = 1, .sg_iovs = &kv_iov};

    daos_mem_attr_t mem_attr = {
        .ma_mem_type = DAOS_MEM_TYPE_CUDA,
        .ma_device_id = device_id,
    };

    if (timing)
        clock_gettime(CLOCK_MONOTONIC, &t_assembled);

    int rc = daos_obj_fetch_gpu(ctx->oh, DAOS_TX_NONE, 0, &dkey, 1, &iod,
                                 &sgl, &mem_attr, NULL, NULL);

    if (timing) {
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        fprintf(stderr,
                "[daosgdr_timing] daosgdr_get: assemble=%.3fms "
                "daos_obj_fetch_gpu=%.3fms total=%.3fms expected_size=%zu\n",
                daosgdr_elapsed_ms(&t_begin, &t_assembled),
                daosgdr_elapsed_ms(&t_assembled, &t_end),
                daosgdr_elapsed_ms(&t_begin, &t_end),
                size);
    }

    daosgdr_log("daos_obj_fetch_gpu(kv)", rc);
    if (rc)
        return rc;

    if ((size_t)iod.iod_size != size) {
        fprintf(stderr,
                "[libdaosgdr] daosgdr_get: size mismatch, expected=%zu actual=%zu\n",
                size, (size_t)iod.iod_size);
        return DAOSGDR_ERR_SIZE_MISMATCH;
    }

    return 0;
}

/* Backward-compatible entry point used by the original /root/discos code. */
int
daosgdr_get(void *ctx_, const char *key,
            void *gpu_ptr, size_t size)
{
    return daosgdr_get_device(ctx_, key, gpu_ptr, size, 0);
}

/* ---------------------------------------------------------------------
 * daosgdr_remove
 *
 * Punches the dkey (== key) and everything under it (both "kv" and
 * "meta" akeys) via daos_obj_punch_dkeys, nr=1.
 *
 * Returns: DAOS rc verbatim, or DAOSGDR_ERR_INVAL for NULL ctx/key.
 * --------------------------------------------------------------------- */
int
daosgdr_remove(void *ctx_, const char *key)
{
    daosgdr_ctx_t *ctx = (daosgdr_ctx_t *)ctx_;

    if (!ctx || !key) {
        fprintf(stderr, "[libdaosgdr] daosgdr_remove: invalid argument(s)\n");
        return DAOSGDR_ERR_INVAL;
    }

    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));

    int rc = daos_obj_punch_dkeys(ctx->oh, DAOS_TX_NONE, 0, 1, &dkey, NULL);
    daosgdr_log("daos_obj_punch_dkeys", rc);
    return rc;
}

/* ---------------------------------------------------------------------
 * daosgdr_fini
 *
 * Mirrors daosgdr_init: closes object, closes container, disconnects
 * pool, and calls daos_fini() exactly once (paired with daosgdr_init's
 * single daos_init()). Safe to call with NULL (no-op).
 * --------------------------------------------------------------------- */
void
daosgdr_fini(void *ctx_)
{
    daosgdr_ctx_t *ctx = (daosgdr_ctx_t *)ctx_;
    if (!ctx)
        return;

    daos_obj_close(ctx->oh, NULL);
    daos_cont_close(ctx->coh, NULL);
    daos_pool_disconnect(ctx->poh, NULL);
    daos_fini();
    free(ctx);
}
