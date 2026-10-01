/* Experimental array/SGL extension. Build separately from the baseline shim.
 * No payload allocation or CPU/GPU copy: only host IOV/recx descriptors.
 * Offsets and lengths are bytes; the persistent array record size is 1.
 * All calls are synchronous, so descriptor and GPU allocation lifetimes end
 * only AFTER DAOS completes. CUDA ordering is the caller's responsibility.
 */
#include "libdaosgdr.c"
#include <stdint.h>
#include <daos/common.h>

#define SCATTER_AKEY "kv_scatter_v1"
#define SCATTER_MAX_IOVS 65536U

static int
scatter_descriptors(void *const *ptrs, const size_t *lens,
                    const uint64_t *offsets, unsigned int nr,
                    d_iov_t **iovs_out, daos_recx_t **recxs_out)
{
    if (!ptrs || !lens || !offsets || !nr || nr > SCATTER_MAX_IOVS)
        return DAOSGDR_ERR_INVAL;
    uint64_t end = 0, total = 0;
    for (unsigned int i = 0; i < nr; ++i) {
        if (!ptrs[i] || !lens[i] || offsets[i] > UINT64_MAX - lens[i] ||
            (i && offsets[i] < end) ||
            (uintptr_t)ptrs[i] > UINTPTR_MAX - lens[i])
            return DAOSGDR_ERR_INVAL;
        end = offsets[i] + lens[i];
        if (total > UINT64_MAX - lens[i]) return DAOSGDR_ERR_INVAL;
        total += lens[i];
    }
    /* This DAOS build's inline path is host-only. Never let a small GPU
     * transfer silently enter it. No staging fallback is permitted. */
    if (total < DAOS_BULK_LIMIT) return DAOSGDR_ERR_INVAL;
    d_iov_t *iovs = calloc(nr, sizeof(*iovs));
    daos_recx_t *recxs = calloc(nr, sizeof(*recxs));
    if (!iovs || !recxs) {
        free(iovs);
        free(recxs);
        return -DER_NOMEM;
    }
    for (unsigned int i = 0; i < nr; ++i) {
        d_iov_set(&iovs[i], ptrs[i], lens[i]);
        recxs[i].rx_idx = offsets[i];
        recxs[i].rx_nr = lens[i];
    }
    *iovs_out = iovs;
    *recxs_out = recxs;
    return 0;
}

/* Publish metadata and the complete immutable chunk in the same update. */
int
daosgdr_putv_device(void *ctx_, const char *key, void *const *ptrs,
                   const size_t *lens, const uint64_t *offsets,
                   unsigned int nr, const void *meta, size_t meta_len,
                   int device_id)
{
    daosgdr_ctx_t *ctx = ctx_;
    d_iov_t *iovs = NULL;
    daos_recx_t *recxs = NULL;
    if (!ctx || !key || !*key || !meta || !meta_len || device_id < 0)
        return DAOSGDR_ERR_INVAL;
    int rc = scatter_descriptors(ptrs, lens, offsets, nr, &iovs, &recxs);
    if (rc)
        return rc;
    /* Writes must cover the entire chunk; sparse writes are never published. */
    uint64_t end = 0;
    for (unsigned int i = 0; i < nr; ++i) {
        if (offsets[i] != end) {
            rc = DAOSGDR_ERR_INVAL;
            goto out;
        }
        end += lens[i];
    }
    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));
    daos_iod_t iods[2] = {0};
    d_iov_set(&iods[0].iod_name, SCATTER_AKEY, strlen(SCATTER_AKEY));
    iods[0].iod_type = DAOS_IOD_ARRAY;
    iods[0].iod_size = 1;
    iods[0].iod_nr = nr;
    iods[0].iod_recxs = recxs;
    d_iov_set(&iods[1].iod_name, AKEY_META, AKEY_META_LEN);
    iods[1].iod_type = DAOS_IOD_SINGLE;
    iods[1].iod_size = meta_len;
    iods[1].iod_nr = 1;
    d_iov_t meta_iov;
    d_iov_set(&meta_iov, (void *)meta, meta_len);
    d_sg_list_t sgls[2] = {
        {.sg_nr = nr, .sg_iovs = iovs},
        {.sg_nr = 1, .sg_iovs = &meta_iov},
    };
    daos_mem_attr_t attrs[2] = {
        {.ma_mem_type = DAOS_MEM_TYPE_CUDA, .ma_device_id = device_id},
        {.ma_mem_type = DAOS_MEM_TYPE_HOST, .ma_device_id = 0},
    };
    rc = daos_obj_update_gpu(ctx->oh, DAOS_TX_NONE, 0, &dkey, 2,
                             iods, sgls, attrs, NULL);
out:
    free(recxs);
    free(iovs);
    return rc;
}

static int
recx_compare(const void *a, const void *b)
{
    uint64_t x = ((const daos_recx_t *)a)->rx_idx;
    uint64_t y = ((const daos_recx_t *)b)->rx_idx;
    return (x > y) - (x < y);
}

int
daosgdr_getv_device(void *ctx_, const char *key, void *const *ptrs,
                   const size_t *lens, const uint64_t *offsets,
                   unsigned int nr, int device_id)
{
    daosgdr_ctx_t *ctx = ctx_;
    d_iov_t *iovs = NULL;
    daos_recx_t *recxs = NULL;
    if (!ctx || !key || !*key || device_id < 0)
        return DAOSGDR_ERR_INVAL;
    int rc = scatter_descriptors(ptrs, lens, offsets, nr, &iovs, &recxs);
    if (rc)
        return rc;
    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));
    daos_iod_t iod = {0};
    d_iov_set(&iod.iod_name, SCATTER_AKEY, strlen(SCATTER_AKEY));
    iod.iod_type = DAOS_IOD_ARRAY;
    iod.iod_size = 1;
    iod.iod_nr = nr;
    iod.iod_recxs = recxs;
    d_sg_list_t sgl = {.sg_nr = nr, .sg_iovs = iovs};
    daos_mem_attr_t attr = {
        .ma_mem_type = DAOS_MEM_TYPE_CUDA, .ma_device_id = device_id,
    };
    /* DAOS allocates the detailed map. Check coverage: holes in an array must
     * not be reported as valid cache data, even if fetch returns success. */
    daos_iom_t iom = {.iom_flags = DAOS_IOMF_DETAIL};
    rc = daos_obj_fetch_gpu(ctx->oh, DAOS_TX_NONE, DAOS_COND_AKEY_FETCH,
                            &dkey, 1, &iod, &sgl, &attr, &iom, NULL);
    if (rc == 0) {
        if (iod.iod_size != 1 || iom.iom_size != 1 || !iom.iom_recxs ||
            iom.iom_nr_out > iom.iom_nr) {
            rc = DAOSGDR_ERR_SIZE_MISMATCH;
        } else {
            qsort(iom.iom_recxs, iom.iom_nr_out, sizeof(daos_recx_t), recx_compare);
            unsigned int j = 0;
            for (unsigned int i = 0; i < nr && rc == 0; ++i) {
                uint64_t cursor = offsets[i], end = cursor + lens[i];
                while (cursor < end && j < iom.iom_nr_out) {
                    daos_recx_t r = iom.iom_recxs[j];
                    if (r.rx_nr > UINT64_MAX - r.rx_idx) break;
                    uint64_t hi = r.rx_idx + r.rx_nr;
                    if (hi <= cursor) { ++j; continue; }
                    if (r.rx_idx > cursor) break;
                    cursor = hi < end ? hi : end;
                }
                if (cursor != end) rc = DAOSGDR_ERR_SIZE_MISMATCH;
            }
        }
    }
    free(iom.iom_recxs);
    free(recxs);
    free(iovs);
    return rc;
}
