/* Isolated experiment: never opens the production OID. */
#include <daos.h>
#include <gurt/common.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    daos_handle_t pool, cont, obj;
    daos_obj_id_t oid;
} placement_ctx;

int placement_open(const char *pool, const char *cont, uint64_t nonce,
                   void **out, uint64_t *hi, uint64_t *lo)
{
    if (!out || !hi || !lo || nonce < 1000000) return -DER_INVAL;
    *out = NULL;
    int rc = daos_init();
    if (rc) return rc;
    placement_ctx *c = calloc(1, sizeof(*c));
    if (!c) { daos_fini(); return -DER_NOMEM; }
    rc = daos_pool_connect(pool, NULL, DAOS_PC_RW, &c->pool, NULL, NULL);
    if (rc) goto free_ctx;
    rc = daos_cont_open(c->pool, cont, DAOS_COO_RW, &c->cont, NULL, NULL);
    if (rc) goto close_pool;
    c->oid.lo = nonce;
    rc = daos_obj_generate_oid(c->cont, &c->oid, DAOS_OT_MULTI_HASHED, OC_SX, 0, 0);
    if (rc) goto close_cont;
    rc = daos_obj_open(c->cont, c->oid, DAOS_OO_RW, &c->obj, NULL);
    if (rc) goto close_cont;
    *hi = c->oid.hi; *lo = c->oid.lo; *out = c;
    return 0;
close_cont: daos_cont_close(c->cont, NULL);
close_pool: daos_pool_disconnect(c->pool, NULL);
free_ctx: free(c); daos_fini(); return rc;
}

/* Same algorithm as this installation's MULTI_HASHED, non-replicated S class.
 * This is version-specific diagnostics, not a general stable DAOS routing API.
 * All predictions must be cross-checked using placement_shard(). */
uint32_t placement_predict(const char *key, uint32_t shards)
{
    return d_hash_jump(d_hash_murmur64((const unsigned char *)key, strlen(key), 5731), shards);
}

int placement_shard(void *handle, const char *key, unsigned int *shard)
{
    placement_ctx *c = handle;
    daos_key_t dkey;
    daos_anchor_t anchor = {0};
    d_iov_set(&dkey, (void *)key, strlen(key));
    int rc = daos_obj_key2anchor(c->obj, DAOS_TX_NONE, &dkey, NULL, &anchor, NULL);
    if (!rc) *shard = anchor.da_shard;
    return rc;
}

int placement_io(void *handle, const char *dk, const char *ak, void *gpu,
                 size_t bytes, int write)
{
    if (!handle || !dk || !ak || !gpu || !bytes) return -DER_INVAL;
    placement_ctx *c = handle;
    daos_key_t dkey;
    daos_iod_t iod = {0};
    d_iov_t iov;
    d_sg_list_t sgl = {0};
    daos_mem_attr_t attr = {0};
    d_iov_set(&dkey, (void *)dk, strlen(dk));
    d_iov_set(&iod.iod_name, (void *)ak, strlen(ak));
    iod.iod_type = DAOS_IOD_SINGLE;
    iod.iod_size = bytes; iod.iod_nr = 1;
    d_iov_set(&iov, gpu, bytes);
    sgl.sg_nr = 1; sgl.sg_iovs = &iov;
    attr.ma_mem_type = DAOS_MEM_TYPE_CUDA; attr.ma_device_id = 0;
    int rc = write ? daos_obj_update_gpu(c->obj, DAOS_TX_NONE, 0, &dkey, 1, &iod, &sgl, &attr, NULL)
                   : daos_obj_fetch_gpu(c->obj, DAOS_TX_NONE, 0, &dkey, 1, &iod, &sgl, &attr, NULL, NULL);
    if (!rc && !write && (iod.iod_size != bytes || iov.iov_len != bytes || sgl.sg_nr_out != 1))
        return -DER_IO;
    return rc;
}

/* Untimed existence/size probe using both keys, including missing-akey checks. */
int placement_size(void *handle, const char *dk, const char *ak, uint64_t *size)
{
    placement_ctx *c = handle;
    daos_key_t dkey;
    daos_iod_t iod = {0};
    d_iov_set(&dkey, (void *)dk, strlen(dk));
    d_iov_set(&iod.iod_name, (void *)ak, strlen(ak));
    iod.iod_type = DAOS_IOD_SINGLE; iod.iod_size = DAOS_REC_ANY; iod.iod_nr = 1;
    int rc = daos_obj_fetch(c->obj, DAOS_TX_NONE, DAOS_COND_AKEY_FETCH, &dkey, 1, &iod, NULL, NULL, NULL);
    if (rc == -DER_NONEXIST) { *size = 0; return 0; }
    if (!rc) *size = iod.iod_size;
    return rc;
}

int placement_close(void *handle, int remove_created_object)
{
    placement_ctx *c = handle;
    if (!c) return -DER_INVAL;
    int rc = 0, next;
    if (remove_created_object) rc = daos_obj_punch(c->obj, DAOS_TX_NONE, 0, NULL);
    next = daos_obj_close(c->obj, NULL); if (!rc) rc = next;
    next = daos_cont_close(c->cont, NULL); if (!rc) rc = next;
    next = daos_pool_disconnect(c->pool, NULL); if (!rc) rc = next;
    free(c);
    next = daos_fini(); if (!rc) rc = next;
    return rc;
}
