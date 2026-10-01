/* Standalone grouping microbenchmark. Never opens the production fixed OID.
 * Each chunk remains an 18 MiB SINGLE akey; only chunks/dkey and IODs/call vary.
 * Uses the installed GPU extension, synchronous calls on caller-owned GPU memory.
 */
#include <daos.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    daos_handle_t pool, cont, obj;
    daos_obj_id_t oid;
} group_ctx;

int group_open(const char *pool, const char *cont, uint64_t nonce,
               void **out, uint64_t *hi, uint64_t *lo)
{
    if (!out || !hi || !lo || nonce < 1000000) return -DER_INVAL;
    *out = NULL;
    int rc = daos_init();
    if (rc) return rc;
    group_ctx *c = calloc(1, sizeof(*c));
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

int group_io(void *handle, const char *key, void *gpu, size_t chunk_bytes,
             unsigned int count, int write)
{
    if (!handle || !key || !gpu || !chunk_bytes || !count || count > 96)
        return -DER_INVAL;
    group_ctx *c = handle;
    daos_key_t dkey;
    d_iov_set(&dkey, (void *)key, strlen(key));
    daos_iod_t iods[96] = {0};
    d_sg_list_t sgls[96] = {0};
    d_iov_t iovs[96] = {0};
    daos_mem_attr_t attrs[96] = {0};
    char names[96][24];
    for (unsigned i = 0; i < count; ++i) {
        snprintf(names[i], sizeof(names[i]), "chunk-%03u", i);
        d_iov_set(&iods[i].iod_name, names[i], strlen(names[i]));
        iods[i].iod_type = DAOS_IOD_SINGLE;
        iods[i].iod_size = chunk_bytes;
        iods[i].iod_nr = 1;
        d_iov_set(&iovs[i], (char *)gpu + i * chunk_bytes, chunk_bytes);
        sgls[i].sg_nr = 1;
        sgls[i].sg_iovs = &iovs[i];
        attrs[i].ma_mem_type = DAOS_MEM_TYPE_CUDA;
        attrs[i].ma_device_id = 0;
    }
    int rc = write ? daos_obj_update_gpu(c->obj, DAOS_TX_NONE, 0, &dkey,
                                        count, iods, sgls, attrs, NULL)
                   : daos_obj_fetch_gpu(c->obj, DAOS_TX_NONE, 0, &dkey,
                                       count, iods, sgls, attrs, NULL, NULL);
    if (!rc && !write)
        for (unsigned i = 0; i < count; ++i)
            if (iods[i].iod_size != chunk_bytes || iovs[i].iov_len != chunk_bytes
                || sgls[i].sg_nr_out != 1) return -DER_IO;
    return rc;
}

/* Read-only diagnostics against installed DAOS ABI. da_shard is internal;
 * never use this field for routing. Cross-check with CLI object layout.
 */
int group_shard(void *handle, const char *key, unsigned int *shard)
{
    group_ctx *c = handle;
    daos_key_t dkey;
    daos_anchor_t anchor = {0};
    d_iov_set(&dkey, (void *)key, strlen(key));
    int rc = daos_obj_key2anchor(c->obj, DAOS_TX_NONE, &dkey, NULL, &anchor, NULL);
    if (!rc) *shard = anchor.da_shard;
    return rc;
}

int group_close(void *handle, int remove_created_object)
{
    group_ctx *c = handle;
    if (!c) return -DER_INVAL;
    int rc = 0, next;
    if (remove_created_object)
        rc = daos_obj_punch(c->obj, DAOS_TX_NONE, 0, NULL);
    next = daos_obj_close(c->obj, NULL); if (!rc) rc = next;
    next = daos_cont_close(c->cont, NULL); if (!rc) rc = next;
    next = daos_pool_disconnect(c->pool, NULL); if (!rc) rc = next;
    free(c);
    next = daos_fini(); if (!rc) rc = next;
    return rc;
}
