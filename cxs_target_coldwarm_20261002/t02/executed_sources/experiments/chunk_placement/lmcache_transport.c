/* Same isolated object owner as the native experiment; add LMCache metadata. */
#include "placement.c"

int placement_put_meta(void *handle, const char *dk, const char *ak, const char *mk,
                       void *gpu, size_t bytes, void *meta, size_t meta_bytes, int device)
{
    if (!handle || !dk || !ak || !mk || !gpu || !bytes || !meta || !meta_bytes || device < 0)
        return -DER_INVAL;
    placement_ctx *c = handle;
    daos_key_t dkey;
    daos_iod_t iods[2] = {0};
    d_iov_t iovs[2];
    d_sg_list_t sgls[2] = {0};
    daos_mem_attr_t attrs[2] = {0};
    const char *names[2] = {ak, mk};
    void *buffers[2] = {gpu, meta};
    size_t sizes[2] = {bytes, meta_bytes};
    d_iov_set(&dkey, (void *)dk, strlen(dk));
    for (int i=0;i<2;i++) {
        d_iov_set(&iods[i].iod_name, (void *)names[i], strlen(names[i]));
        iods[i].iod_type=DAOS_IOD_SINGLE; iods[i].iod_nr=1; iods[i].iod_size=sizes[i];
        d_iov_set(&iovs[i], buffers[i], sizes[i]); sgls[i].sg_nr=1; sgls[i].sg_iovs=&iovs[i];
        attrs[i].ma_mem_type=i ? DAOS_MEM_TYPE_HOST : DAOS_MEM_TYPE_CUDA;
        attrs[i].ma_device_id=i ? 0 : device;
    }
    return daos_obj_update_gpu(c->obj, DAOS_TX_NONE, 0, &dkey, 2, iods, sgls, attrs, NULL);
}

int placement_meta(void *handle, const char *dk, const char *mk, void *buf, size_t *size)
{
    placement_ctx *c=handle;
    daos_key_t dkey; daos_iod_t iod={0}; d_iov_t iov; d_sg_list_t sgl={0};
    d_iov_set(&dkey,(void *)dk,strlen(dk));
    d_iov_set(&iod.iod_name,(void *)mk,strlen(mk));
    iod.iod_type=DAOS_IOD_SINGLE; iod.iod_nr=1; iod.iod_size=DAOS_REC_ANY;
    d_iov_set(&iov,buf,*size); sgl.sg_nr=1; sgl.sg_iovs=&iov;
    int rc=daos_obj_fetch(c->obj,DAOS_TX_NONE,0,&dkey,1,&iod,&sgl,NULL,NULL);
    if (rc==-DER_NONEXIST) { *size=0; return 0; }
    *size=iod.iod_size;
    return rc;
}

int placement_remove(void *handle, const char *dk, const char *ak, const char *mk)
{
    placement_ctx *c=handle;
    daos_key_t dkey, akeys[2];
    d_iov_set(&dkey,(void *)dk,strlen(dk));
    d_iov_set(&akeys[0],(void *)ak,strlen(ak));
    d_iov_set(&akeys[1],(void *)mk,strlen(mk));
    /* Never punch the shared placement dkey: that would remove other chunks. */
    return daos_obj_punch_akeys(c->obj,DAOS_TX_NONE,0,&dkey,2,akeys,NULL);
}
