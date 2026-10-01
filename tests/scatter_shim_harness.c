/* Exercise the REAL C descriptor/coverage code with an in-memory DAOS double.
 * Links DAOS for unused baseline entry points; never opens a pool or a GPU. */
#define daos_obj_fetch_gpu test_fetch_gpu
#define daos_obj_update_gpu test_update_gpu
#include "../libdaosgdr_scatter.c"
#include <assert.h>

static unsigned char payload[131072];
static int calls, hole, failure;

int test_update_gpu(daos_handle_t oh, daos_handle_t th, uint64_t flags,
                    daos_key_t *dkey, unsigned int nr, daos_iod_t *iods,
                    d_sg_list_t *sgls, daos_mem_attr_t *attrs, daos_event_t *ev)
{
    (void)oh; (void)th; (void)flags; (void)dkey; (void)ev;
    ++calls;
    assert(nr == 2 && iods[0].iod_type == DAOS_IOD_ARRAY && iods[0].iod_size == 1);
    assert(attrs[0].ma_mem_type == DAOS_MEM_TYPE_CUDA && attrs[0].ma_device_id == 2);
    assert(attrs[1].ma_mem_type == DAOS_MEM_TYPE_HOST);
    assert(iods[1].iod_type == DAOS_IOD_SINGLE && sgls[1].sg_nr == 1);
    for (unsigned int i=0; i<sgls[0].sg_nr; ++i) {
        daos_recx_t r=iods[0].iod_recxs[i];
        assert(r.rx_nr == sgls[0].sg_iovs[i].iov_len);
        assert(r.rx_idx+r.rx_nr <= sizeof(payload));
        memcpy(payload+r.rx_idx,sgls[0].sg_iovs[i].iov_buf,r.rx_nr);
    }
    return failure;
}

int test_fetch_gpu(daos_handle_t oh, daos_handle_t th, uint64_t flags,
                   daos_key_t *dkey, unsigned int nr, daos_iod_t *iods,
                   d_sg_list_t *sgls, daos_mem_attr_t *attrs,
                   daos_iom_t *iom, daos_event_t *ev)
{
    (void)oh; (void)th; (void)dkey; (void)ev;
    ++calls;
    assert(nr == 1 && flags == DAOS_COND_AKEY_FETCH);
    assert(attrs[0].ma_mem_type == DAOS_MEM_TYPE_CUDA && attrs[0].ma_device_id == 2);
    assert(iom->iom_flags == DAOS_IOMF_DETAIL);
    if (failure) return failure;
    iom->iom_size=1;
    iom->iom_nr=iom->iom_nr_out=iods[0].iod_nr;
    iom->iom_recxs=calloc(iom->iom_nr,sizeof(daos_recx_t));
    for (unsigned int i=0; i<sgls[0].sg_nr; ++i) {
        daos_recx_t r=iods[0].iod_recxs[i];
        assert(r.rx_nr == sgls[0].sg_iovs[i].iov_len);
        assert(r.rx_idx+r.rx_nr <= sizeof(payload));
        memcpy(sgls[0].sg_iovs[i].iov_buf,payload+r.rx_idx,r.rx_nr);
        iom->iom_recxs[i]=r;
    }
    if (hole) iom->iom_recxs[0].rx_nr--;
    return 0;
}

int main(void)
{
    unsigned char a[32770],b[32770],c[32770];
    for (size_t i=0; i<sizeof(a); ++i) {a[i]=i%251;b[i]=i%239;c[i]=i%227;}
    void *ptrs[]={a+1,b+1,c+1};
    size_t lengths[]={32768,32768,32768};
    uint64_t offsets[]={0,32768,65536};
    daosgdr_ctx_t ctx={0};
    assert(daosgdr_putv_device(&ctx,"key",ptrs,lengths,offsets,3,"meta",4,2)==0);
    unsigned char expected[98304];memcpy(expected,payload,sizeof(expected));
    memset(a,0xcc,sizeof(a));memset(b,0xcc,sizeof(b));memset(c,0xcc,sizeof(c));
    assert(daosgdr_getv_device(&ctx,"key",ptrs,lengths,offsets,3,2)==0);
    assert(!memcmp(a+1,expected,32768));
    assert(!memcmp(b+1,expected+32768,32768));
    assert(!memcmp(c+1,expected+65536,32768));
    assert(a[0]==0xcc && a[32769]==0xcc && b[0]==0xcc && c[32769]==0xcc);
    /* Read disjoint ranges; skipped bytes in destinations remain untouched. */
    uint64_t sparse[]={16384,65536};size_t small[]={16384,32768};
    memset(a,0xcc,sizeof(a));
    assert(daosgdr_getv_device(&ctx,"key",ptrs,small,sparse,2,2)==0);
    assert(!memcmp(a+1,expected+16384,16384) && a[16385]==0xcc);
    hole=1;
    assert(daosgdr_getv_device(&ctx,"key",ptrs,lengths,offsets,3,2)==DAOSGDR_ERR_SIZE_MISMATCH);
    hole=0;failure=-1234;
    assert(daosgdr_getv_device(&ctx,"key",ptrs,lengths,offsets,3,2)==-1234);
    failure=0;
    int before=calls;
    assert(daosgdr_putv_device(&ctx,"key",ptrs,small,sparse,2,"m",1,2)==DAOSGDR_ERR_INVAL);
    size_t tiny=16;uint64_t zero=0;
    assert(daosgdr_getv_device(&ctx,"key",ptrs,&tiny,&zero,1,2)==DAOSGDR_ERR_INVAL);
    uint64_t overlap[]={0,1,65536};
    assert(daosgdr_getv_device(&ctx,"key",ptrs,lengths,overlap,3,2)==DAOSGDR_ERR_INVAL);
    assert(calls==before);
    puts("C shim checks passed: gather/scatter, sparse extents, guards, holes, DAOS errors, invalid vectors, inline rejection");
    return 0;
}
