/* Read-only enumeration of the known KV object; no deletion API here. */
#include <daos.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    daos_handle_t poh = DAOS_HDL_INVAL, coh = DAOS_HDL_INVAL, oh = DAOS_HDL_INVAL;
    int rc = daos_init();
    if (rc) return 1;
    rc = daos_pool_connect("f973c142-2353-41da-b154-5079ba6969f2", NULL,
                           DAOS_PC_RO, &poh, NULL, NULL);
    if (rc) goto done;
    rc = daos_cont_open(poh, "a2d875e3-195b-4598-98b4-b381ef49867a",
                        DAOS_COO_RO, &coh, NULL, NULL);
    if (rc) goto pool;
    daos_obj_id_t oid = {.hi = 281543696187392ULL, .lo = 1000};
    rc = daos_obj_open(coh, oid, DAOS_OO_RO, &oh, NULL);
    if (rc) goto cont;
    daos_anchor_t anchor = {0};
    char *buffer = malloc(1024 * 1024);
    if (!buffer) { rc = -1; goto obj; }
    while (!daos_anchor_is_eof(&anchor)) {
        uint32_t nr = 1024;
        daos_key_desc_t kds[1024] = {0};
        d_iov_t iov;
        d_iov_set(&iov, buffer, 1024 * 1024);
        d_sg_list_t sgl = {.sg_nr = 1, .sg_iovs = &iov};
        rc = daos_obj_list_dkey(oh, DAOS_TX_NONE, &nr, kds, &sgl, &anchor, NULL);
        if (rc) break;
        size_t offset = 0;
        for (uint32_t i = 0; i < nr; i++) {
            uint32_t length = kds[i].kd_key_len;
            if (!length || offset + length > iov.iov_len ||
                fwrite(&length, sizeof(length), 1, stdout) != 1 ||
                fwrite(buffer + offset, 1, length, stdout) != length) {
                rc = -1; break;
            }
            offset += length;
        }
        if (rc) break;
    }
    free(buffer);
obj: daos_obj_close(oh, NULL);
cont: daos_cont_close(coh, NULL);
pool: daos_pool_disconnect(poh, NULL);
done: daos_fini();
    if (rc) fprintf(stderr, "read-only dkey enumeration rc=%d\n", rc);
    return rc ? 1 : 0;
}
