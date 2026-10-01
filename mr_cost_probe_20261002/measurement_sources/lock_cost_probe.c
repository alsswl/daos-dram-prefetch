/* Diagnostic-only LD_PRELOAD. Time acquisition, not hold time; classify the
 * immediate caller using installed binary symbol ranges. No scheduling changes.
 * Counters are per-thread. They are read only after all measured I/O has joined.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lock_cost_ranges.h"

static int (*real_mutex)(pthread_mutex_t *);
static int (*real_spin)(pthread_spinlock_t *);
static _Atomic unsigned active;
static uintptr_t common_base, fabric_base;
struct count { uint64_t calls, ns, max_ns, over_1us, failures; };
struct node { struct node *next; unsigned tag; struct count c[3]; };
static _Atomic(struct node *) nodes;
static __thread struct node *mine;
static FILE *output;

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;
}
__attribute__((constructor)) static void init(void) {
    real_mutex=dlsym(RTLD_NEXT,"pthread_mutex_lock");
    real_spin=dlsym(RTLD_NEXT,"pthread_spin_lock");
    if (!real_mutex || !real_spin) abort();
}
static int bases(struct dl_phdr_info *info,size_t size,void *arg) {
    (void)size; (void)arg;
    if (strstr(info->dlpi_name,"libdaos_common.so")) common_base=info->dlpi_addr;
    if (strstr(info->dlpi_name,"libfabric.so")) fabric_base=info->dlpi_addr;
    return 0;
}
static int category(uintptr_t caller,int spin) {
    if (spin && fabric_base) {
        uintptr_t o=caller-fabric_base;
        if (o>=cuda_register_start && o<cuda_register_end) return 1;
        if (o>=cuda_unregister_start && o<cuda_unregister_end) return 2;
    }
    if (!spin && common_base) {
        uintptr_t o=caller-common_base;
        for (size_t i=0;i<sizeof(scheduler_ranges)/sizeof(scheduler_ranges[0]);++i)
            if (o>=scheduler_ranges[i][0] && o<scheduler_ranges[i][1]) return 0;
    }
    return -1;
}
static struct count *counter(unsigned tag,int cat) {
    if (!mine) {
        mine=calloc(1,sizeof(*mine)); if (!mine) abort();
        struct node *head=atomic_load(&nodes);
        do { mine->next=head; } while (!atomic_compare_exchange_weak(&nodes,&head,mine));
    }
    if (mine->tag!=tag) { mine->tag=tag; memset(mine->c,0,sizeof(mine->c)); }
    return &mine->c[cat];
}
static void account(struct count *c,uint64_t elapsed,int rc) {
    ++c->calls;c->ns+=elapsed;if(elapsed>c->max_ns)c->max_ns=elapsed;
    c->over_1us+=elapsed>=1000;c->failures+=rc!=0;
}
int pthread_mutex_lock(pthread_mutex_t *m) {
    unsigned tag=atomic_load_explicit(&active,memory_order_relaxed);
    int cat=tag ? category((uintptr_t)__builtin_return_address(0),0) : -1;
    if (cat<0) return real_mutex(m);
    struct count *c=counter(tag,cat);uint64_t t=now_ns();
    int rc=real_mutex(m);account(c,now_ns()-t,rc);return rc;
}
int pthread_spin_lock(pthread_spinlock_t *m) {
    unsigned tag=atomic_load_explicit(&active,memory_order_relaxed);
    int cat=tag ? category((uintptr_t)__builtin_return_address(0),1) : -1;
    if (cat<0) return real_spin(m);
    struct count *c=counter(tag,cat);uint64_t t=now_ns();
    int rc=real_spin(m);account(c,now_ns()-t,rc);return rc;
}
__attribute__((noinline)) void cost_phase_begin(unsigned tag) {
    dl_iterate_phdr(bases,NULL);
    if (!common_base || !fabric_base) abort();
    if (!output) {
        const char *path=getenv("DAOS_LOCK_COST_LOG"); if (!path) abort();
        output=fopen(path,"a"); if (!output) abort();
        uint64_t lo=UINT64_MAX;
        for(int i=0;i<1000;++i){uint64_t t=now_ns(),d=now_ns()-t;if(d<lo)lo=d;}
        fprintf(output,"{\"event\":\"init\",\"clock_pair_min_ns\":%lu}\n",lo);
    }
    atomic_store(&active,tag);
}
__attribute__((noinline)) void cost_phase_end(void) {
    unsigned tag=atomic_exchange(&active,0);struct count sums[3]={0};
    for(struct node *n=atomic_load(&nodes);n;n=n->next) if(n->tag==tag)
        for(int i=0;i<3;++i){struct count *c=&n->c[i],*s=&sums[i];
            s->calls+=c->calls;s->ns+=c->ns;s->over_1us+=c->over_1us;s->failures+=c->failures;
            if(c->max_ns>s->max_ns)s->max_ns=c->max_ns;}
    const char *names[]={"scheduler_mutex","cuda_register_spin","cuda_unregister_spin"};
    for(int i=0;i<3;++i) fprintf(output,
        "{\"tag\":%u,\"category\":\"%s\",\"calls\":%lu,\"sum_ns\":%lu,\"max_ns\":%lu,\"over_1us\":%lu,\"failures\":%lu}\n",
        tag,names[i],sums[i].calls,sums[i].ns,sums[i].max_ns,sums[i].over_1us,sums[i].failures);
    fflush(output);
}
