static const uintptr_t scheduler_ranges[][2]={
{0x52cc0,0x52f07}, /* tse_sched_addref */
{0x54b40,0x54db7}, /* tse_sched_check_complete */
{0x55af0,0x563da}, /* tse_sched_complete */
{0x52f10,0x52f19}, /* tse_sched_decref */
{0x52330,0x52954}, /* tse_sched_fini */
{0x53470,0x53783}, /* tse_sched_init */
{0x52960,0x52cb2}, /* tse_sched_priv_decref */
{0x53a50,0x54b36}, /* tse_sched_process_complete */
{0x55870,0x55af0}, /* tse_sched_progress */
{0x52f20,0x5346e}, /* tse_sched_register_comp_cb */
{0x54dc0,0x55867}, /* tse_sched_run */
};
static const uintptr_t cuda_register_start=0x22900, cuda_register_end=0x22ac5;
static const uintptr_t cuda_unregister_start=0x22ad0, cuda_unregister_end=0x22bf0;
