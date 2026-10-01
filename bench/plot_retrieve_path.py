import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(sys.argv[1]);out=base/'retrieve_path_profile_20261002'
read=lambda name:json.loads((base/name/'summary.json').read_text())
a=read('retrieve_path_profile_20261002');b=read('retrieve_path_profile64_20261002');c=read('retrieve_path_omp1_20261002')
ttft=json.loads((out/'ttft_attribution.json').read_text())['means']
fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for phase,color in [('engine','#167d8d'),('native','#c27426')]:
    ax[0,0].plot([r['workers'] for r in a],[r[phase]['ms']['median'] for r in a],'-o',color=color,label=f'{phase}: sweep A')
    ax[0,0].plot([r['workers'] for r in b],[r[phase]['ms']['median'] for r in b],'--s',color=color,label=f'{phase}: sweep B')
ax[0,0].set_xscale('log',base=2);ax[0,0].set_xticks([1,2,4,8,16,32,64],[1,2,4,8,16,32,64])
ax[0,0].set_ylabel('9 GiB read latency (ms)');ax[0,0].set_xlabel('Concurrent payload workers')
ax[0,0].set_title('Scaling flattens around 16-32 workers');ax[0,0].legend(fontsize=8)
for data,label,color in [(b,'Default CPU threading','#858b92'),(c,'OMP / BLAS threads = 1','#167d8d')]:
    ax[0,1].plot([r['workers'] for r in data],[r['engine']['ms']['median'] for r in data],'-o',label=label,color=color)
ax[0,1].set_xticks([16,32,64]);ax[0,1].set_ylim(250,380)
ax[0,1].set_ylabel('Full retrieve latency (ms)');ax[0,1].set_xlabel('Concurrent payload workers')
ax[0,1].set_title('CPU thread control helps, plateau remains');ax[0,1].legend(fontsize=8)
row=next(r for r in a if r['workers']==16)['engine']['wall_partition_mean_ms']
parts=[('Payload active',row['payload_without_scatter']+row['payload_with_scatter'],'#167d8d'),
       ('Scatter only',row['scatter_without_payload'],'#c27426'),
       ('Metadata only',row['metadata_without_payload_or_scatter'],'#b68caf'),('Other',row['other'],'#9a9da1')]
left=0
for label,v,color in parts:
    ax[1,0].barh(['Retrieve'],[v],left=left,label=f'{label}: {v:.1f} ms',color=color);left+=v
ax[1,0].set_xlabel('Wall time, mutually exclusive categories (ms)')
ax[1,0].set_title('16 workers: most scatter overlaps payload I/O')
ax[1,0].legend(fontsize=8,loc='lower right')
left=0
for key,label,color in [('before_retrieve_ms','Before retrieve','#9298a0'),('retrieve_ms','Retrieve','#167d8d'),('after_retrieve_ms','After retrieve','#c27426')]:
    v=ttft[key];ax[1,1].barh(['TTFT'],[v],left=left,color=color,label=f'{label}: {v:.0f} ms')
    ax[1,1].text(left+v/2,0,f'{v:.0f}',ha='center',va='center',color='white');left+=v
ax[1,1].set_xlabel('Mean time (ms), 418 cache-restoring CXS requests')
ax[1,1].set_title('Existing CXS trace: retrieve is 43% of TTFT')
ax[1,1].legend(fontsize=8,loc='lower right')
for axes in ax.flat:axes.grid(axis='x' if axes in ax[1] else 'y',alpha=.15)
fig.suptitle('DAOS + GDR retrieve-path profiling\n2 GiB staging, 486 MiB windows, no client DRAM payload cache; 3 repeats per condition',fontsize=13)
fig.savefig(out/'overview.png',dpi=180);fig.savefig(out/'overview.svg')
