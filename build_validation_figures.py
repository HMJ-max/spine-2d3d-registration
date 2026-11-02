"""Build the revision from audited complete cohorts without running registration."""
from pathlib import Path
from copy import deepcopy
import json, csv, shutil, re, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





ROOT=Path(__file__).resolve().parent
HERE=Path(__file__).resolve().parent
OUT=ROOT
WORK=HERE/'revision';WORK.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
DATA=OUT/'analysis'
S=json.loads((DATA/'analysis_summary.json').read_text())
COHORTS=[('legacy','Reused cohort'),('external','External-source cohort')]
METHODS=['B','lambda_0.003','lambda_0.01','C_adaptive']
NAMES={'B':'B','lambda_0.003':'Fixed 0.003','lambda_0.01':'Fixed 0.01','C_adaptive':'Adaptive'}
COLORS=['#197D8C','#D77B65']
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600})
def primary(c,e):return next(x for x in S['cohorts'][c]['conditions']['combined']['contrasts'] if x['primary'] and x['endpoint']==e)
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.08);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(6.85,2.55),layout='constrained')
for a,endpoint,scale,label in zip(axs,['rms_mm','failure_frequency'],[1,100],['RMS-TRE difference (mm)','Failure-frequency difference (percentage points)']):
    for i,(c,name) in enumerate(COHORTS):
        e=primary(c,endpoint);v=e['difference']*scale;lo,hi=np.array(e['ci975'])*scale
        a.errorbar(v,1-i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[i],capsize=3,lw=1.6,ms=5)
        a.annotate(f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]',(v,1-i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
    a.axvline(0,color='#777777',lw=.8,ls='--');a.set_yticks([1,0],[x[1] for x in COHORTS]);a.set_ylim(-.45,1.55)
    a.set_xlabel(label);a.grid(axis='x',color='#EEEEEE');a.set_axisbelow(True)
axs[0].set_xlim(-.075,.012);axs[1].set_xlim(-1.5,8);axs[1].set_yticklabels([])
for tag,a in zip('ab',axs):a.set_title(tag,loc='left',fontweight='bold',color='black')
save(fig,'Fig5')

with (DATA/'volume_outcomes.csv').open() as f:rows=list(csv.DictReader(f))
fig,axs=plt.subplots(1,2,figsize=(6.85,3.25),layout='constrained')
for i,((c,name),a) in enumerate(zip(COHORTS,axs)):
    select=lambda m:{r['case']:float(r['rms_mm']) for r in rows if r['cohort']==c and r['condition']=='combined' and r['method']==m}
    x=select('lambda_0.01');y=select('C_adaptive');ids=sorted(x)
    lim=max(max(x.values()),max(y.values()))*1.08
    a.plot([0,lim],[0,lim],ls='--',color='#888888',lw=.8)
    a.scatter([x[k] for k in ids],[y[k] for k in ids],s=20,color=COLORS[i],alpha=.75,edgecolors='white',linewidths=.4)
    a.set(xlim=(0,lim),ylim=(0,lim),xlabel='Fixed 0.01 RMS-TRE (mm)',ylabel='Adaptive RMS-TRE (mm)')
    a.set_title(f'{chr(97+i)}  {name}',loc='left',color='black',fontweight='bold');a.set_aspect('equal')
save(fig,'FigS10')

det=json.loads((DATA/'detector_comparisons.json').read_text())
mid=json.loads((DATA/'midpoint_summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(6.85,2.8),layout='constrained')
for view,col,off in [('AP',COLORS[0],-.13),('LAT',COLORS[1],.13)]:
    v=[r['relative_L2']*100 for r in det if r['view']==view]
    axs[0].scatter(np.arange(1,11)+off,v,s=18,color=col,label=view)
axs[0].axhline(.5,color='#777777',ls='--',lw=.8,label='Review flag');axs[0].set(xlabel='Prespecified development case index',ylabel='Relative L2 (%)',ylim=(0,7));axs[0].legend(frameon=False,fontsize=7)
keys=['(1.0, 0.5)','(0.5, 0.25)','(0.25, 0.125)']
axs[1].plot(range(3),[mid['comparisons'][k]['max_relative_L2']*100 for k in keys],'-o',color=COLORS[0],ms=4)
axs[1].set_xticks(range(3),['1 vs 0.5','0.5 vs 0.25','0.25 vs 0.125']);axs[1].set(xlabel='Requested ray intervals (mm)',ylabel='Maximum relative L2 (%)',yscale='log')
for tag,a in zip('ab',axs):a.set_title(tag,loc='left',fontweight='bold',color='black')
save(fig,'FigS11')

