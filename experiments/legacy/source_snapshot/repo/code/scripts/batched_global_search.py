"""Batched no-gradient global DRRs for standard CMA-ES population searches."""
from __future__ import annotations
import math,time
import numpy as np
import torch
from torch.nn import functional as F
from spine_registration.torch_registration import pose_matrix
from spine_registration.benchmark_optimizers import kd_partition_starts

from active_ray_renderer import render_population_active as render_batch

@torch.no_grad()
def batch_objective(problem,poses,level=64,batch_size=16):
    problem.ensure_level(level);outputs=[]
    for start in range(0,len(poses),batch_size):
        p=torch.as_tensor(np.asarray(poses[start:start+batch_size]),dtype=torch.float32,device=problem.device)
        transforms=pose_matrix(p);losses=[]
        for view,renderers in problem.renderers[level].items():
            image=None
            for renderer in renderers:
                part=render_batch(renderer,transforms)
                image=part if image is None else image+part
            a=problem.targets[level][view].reshape(1,-1);a=a-a.mean(-1,keepdim=True)
            b=image.flatten(1);b=b-b.mean(-1,keepdim=True)
            na=torch.linalg.vector_norm(a,dim=-1,keepdim=True);nb=torch.linalg.vector_norm(b,dim=-1,keepdim=True)
            valid=na*nb>=1e-12
            value=.5*(a/torch.where(valid,na,torch.ones_like(na))-b/torch.where(valid,nb,torch.ones_like(nb))).square().sum(-1)
            losses.append(torch.where(valid[:,0],value,torch.ones_like(value)))
        outputs.extend(torch.stack(losses).mean(0).cpu().tolist())
    return np.asarray(outputs)

def coarse_cma(problem,*,seed,starts=8,population=32,max_iterations=500):
    import cma
    scale=np.array([math.pi/180]*3+[1.]*3);bounds=np.array([20.]*3+[40.]*3)
    candidates=kd_partition_starts(starts,bounds)
    best=np.zeros(6);best_loss=float(batch_objective(problem,[best])[0])
    diagnostics=[];evaluations=1;history=[]
    for index,initial in enumerate(candidates):
        search=cma.CMAEvolutionStrategy(initial.tolist(),5.,dict(bounds=[(-bounds).tolist(),bounds.tolist()],
            popsize=population,maxiter=max_iterations,tolx=1e-4,tolfun=1e-5,seed=seed+index,
            verbose=-9,verb_log=0,verb_disp=0))
        while not search.stop():
            samples=search.ask();losses=batch_objective(problem,np.asarray(samples)*scale)
            search.tell(samples,losses);evaluations+=len(samples)
            j=int(np.argmin(losses))
            if losses[j]<best_loss:best_loss=float(losses[j]);best=np.asarray(samples[j])*scale
            history.append(dict(start=index,generation=int(search.countiter),best_image_loss=best_loss,sigma=float(search.sigma)))
        diagnostics.append(dict(start=index,initial_normalized=initial.tolist(),generations=int(search.countiter),
            evaluations=int(search.countevals),stop=dict(search.stop()),sigma=float(search.sigma),
            stds=np.asarray(search.stds).tolist(),converged='maxiter' not in search.stop(),
            hit_max_iterations='maxiter' in search.stop()))
    return dict(pose=best.tolist(),loss=best_loss,diagnostics=diagnostics,history=history,
        evaluations=evaluations,iterations=sum(x['generations'] for x in diagnostics))
