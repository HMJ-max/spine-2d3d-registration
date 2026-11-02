"""Skip exactly zero-intersection rays in no-gradient population rendering."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def render_population_active(renderer,transforms):
    batch=len(transforms);height,width=renderer.output_size
    local=renderer.world_to_voxel[None]@torch.linalg.inv(transforms)
    source=(local[:,:3,:3]@renderer.rays.source[None,:,None]).squeeze(-1)+local[:,:3,3]
    directions=torch.einsum('pj,bkj->bpk',renderer.rays.directions,local[:,:3,:3])
    upper=source.new_tensor(renderer.volume.data.shape)-1
    parallel=directions.abs()<1e-12;divisor=torch.where(parallel,torch.ones_like(directions),directions)
    t0=-source[:,None,:]/divisor;t1=(upper-source)[:,None,:]/divisor
    outside=(parallel&((source<0)|(source>upper))[:,None,:]).any(-1)
    t0=torch.where(parallel,-torch.inf,t0);t1=torch.where(parallel,torch.inf,t1)
    near=torch.minimum(t0,t1).amax(-1).clamp_min(0);far=torch.maximum(t0,t1).amin(-1)
    valid=~outside&torch.isfinite(near)&torch.isfinite(far)&(far>near)
    pair=torch.nonzero(valid,as_tuple=False);output=torch.zeros_like(near).flatten()
    if len(pair):
        b,r=pair[:,0],pair[:,1];first=near[b,r];length=far[b,r]-first
        distances=first[:,None]+length[:,None]*renderer.fractions[None]
        positions=source[b,None,:]+distances[...,None]*directions[b,r,None,:]
        d,h,w=renderer.volume.data.shape
        grid=torch.stack((positions[...,2]/(w-1)*2-1,positions[...,1]/(h-1)*2-1,positions[...,0]/(d-1)*2-1),-1)
        samples=F.grid_sample(renderer.data,grid[None,:,:,None,:],mode='bilinear',padding_mode='zeros',align_corners=True)
        output=output.index_copy(0,b*height*width+r,samples[0,0,:,:,0].mean(-1)*length)
    return output.reshape(batch,height,width)
