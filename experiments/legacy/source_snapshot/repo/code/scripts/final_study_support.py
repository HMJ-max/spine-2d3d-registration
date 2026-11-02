"""Versioned study utilities; the corrected historical benchmark stays frozen."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from posture_geometry import _penetration, audit_geometry, _overlap_on_common_grid


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def articulated(prepared, angles_deg, rotation_rad, translation_mm):
    """Midpoint hinges plus the smallest feasible bounded axial separation.

    shortcut: bone-only geometric proxy, not measured disc kinematics; upgrade
    to patient-matched paired imaging before interpreting it as clinical motion.
    """
    centres=prepared['centres'];angles=np.asarray(angles_deg,float)
    if angles.shape!=(4,) or not np.isfinite(angles).all():
        raise ValueError('Four finite joint angles required')
    pivots=(centres[:-1]+centres[1:])/2
    transforms=np.tile(np.eye(4),(5,1,1));separations=[];feasible=True
    for i in range(3,-1,-1):
        first,second=prepared['components'][i:i+2]
        hinge=np.eye(4);hinge[:3,:3]=Rotation.from_rotvec([np.deg2rad(angles[i]),0,0]).as_matrix()
        baseline=pivots[i]-hinge[:3,:3]@pivots[i]
        axis=centres[i]-centres[i+1];axis/=np.linalg.norm(axis)
        accepted=False
        for separation in np.arange(21)*.1:
            hinge[:3,3]=baseline+separation*axis
            audits=(_penetration(first,second,hinge,prepared['voxel_tolerance_mm']),
                    _penetration(second,first,np.linalg.inv(hinge),prepared['voxel_tolerance_mm']))
            if all(a['penetration_fraction_beyond_tolerance']==0 for a in audits):
                overlap=_overlap_on_common_grid(prepared,first,second,hinge,np.eye(4))
                if overlap['fraction_smaller_volume']<=.01:
                    accepted=True;break
        feasible &= accepted
        separations.insert(0,float(separation))
        transforms[i]=transforms[i+1]@hinge
    global_matrix=np.eye(4);global_matrix[:3,:3]=Rotation.from_rotvec(rotation_rad).as_matrix()
    global_matrix[:3,3]=centres[-1]+translation_mm-global_matrix[:3,:3]@centres[-1]
    local=transforms.copy();transforms=global_matrix[None]@transforms
    moved=np.einsum('nij,nj->ni',transforms[:,:3,:3],centres)+transforms[:,:3,3]
    params=np.column_stack((Rotation.from_matrix(transforms[:,:3,:3]).as_rotvec(),moved-centres))
    audit=audit_geometry(prepared,local,pivots,include_overlap=True)
    ratios=[edge['centroid_distance_ratio'] for edge in audit['edges']]
    feasible &= (audit['max_penetration_fraction_beyond_tolerance']==0 and
                 audit['max_overlap_fraction_smaller_volume']<=.01 and min(ratios)>=.9 and max(ratios)<=1.1)
    return dict(params=params,transforms=transforms,global_transform=global_matrix,
        articulated_transforms=local,pivots=pivots,segment_angles_deg=angles,
        axial_separation_mm=separations,geometry=audit,accepted=bool(feasible))


def matrices(params,centres,global_pose=False):
    p=np.asarray(params,float)
    if global_pose:p=np.tile(p.reshape(6),(5,1))
    p=p.reshape(5,6)
    result=np.tile(np.eye(4),(5,1,1));result[:,:3,:3]=Rotation.from_rotvec(p[:,:3]).as_matrix()
    result[:,:3,3]=p[:,3:]
    if not global_pose:result[:,:3,3]+=centres-np.einsum('nij,nj->ni',result[:,:3,:3],centres)
    return result


def extra_metrics(params,case,global_pose=False):
    estimates=matrices(params,case['centres'],global_pose);truth=case['gt_matrices'];rows=[]
    for i,(estimate,gt,points) in enumerate(zip(estimates,truth,case['landmarks'])):
        delta=points@(estimate[:3,:3]-gt[:3,:3]).T+estimate[:3,3]-gt[:3,3]
        centre=case['centres'][i]
        # A fixed synthetic 40-mm vector gives lever-arm error, not a planned pedicle screw.
        entry=centre+np.array([0.,-15.,0.]);tip=entry+np.array([0.,40.,0.])
        def point_error(x):return float(np.linalg.norm((estimate[:3,:3]-gt[:3,:3])@x+estimate[:3,3]-gt[:3,3]))
        a=estimate[:3,:3]@np.array([0.,1.,0.]);b=gt[:3,:3]@np.array([0.,1.,0.])
        rows.append(dict(surface_rms_tre_mm=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
             virtual_entry_error_mm=point_error(entry),virtual_tip_error_mm=point_error(tip),
             virtual_direction_error_deg=float(np.rad2deg(np.arccos(np.clip(a@b,-1,1))))))
    return rows


def degrade(targets,parts,*,seed,photons=None,gaussian=0.,occlusion=0.,endpoint=0,scales=None):
    """Noise and structured attenuation in a declared transmission proxy domain."""
    import torch
    rng=np.random.default_rng(seed);output={};metadata={}
    for view,tensor in targets.items():
        original=tensor.detach().cpu().numpy().astype(np.float64)
        positive=original[original>0]
        scale=scales[view] if scales else np.log(20.)/np.quantile(positive,.995)
        transmission=np.exp(-scale*original)
        covered=np.zeros_like(original,dtype=bool)
        if occlusion:
            image=parts[view][endpoint].detach().cpu().numpy()
            support=image>.01*image.max();ys,xs=np.nonzero(support)
            if len(xs)<10:raise ValueError('Occluded vertebra not visible')
            # A vertical blade proxy; coverage counts actual vertebral support pixels.
            x0=int(rng.integers(xs.min(),max(xs.min()+1,int(np.median(xs)))))
            candidates=[]
            for width in range(1,int(xs.max()-xs.min()+2)):
                rectangle=np.zeros_like(support);rectangle[max(0,ys.min()-2):ys.max()+3,x0:x0+width]=True
                fraction=float((rectangle&support).sum()/support.sum())
                candidates.append((abs(fraction-occlusion),width,fraction))
            _,width,fraction=min(candidates)
            covered[max(0,ys.min()-2):ys.max()+3,x0:x0+width]=True
            transmission[covered]*=.1
        if photons is not None:transmission=rng.poisson(photons*transmission)/photons
        if gaussian:transmission+=rng.normal(0.,gaussian,transmission.shape)
        clipped=(transmission<1e-5)|(transmission>1.)
        observed=-np.log(np.clip(transmission,1e-5,1.))/scale
        if photons is None and not gaussian and not occlusion:
            assert np.allclose(observed,original,atol=1e-7)
        output[view]=torch.as_tensor(observed,dtype=tensor.dtype,device=tensor.device)
        metadata[view]=dict(scale=float(scale),photons=photons,gaussian_sd=gaussian,
            requested_occlusion=occlusion,actual_occlusion=float(fraction) if occlusion else 0.,
            clipped_fraction=float(clipped.mean()),occluded_label=20+endpoint if occlusion else None,
            image_sha256=hashlib.sha256(output[view].cpu().numpy().tobytes()).hexdigest())
    return output,metadata
