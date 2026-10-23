"""Operator-aware data/continuity fits, isolated from historical training outputs."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from observations import torch_observe


@dataclass
class NeuralField:
    net: object
    scales: dict
    device: str

    def predict(self, points_m, time_ms, chunk=16384):
        import torch
        result = []
        s = self.scales
        with torch.no_grad():
            for start in range(0, len(points_m), chunk):
                x = torch.as_tensor((points_m[start:start+chunk]-s['origin'])/s['length'],
                                    dtype=torch.float32, device=self.device)
                t = torch.full((len(x),1), (time_ms/1000-s['tc'])/s['tau'], device=self.device)
                result.append(self.net(x,t)[:,:3].cpu().numpy().astype(float)*s['velocity'])
        return np.concatenate(result)

    def observe(self, observations):
        result = []
        for time_ms in observations.times_ms:
            values = self.predict(observations.support_m.reshape(-1,3), time_ms)
            result.append(np.einsum('nqc,q->nc',values.reshape(observations.support_m.shape), observations.weights))
        return np.stack(result)


def fit_neural(P, obs, boundary_xyz, inlet_xyz, inlet_values, arm, steps, seed,
               device='cuda', wphys=.01, lr=.001, obs_batch=128, col_batch=512,
               wall_batch=512, inlet_batch=256, log_every=250,measurements_only=False):
    import torch
    from historical import api_module
    FieldNet=api_module('fields').FieldNet
    if arm not in ('data','cont') or steps < 1:
        raise ValueError('data/cont arm and positive step count required')
    if measurements_only:
        if any(a is not None and len(a) for a in (boundary_xyz,inlet_xyz,inlet_values)):
            raise ValueError('measurements-only fitting must not receive boundary targets')
    elif any(a is None or not len(a) for a in (boundary_xyz,inlet_xyz,inlet_values)):
        raise ValueError('native fitting requires explicit known wall and inlet information')
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    origin = P.X.mean(0)
    length = .5*float(np.ptp(P.X,axis=0).max())
    velocity = max(float(np.quantile(np.linalg.norm(obs.values[:,~obs.validation],axis=2),.995)),1e-6)
    lo,hi = np.array(sorted(obs.times_ms))/1000
    scales = dict(origin=origin,length=length,velocity=velocity,tc=.5*(lo+hi),tau=.5*(hi-lo))
    tensor = lambda a: torch.tensor(np.asarray(a),dtype=torch.float32,device=device)
    support = tensor((obs.support_m-origin)/length)
    weights = tensor(obs.weights)
    values = tensor(obs.values/velocity)
    fit_ids = torch.as_tensor(np.flatnonzero(~obs.validation),device=device)
    val_ids = torch.as_tensor(np.flatnonzero(obs.validation),device=device)
    wall = tensor((boundary_xyz-origin)/length) if not measurements_only else None
    inlet = tensor((inlet_xyz-origin)/length) if not measurements_only else None
    in_values = tensor(inlet_values/velocity) if not measurements_only else None
    geometry = tensor((P.X-origin)/length)
    outlet = tensor((P.outlet_xyz-origin)/length) if not measurements_only else None
    net = FieldNet(seed=seed).to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=lr*.01)
    streams = {name:torch.Generator(device=device).manual_seed(seed+offset)
               for name,offset in [('obs',100),('wall',200),('inlet',300),('col',400),('gauge',500)]}
    def sample(n,k,name):
        return torch.randint(n,(k,),device=device,generator=streams[name])
    started = time.monotonic()
    history = []
    for step in range(steps):
        rows = fit_ids[sample(len(fit_ids),obs_batch,'obs')]
        data_loss = sum(((torch_observe(net,support[rows],weights,t)-values[f,rows])**2).sum(1).mean()
                        for f,t in enumerate([-1.,1.]))/2
        wall_loss = torch.zeros((),device=device)
        if not measurements_only:
            iw = sample(len(wall),wall_batch,'wall')
            tw = 2*torch.rand((wall_batch,1),device=device,generator=streams['wall'])-1
            wall_loss = net(wall[iw],tw)[:,:3].square().sum(1).mean()
        inlet_loss = torch.zeros((),device=device)
        gauge_loss = torch.zeros((),device=device)
        for f,t in ([] if measurements_only else enumerate([-1.,1.])):
            ii = sample(len(inlet),inlet_batch,'inlet')
            inlet_loss = inlet_loss + (net(inlet[ii],torch.full((inlet_batch,1),t,device=device))[:,:3]
                                        -in_values[f,ii]).square().sum(1).mean()
            io = sample(len(outlet),min(256,len(outlet)),'gauge')
            gauge_loss = gauge_loss + net(outlet[io],torch.full((len(io),1),t,device=device))[:,3].mean().square()
        cont_loss = torch.zeros((),device=device)
        if arm == 'cont':
            xc = geometry[sample(len(geometry),col_batch,'col')].detach().requires_grad_(True)
            tc = 2*torch.rand((col_batch,1),device=device,generator=streams['col'])-1
            u = net(xc,tc)[:,:3]
            divergence = sum(torch.autograd.grad(u[:,c].sum(),xc,create_graph=True,retain_graph=True)[0][:,c]
                             for c in range(3))
            cont_loss = divergence.square().mean()
        loss = data_loss+wall_loss+inlet_loss+gauge_loss+wphys*cont_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step()
        scheduler.step()
        if step % log_every == 0 or step == steps-1:
            with torch.no_grad():
                numerator,denominator = 0.,0.
                for f,t in enumerate([-1.,1.]):
                    for start in range(0,len(val_ids),128):
                        j=val_ids[start:start+128]
                        p=torch_observe(net,support[j],weights,t)
                        numerator += float((p-values[f,j]).square().sum())
                        denominator += float(values[f,j].square().sum())
            row=dict(step=step+1,loss=float(loss),data=float(data_loss),wall=float(wall_loss),
                     inlet=float(inlet_loss),cont=float(cont_loss),val_observed_rel_l2=(numerator/denominator)**.5,
                     elapsed_s=time.monotonic()-started)
            history.append(row)
            print({'arm':arm,**row},flush=True)
    net.eval()
    field = NeuralField(net,scales,device)
    return field, dict(steps=steps,history=history,initialization_seed=seed,
                       batch_sizes=dict(observation_voxels=obs_batch,collocation=col_batch,wall=wall_batch,inlet=inlet_batch),
                       seconds=time.monotonic()-started, learning_rate=lr,
                       physics_weight=wphys if arm=='cont' else 0.,
                       boundary_contract='measurements only; no wall/inlet/outlet targets' if measurements_only else 'known wall/inlet and outlet pressure gauge',
                       peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device.startswith('cuda') else 0,
                       fit_policy=obs.metadata['fitting_policy'],checkpoint_policy='fixed terminal step; no hidden-target or validation checkpoint selection')
