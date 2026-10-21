"""Original two-layer GraphSAGE and compatibility-training objectives."""
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import expit, logit
import torch
from torch import nn
from .data import load_graph, roles_from_data, save_record, array_sha256

def probability(p):return np.clip(np.asarray(p,float),1e-12,1-1e-12)

def seed(ds,draw,member=0):return int(hashlib.sha256(('gnn-v21|'+ds+'|'+str(draw)+'|'+str(member)).encode()).hexdigest()[:8],16)%2147483647

def affine(raw,y,mask):
 from scipy.optimize import minimize
 z=logit(probability(raw));yy=y[mask]
 def fun(v):
  a=np.exp(-v[0]);zz=a*z[mask]+v[1];d=expit(zz)-yy
  return float(np.mean(np.logaddexp(0,zz)-yy*zz)+.005*np.dot(v,v)),np.array([np.mean(-d*a*z[mask]),np.mean(d)])+.01*v
 r=minimize(fun,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-2,2),(-5,5)])
 return probability(expit(np.exp(-r.x[0])*z+r.x[1])),dict(parameters=r.x.tolist(),success=bool(r.success))

def adj(A, device='cuda'):
 deg=np.asarray(A.sum(1)).ravel();P=sparse.diags(1/np.maximum(deg,1))@A;P=P.tocsr();P.sort_indices()
 return torch.sparse_csr_tensor(torch.tensor(P.indptr,device=device,dtype=torch.int64),torch.tensor(P.indices,device=device,dtype=torch.int64),torch.tensor(P.data,device=device,dtype=torch.float32),size=P.shape),P

class SAGE(nn.Module):
 def __init__(self,d):
  super().__init__();self.a=nn.Linear(2*d,32);self.b=nn.Linear(64,2)
 def forward(self,x,px,P):
  h=nn.functional.relu(self.a(torch.cat((x,px),1)));h=nn.functional.dropout(h,p=.3,training=self.training)
  return self.b(torch.cat((h,torch.sparse.mm(P,h)),1))

def train(x,P,px,y,fit,stop,initial,mode='CE',teacher=None,value=1.,weights=None):
 device=x.device;torch.manual_seed(initial);torch.cuda.manual_seed_all(initial);m=SAGE(x.shape[1]).to(device);opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=.0005)
 ii=torch.tensor(np.flatnonzero(fit),device=device);aa=torch.tensor(np.flatnonzero(stop),device=device);yy=torch.tensor(np.maximum(y,0),device=device,dtype=torch.long)
 bb=None if teacher is None else torch.tensor(teacher,device=device,dtype=torch.float64)
 best=float('inf');best_epoch=0;state=None;history=[];start=time.perf_counter()
 for epoch in range(80):
  m.train();opt.zero_grad();z=m(x,px,P);ce=nn.functional.cross_entropy(z[ii],yy[ii])
  if mode=='Distill':
   soft=value*yy[ii].double()+(1-value)*bb[ii];loss=nn.functional.binary_cross_entropy_with_logits((z[ii,1]-z[ii,0]).double(),soft)
  elif mode=='FD':
   temperature=100.;old=torch.stack((torch.zeros_like(bb[ii]),torch.logit(bb[ii])),1);kd=nn.functional.kl_div(nn.functional.log_softmax(z[ii].double()/temperature,dim=1),nn.functional.softmax(old/temperature,dim=1),reduction='none').sum(1)*temperature**2
   focus=1.+5.*((bb[ii]>=.5).long()==yy[ii]).double();loss=ce+value*(focus*kd).mean()
  else:loss=ce
  loss.backward();opt.step();m.eval()
  with torch.no_grad():criterion=float(nn.functional.cross_entropy(m(x,px,P)[aa],yy[aa]))
  history.append([epoch,float(loss),criterion])
  if criterion<best-1e-7:best=criterion;best_epoch=epoch;state={k:v.detach().cpu().clone()for k,v in m.state_dict().items()}
  if epoch-best_epoch>=12:break
 m.load_state_dict(state);m.eval()
 with torch.no_grad():raw=probability(torch.softmax(m(x,px,P),1)[:,1].cpu().numpy())
 return m,raw,dict(mode=mode,strength=float(value),initialization=initial,epochs=len(history),best_epoch=best_epoch,stop_NLL=best,history=history,seconds=time.perf_counter()-start,fit_count=int(fit.sum()),stopping_count=int(stop.sum()))


def calibration_roles(masks, dataset, draw):
    order = np.random.default_rng(seed(dataset, draw, 777)).permutation(np.flatnonzero(masks["calibrate"]))
    if len(order) != 200:
        raise ValueError("The original GraphSAGE protocol requires 200 calibration labels.")
    result = {}
    for name, indices in (("Astop", order[:60]), ("Acal", order[60:130]), ("Atarget", order[130:])):
        result[name] = np.zeros_like(masks["train"])
        result[name][indices] = True
    return result


def run(data_path, output, dataset, draw, device="cuda", selected_variants=False):
    """Train the deployed model and all six prespecified update objectives."""
    data = load_graph(data_path)
    masks = roles_from_data(data, dataset, draw)
    roles = calibration_roles(masks, dataset, draw)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a CPU run.")
    visible = np.where(masks["train"] | masks["select"] | masks["calibrate"], data["y"], -1)
    if np.any(visible[masks["test"]] != -1):
        raise AssertionError("Evaluation responses entered the fitting view.")
    center = data["x"][masks["train"]].mean(0)
    scale = data["x"][masks["train"]].std(0)
    scale[scale < 1e-6] = 1
    x = torch.tensor(np.clip((data["x"] - center) / scale, -10, 10).astype(np.float32), device=device)
    P, _ = adj(data["adjacency"], device)
    px = torch.sparse.mm(P, x)
    old, raw, old_record = train(x, P, px, visible, masks["train"], roles["Astop"], seed(dataset, draw))
    b, old_calibration = affine(raw, visible, roles["Acal"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(old.state_dict(), output / (dataset + "__draw_" + str(draw) + "__old.pt"))
    del old
    fit = masks["train"] | masks["select"]
    targets, records = {}, {}
    cases = (("CE", "CE", 1.), ("D.25", "Distill", .25), ("D.5", "Distill", .5), ("D.75", "Distill", .75), ("FD.1", "FD", .1), ("FD1", "FD", 1.))
    for name, mode, value in cases:
        fitted, raw, record = train(x, P, px, visible, fit, roles["Astop"], seed(dataset, draw), mode, b, value)
        targets[name], cal = affine(raw, visible, roles["Acal"])
        records[name] = dict(training=record, calibration=cal)
        torch.save(fitted.state_dict(), output / (dataset + "__draw_" + str(draw) + "__" + name + ".pt"))
        del fitted
    selection = None
    if selected_variants:
        choose = roles["Atarget"]
        def nll(value):
            return float(np.mean(-np.where(visible[choose] == 1, np.log(value[choose]), np.log1p(-value[choose]))))
        def kl(value):
            return float(np.mean(b[choose] * np.log(b[choose] / value[choose]) + (1 - b[choose]) * np.log((1 - b[choose]) / (1 - value[choose]))))
        inventory = {name: dict(NLL=nll(value), KL=kl(value)) for name, value in targets.items()}
        inventory["Input"] = dict(NLL=nll(b), KL=0.)
        eligible = [name for name in ("D.25", "D.5", "D.75", "CE", "Input") if inventory[name]["KL"] <= .01]
        selected = min(eligible, key=lambda name: (inventory[name]["NLL"], name))
        focal_selected = min(("FD.1", "FD1"), key=lambda name: (inventory[name]["NLL"], name))
        targets["DistillSelected"] = b if selected == "Input" else targets[selected]
        targets["FDSelected"] = targets[focal_selected]
        selection = dict(distillation=selected, focal_distillation=focal_selected, candidates=inventory)
    evaluate, cal = masks["test"], roles["Acal"]
    for name, t in targets.items():
        filename = dataset + "__draw_" + str(draw) + "__SAGE_" + name + ".npz"
        np.savez_compressed(output / filename, b=b[evaluate], t=t[evaluate], y=data["y"][evaluate], ids=data["ids"][evaluate], graph=np.asarray(dataset), draw=np.asarray(draw), reference=np.asarray("GraphSAGE"), experiment=np.asarray("graphsage_" + name), cal_old=b[cal], cal_new=t[cal], cal_y=visible[cal])
    save_record(output / (dataset + "__draw_" + str(draw) + "__SAGE_training.json"), dict(old=dict(training=old_record, calibration=old_calibration), targets=records, selection=selection, role_hashes={key: array_sha256(value) for key, value in dict(masks, **roles).items()}, evaluation_responses_used_for_fitting=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selected-variants", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, args.dataset, args.draw, args.device, args.selected_variants)


if __name__ == "__main__":
    main()
