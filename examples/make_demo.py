"""Generate artificial smoke-test inputs; these are not paper measurements."""
from pathlib import Path
import argparse
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("data/demo"))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(731)
    frozen=args.output/"frozen";frozen.mkdir(exist_ok=True)
    for graph in range(2):
        for draw in range(2):
            n=160;eta=rng.uniform(.02,.8,n)
            b=np.clip(eta+rng.normal(0,.12,n),.005,.995)
            t=np.clip(eta+rng.normal(0,.06,n),.005,.995)
            y=rng.binomial(1,eta);y[::17]=-1
            np.savez_compressed(frozen/(f"demo{graph}_draw{draw}.npz"),
                b=b,t=t,y=y,ids=np.arange(n),graph=np.asarray("demo"+str(graph)),
                draw=np.asarray(draw),reference=np.asarray("Demo"),experiment=np.asarray("demo"))
    n=2000;x=rng.normal(size=(n,12)).astype(np.float32)
    eta=1/(1+np.exp(-(x[:,0]-.6*x[:,1]-.8)))
    y=rng.binomial(1,eta)
    edge_index=np.vstack((rng.integers(n,size=10000),rng.integers(n,size=10000)))
    order=rng.permutation(n);roles={}
    for name,selection in [("train",order[:500]),("calibrate",order[500:700]),
                           ("acquire",order[700:750]),("evaluate",order[750:])]:
        mask=np.zeros(n,bool);mask[selection]=True;roles[name]=mask
    np.savez_compressed(args.output/"graph.npz",x=x,y=y,edge_index=edge_index,**roles)
    print("Wrote artificial frozen cases and graph to",args.output)


if __name__=="__main__":main()
