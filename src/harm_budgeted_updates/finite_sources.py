"""Generate the finite-binomial inputs of the known-probability experiment.

The random-number order and default factorial settings match the executed
finite-source study. True probabilities are saved for later diagnostics only.
Target and interval estimation use the binomial counts and reference values.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np
from .finite_intervals import clopper_pearson_table, estimate_from_counts


def generate(output, draws=20):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(draws=draws,rarities=[1e-4,1e-3,1e-2],groups=16,
        vertices_per_group=16,trials=100,bias_factors=[.125,.5,1.,2.,8.],
        widths=[.5,1.,2.],affected_fractions=[.25,1.],
        random_state=180320,simultaneous_miscoverage=.05,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol_path=output/'finite_source_protocol.json'
    with protocol_path.open('x',encoding='utf8') as stream:
        json.dump(protocol,stream,indent=2)
    cp=clopper_pearson_table(100,16,.05)
    input_rng=np.random.default_rng(180320)
    inputs={};settings={}
    conditions=list(itertools.product(protocol['bias_factors'],protocol['widths'],protocol['affected_fractions']))
    repeat=lambda value:np.repeat(value,16)
    for rindex,rarity in enumerate(protocol['rarities']):
        for draw in range(draws):
            base=f'r{rindex}__d{draw:02d}'
            bg=np.r_[rarity*input_rng.uniform(.5,1.5,8),input_rng.uniform(.15,.45,8)]
            etag=np.r_[bg[:8],bg[8:]+.2]
            counts=input_rng.binomial(100,etag)
            b,eta=repeat(bg),repeat(etag)
            for name,value in [('b',b),('eta',eta),('group_reference',bg),('group_eta',etag),('source_counts',counts)]:
                inputs[base+'__'+name]=value
            for factor,width,concentration in conditions:
                key=base+f'__f{factor:g}__w{width:g}__c{concentration:g}'
                tg,lg,ug,shift=estimate_from_counts(bg,counts,100,cp,np.log(factor),width,concentration)
                for name,value in [('target',repeat(tg)),('lower',repeat(lg)),('upper',repeat(ug)),('source_logit_bias',repeat(shift))]:
                    settings[key+'__'+name]=value
    np.savez_compressed(output/'finite_source_inputs_v18.npz',**inputs)
    np.savez_compressed(output/'finite_source_targets_intervals_v18.npz',**settings)
    return dict(laws=3*draws,conditions=3*draws*len(conditions))


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=20)
    args=parser.parse_args()
    if args.draws<1:parser.error('--draws must be positive')
    print(json.dumps(generate(args.output,args.draws)))


if __name__=='__main__':cli()
