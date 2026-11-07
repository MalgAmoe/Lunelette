"""Model-free main check interface: python3 check.py BUILD TOKENIZER."""
import json
import os
from pathlib import Path
import subprocess
import sys

def check(build, tokenizer):
    build=Path(build).resolve()
    for mode,block in (('normal',256),('cancel',128)):
        for policy in ('on','off'):
            out=build/(mode+'-'+policy)
            out.mkdir()
            with (out/'trace.jsonl').open('x') as log:
                subprocess.run([str(build/'short-stub'),'UNUSED',str(Path(tokenizer).resolve()),
                                str(block),'mps','compact',str(out),policy,mode],
                               stdout=log,check=True,timeout=60)
            rows=[json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()]
            budget=next(r for r in rows if r['type']=='token_budget')
            if budget['total_model_tokens'] > (2048 if mode=='cancel' else 1024):
                raise RuntimeError('token ceiling')
            if rows[-1]['type']!='short_complete':
                raise RuntimeError('incomplete')
            print(mode,policy,budget['total_model_tokens'],'stub tokens: PASS (not native evidence)')
    for fault in ('logits','cursor','input_state'):
        out=build/('fault-'+fault)
        out.mkdir()
        with (out/'trace.jsonl').open('x') as log, (out/'stderr').open('x') as err:
            p=subprocess.run([str(build/'short-stub'),'UNUSED',str(Path(tokenizer).resolve()),
                              '256','mps','compact',str(out),'on','normal'],
                             env=dict(os.environ,SHORT_FAULT=fault),stdout=log,stderr=err,timeout=60)
        if p.returncode==0:
            raise RuntimeError('undetected fault '+fault)
        print('fault',fault,'rejected (stub observation only)')
if __name__=='__main__':
    check(*sys.argv[1:])
