"""Seven-cell manifest composition, preserving reviewed writer and comparison contracts."""
import runner as r

def compose(root, build_root, model, tokenizer, sources, pin):
    contracts = r.json.loads((r.HERE/'contracts.json').read_text())
    def writer(name, maximum, required=True, pipe=False):
        return dict(name=name,max_bytes=maximum,required=required,
                    writer='supervisor-pipe' if pipe else 'reviewed-direct',
                    contract='original bounded stream writer' if pipe else 'reviewed probe exact exclusive writer / suite bounded callback; see README.md')
    jobs=[]
    cells=[('short-compact-on','short',256,'compact','on','normal'),
           ('short-resident-on','short',256,'resident','on','normal'),
           ('short-compact-off','short',256,'compact','off','normal'),
           ('short-cancel','short',128,'compact','on','cancel'),
           ('long-resident256','long',256,'resident-fp32',None,None),
           ('long-compact256','long',256,'compact-fp8-arithmetic',None,None),
           ('long-compact128','long',128,'compact-fp8-arithmetic',None,None)]
    checker=r.helpers()
    for name,kind,block,storage,policy,mode in cells:
        build=build_root
        binary=pin(build/('runtime-probe' if kind=='long' else 'short-linked'))
        out=root/name
        writers=[writer(n,2*1024**2 if n!='diagnostics' else 16384,pipe=True) for n in ('stdout','stderr','diagnostics')]
        writers.append(writer('validation.json',128*1024))
        writers.append(writer('timing-start.json',4096))
        if kind=='short':
            argv=[binary['path'],model['path'],tokenizer['path'],str(block),'mps',storage,str(out),policy,mode]
            names=contracts[mode+'-'+policy]['files']
            for f in ('failure-source.txt','failure.failed-actual.f32','failure.failed-actual-kv.sha256','failure.failed-expected-kv.sha256'):
                writers.append(writer(f,524288 if f.endswith('.f32') else 4096,False))
        else:
            argv=[binary['path'],'--execute-authorized-LONG',model['path'],str(block),'mps',storage,str(out),'--ack-model-load']
            names=[op['file'] for op in checker.schedule(block)['operations'] if op['api']=='snapshot']
            writers.append(writer('stdout.log',2*1024**2))
            writers.append(writer('failure-logits.f32',524288,False))
        writers.extend(writer(f,524288) for f in names)
        jobs.append(dict(id=name,argv=argv,cwd=str(out),env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},sources=sources,binary=binary,interpreter=None,inputs=[model,tokenizer],prerequisites=[jobs[-1]['id']] if jobs else [],accept='pass',accepted_returncodes=[0],owners=1,timeout=1200,writers=writers))
    campaign=dict(schema='supervisor-manifest-v1',jobs=jobs,artifact_cap=256*1024**2,build_cap=0,build_reservations=[],headroom=1024**3,total_seconds=14400,process_seconds=1200)
    return campaign
