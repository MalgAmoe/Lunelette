"""All specimens are model-free; argv is evidence, never the launched argv."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import manifest as m
import supervisor as s


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), size=path.stat().st_size, sha256=s.digest(path))


def specimen(n=1):
    # Phase/mode/block-shaped real specifications; stand-in pinned entry is explicit.
    jobs = []
    for i in range(n):
        phase = ('A1','A2','B','C','D')[i % 5]
        jobs.append(dict(id=phase+'-'+str(i), argv=[str(Path(sys.executable).resolve()), '-I', str(s.HERE/'fake_child.py'), '--phase',phase,'--block','256','--mode','mps','--storage','resident-fp32'], cwd=str(s.HERE), env={'PATH':'/usr/bin:/bin'}, sources=[pin(s.HERE/'fake_child.py')], binary=pin(s.HERE/'fake_child.py'), interpreter=pin(sys.executable), inputs=[], prerequisites=[jobs[-1]['id']] if jobs else [], accept='pass', accepted_returncodes=[0], owners=2 if i<16 else 1, timeout=.5, writers=[dict(name=name,max_bytes=4096 if name != 'result' else 64,writer='supervisor-pipe' if name != 'result' else 'reviewed-direct',contract='bundled bounded writer v1',required=True) for name in ('stdout','stderr','diagnostics','result')]))
    return dict(schema='supervisor-manifest-v1',jobs=jobs,artifact_cap=2*s.MiB,build_cap=0,build_reservations=[],headroom=s.MiB,total_seconds=20,process_seconds=1)


class ManifestTests(unittest.TestCase):
    def test_explicit_output_metadata_reservation(self):
        spec = specimen()
        spec['artifact_cap'] = 8*s.MiB
        for i in range(3000):
            spec['jobs'][0]['writers'].append(dict(name='vectors/C-A2-output-%04d.f32' % i,
                max_bytes=1024, writer='reviewed-direct',
                contract='placeholder only; central producer pin validation required', required=True))
        size = len(json.dumps(spec).encode())
        self.assertGreater(size, 256*1024)
        self.assertLess(size, m.META-8192)
        self.assertEqual(m.validate(spec)['real_gate'], 'CLOSED')

    def test_32_48_serial_fake(self):
        spec=specimen(32)
        admission=m.preflight(spec)
        self.assertEqual((admission['launches'],admission['owners']),(32,48))
        self.assertFalse(admission['inputs_verified'])
        with tempfile.TemporaryDirectory(dir=s.HERE) as root:
            result=s.run_manifest_fake(root,spec)
            self.assertEqual(result['launches'],32)
            self.assertEqual(result['real_gate'],'CLOSED')
            self.assertEqual(len(list(Path(root).glob('*.launch.json'))),32)
            self.assertLess(sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()), admission['artifact_reserved'])
            recorded=json.loads((Path(root)/'campaign.json').read_text())
            self.assertEqual(recorded['manifest'],spec)

    def test_fatal_and_partial_stop_before_dependent(self):
        for mode in ('nonfinite','finitecount','state','memory','partial','flood','sleep'):
            spec=specimen(2)
            if mode=='flood': spec['jobs'][0]['writers'][0]['max_bytes']=128
            start=time.monotonic()
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=s.HERE) as root:
                with self.assertRaises(s.Stop):
                    s.run_manifest_fake(root,spec,{spec['jobs'][0]['id']:mode})
                self.assertLess(time.monotonic()-start,3)
                self.assertFalse((Path(root)/(spec['jobs'][1]['id']+'.launch.json')).exists())
                work=Path(root)/spec['jobs'][0]['id']
                self.assertIn(b'fake stdout', (work/'stdout').read_bytes())
                self.assertIn(b'fake stderr', (work/'stderr').read_bytes())
                receipt=json.loads((Path(root)/(spec['jobs'][0]['id']+'.result.json')).read_text())
                if mode=='partial':
                    self.assertTrue((work/'stdout').read_bytes().endswith(b'without newline'))
                    self.assertTrue((work/'stderr').read_bytes().endswith(b'without newline'))
                    self.assertEqual(receipt['returncode'],7)
                if mode=='flood':
                    self.assertEqual((work/'stdout').stat().st_size,128)
                    self.assertTrue(receipt['streams_truncated'])

    def test_schema_budget_dependencies_and_closed_gate(self):
        for change in ('owners','artifact','build','argv','dependency','pin','timeout'):
            spec=specimen(32)
            if change=='owners': spec['jobs'][0]['owners']=3
            if change=='artifact': spec['artifact_cap']=1
            if change=='build': spec['build_cap']=s.GiB+1
            if change=='argv': spec['jobs'][0]['argv'][0]='/unbound'
            if change=='dependency': spec['jobs'][0]['prerequisites']=['later']
            if change=='pin': spec['jobs'][0]['binary']['sha256']='bad'
            if change=='timeout': spec['jobs'][0]['timeout']=2
            with self.subTest(change=change), self.assertRaises(m.ManifestError): m.validate(spec)
        spec=specimen(); spec['artifact_cap']=2*s.GiB; spec['build_cap']=s.GiB
        m.validate(spec)
        with tempfile.TemporaryDirectory(dir=s.HERE) as root:
            with self.assertRaises(s.Stop): s.run_manifest_fake(root,spec)
            self.assertFalse(list(Path(root).iterdir()))
        with self.assertRaises(s.Stop): s.real_launch(spec, execution_authorized=True)

    def test_prerequisite_classification(self):
        spec=specimen(2)
        with tempfile.TemporaryDirectory(dir=s.HERE) as root:
            with self.assertRaises(s.Stop): s.run_manifest_fake(root,spec,{spec['jobs'][0]['id']:'known'})
            self.assertFalse((Path(root)/(spec['jobs'][1]['id']+'.launch.json')).exists())
        spec['jobs'][0]['accept']='known-independent-mismatch'
        with tempfile.TemporaryDirectory(dir=s.HERE) as root:
            self.assertEqual(s.run_manifest_fake(root,spec,{spec['jobs'][0]['id']:'known'})['launches'],2)

    def test_latest_pin_drift(self):
        spec=specimen()
        original=m.verify
        count=[0]
        def drift(p):
            count[0]+=1
            if count[0]>3: raise m.ManifestError('latest source drift')
            return original(p)
        with tempfile.TemporaryDirectory(dir=s.HERE) as root, patch.object(m,'verify',side_effect=drift):
            with self.assertRaises(s.Stop): s.run_manifest_fake(root,spec)
            self.assertFalse(list(Path(root).glob('*.launch.json')))

    def test_drift_during_child_and_prelaunch_deadline(self):
        original=m.verify
        for mode in ('drift','deadline'):
            spec=specimen()
            if mode=='deadline': spec['total_seconds']=.02
            count=[0]
            def changed(p):
                count[0]+=1
                if mode=='drift' and count[0]>6: raise m.ManifestError('running source drift')
                if mode=='deadline' and count[0]>3: time.sleep(.03)
                return original(p)
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=s.HERE) as root, patch.object(m,'verify',side_effect=changed):
                with self.assertRaises(s.Stop): s.run_manifest_fake(root,spec,{spec['jobs'][0]['id']:'sleep'})
                self.assertEqual(bool(list(Path(root).glob('*.launch.json'))),mode=='drift')
                self.assertEqual(json.loads((Path(root)/'campaign.result.json').read_text())['state'],'stopped')

    def test_real_output_paths_and_failed_prerequisite_preflight(self):
        spec=specimen()
        spec['jobs'][0]['writers'].append(dict(name='vectors/endpoint.f32', max_bytes=32768, writer='reviewed-direct', contract='future fixed vector census: 8192*4', required=True))
        with tempfile.TemporaryDirectory(dir=s.HERE) as root:
            with self.assertRaisesRegex(s.Stop, 'missing required output'):
                s.run_manifest_fake(root,spec)
            self.assertFalse((Path(root)/spec['jobs'][0]['id']/'vectors/endpoint.f32').exists())
        spec['jobs'][0]['prerequisites']=['missing-gate']
        with tempfile.TemporaryDirectory(dir=s.HERE) as root:
            with self.assertRaises(s.Stop): s.run_manifest_fake(root,spec)
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_input_pins_not_opened_by_default(self):
        spec=specimen()
        spec['jobs'][0]['inputs']=[dict(path='/absent/model',sha256='0'*64,size=99)]
        self.assertFalse(m.preflight(spec)['inputs_verified'])
        with self.assertRaises(m.ManifestError): m.preflight(spec,verify_inputs=True)


if __name__=='__main__': unittest.main()
