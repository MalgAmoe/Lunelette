"""Public real-command route tested only with tiny pinned Python children."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import supervisor as s
from test_manifest import pin, specimen


class RealTests(unittest.TestCase):
    def spec(self, root, script, count=1):
        m = specimen(count)
        for j in m['jobs']:
            j.update(argv=[str(Path(sys.executable).resolve()), '-I', '-B', str(script)],
                     cwd=str(Path(root)/j['id']), sources=[pin(script)], binary=pin(script),
                     interpreter=pin(sys.executable), timeout=2)
        m.update(process_seconds=2, total_seconds=10)
        return m

    def test_real_preserves_inherited_filesize_limit(self):
        import resource
        inherited=resource.getrlimit(resource.RLIMIT_FSIZE)
        with tempfile.TemporaryDirectory(dir=s.HERE) as tmp:
            base=Path(tmp);root=base/'campaign';root.mkdir()
            script=base/'child.py'
            script.write_text("import resource, pathlib\n"
                "if resource.getrlimit(resource.RLIMIT_FSIZE) != %r: raise RuntimeError('global file cap changed')\n"
                "if resource.getrlimit(resource.RLIMIT_CORE) != (0,0): raise RuntimeError('core enabled')\n"
                "pathlib.Path('result').write_text('ok')\n" % (inherited,))
            m=self.spec(root,script)
            result=s.real_launch(root,m,lambda *a:'pass',phase_prepare=lambda *a:None)
            self.assertEqual(result['state'],'complete')

    def test_real_oversized_owned_output_still_stops(self):
        with tempfile.TemporaryDirectory(dir=s.HERE) as tmp:
            base=Path(tmp);root=base/'campaign';root.mkdir()
            script=base/'child.py'
            script.write_text("from pathlib import Path\nPath('result').write_bytes(b'x'*65)\n")
            m=self.spec(root,script)
            with self.assertRaisesRegex(s.Stop,'per-file limit'):
                s.real_launch(root,m,lambda *a:'pass',phase_prepare=lambda *a:None)
            self.assertEqual(json.loads((root/'campaign.result.json').read_text())['state'],'stopped')

    def test_acceptance_labels_exact_command_and_no_retry(self):
        for classification, code in [('pass', 0), ('known-independent-mismatch', 1)]:
            with self.subTest(classification=classification), tempfile.TemporaryDirectory(dir=s.HERE) as tmp:
                base = Path(tmp)
                root = base/'campaign'; root.mkdir()
                script = base/'child.py'
                script.write_text("import os, pathlib, sys\npathlib.Path('result').write_text(os.environ['EXACT'])\nprint('tiny actual child')\nsys.exit(%d)\n" % code)
                m = self.spec(root, script)
                j = m['jobs'][0]
                j.update(env={'EXACT':'pinned'}, accept=classification, accepted_returncodes=[code])
                calls = []
                def prepare(job, work):
                    self.assertEqual(list(work.iterdir()), [])
                    calls.append('prepare')
                def validate(job, work, returncode):
                    calls.append('validate')
                    self.assertEqual(returncode, code)
                    self.assertEqual((work/'result').read_text(), 'pinned')
                    return classification
                result = s.real_launch(root, m, validate, phase_prepare=prepare)
                self.assertEqual(calls, ['prepare', 'validate'])
                self.assertEqual(result['backend'], 'real-command')
                self.assertEqual(result['real_gate'], 'AUTHORIZED')
                receipt = json.loads((root/(j['id']+'.result.json')).read_text())
                self.assertTrue(receipt['real_outputs_verified'])
                self.assertEqual(receipt['classification'], classification)
                with self.assertRaises(s.Stop):
                    s.real_launch(root, m, validate, phase_prepare=prepare)

    def test_all_inputs_before_any_launch_and_prepare_drift(self):
        for drift in ('later-input', 'prepare'):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory(dir=s.HERE) as tmp:
                base = Path(tmp); root = base/'campaign'; root.mkdir()
                script = base/'child.py'; script.write_text("print('tiny')\n")
                m = self.spec(root, script, 2)
                data = base/'input'; data.write_text('tiny input')
                m['jobs'][1]['inputs'] = [pin(data)]
                if drift == 'later-input':
                    data.write_text('changed')
                def prepare(*args):
                    script.write_text("print('drift')\n")
                with patch.object(s.subprocess, 'Popen', side_effect=AssertionError('must not spawn')):
                    with self.assertRaises(s.Stop):
                        s.real_launch(root, m, lambda *args:'pass', phase_prepare=prepare)
                self.assertEqual(list(root.glob('*.launch.json')), [])

    def test_admission_and_callbacks_mandatory(self):
        with tempfile.TemporaryDirectory(dir=s.HERE) as tmp:
            base = Path(tmp); root = base/'campaign'; root.mkdir()
            script = base/'child.py'; script.write_text('pass\n')
            m = self.spec(root, script)
            for key, value in [('artifact_cap', 2*s.GiB+1), ('build_cap', s.GiB+1),
                               ('process_seconds', 1201), ('total_seconds', 14401)]:
                bad = copy.deepcopy(m); bad[key] = value
                with self.assertRaises(s.Stop):
                    s.real_launch(root, bad, lambda *a:'pass', phase_prepare=lambda *a:None)
            with self.assertRaises(s.Stop):
                s.real_launch(root, m, lambda *a:'pass')
            with self.assertRaises(s.Stop):
                s.real_launch(root, m, phase_prepare=lambda *a:None)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
