from pathlib import Path
import math
import struct
import subprocess
import unittest
from schedule import schedule, HERE
from check import trace, vector


def synthetic(block):
    lines = []
    length = prefix = 0
    for i, op in enumerate(schedule(block)['operations']):
        lines.append('CALL index=%d api=%s start=%d count=%d ids=%s' %
                     (i,op['api'],op['start'],len(op['ids']),','.join(map(str,op['ids']))))
        if op['api'] == 'reset':
            length = prefix = 0
        elif op['api'] in ('prefill','append'):
            length += len(op['ids'])
        elif op['api'] == 'save':
            prefix = length
        elif op['api'] == 'restore':
            length = prefix
        elif op['api'] == 'snapshot':
            for k in range(52):
                lines.append('ENDKV file=%s prefix=%d buffer=%d sha256=%s' %
                             (op['file'],prefix or length,k,'a'*64))
            if not prefix:
                for j,n in enumerate([0,block+1]+([1,1,4353] if length == 4352 else [])):
                    lines.append('REJECT test=%d count=%d status=-1 diagnostic=synthetic' % (j,n))
        lines.append('RESULT index=%d status=0 length=%d' % (i,length))
    lines.append('CLEANUP checkpoint_destroyed runtime_destroyed host_copy_freed status=0')
    return '\n'.join(lines)


class LongTests(unittest.TestCase):
    def test_budgets_and_replay_partitions(self):
        for b in (128,256):
            p = schedule(b)
            self.assertLessEqual(p['tokens'], 2048 if b == 128 else 20000)
            self.assertEqual(p['tokens'],1672 if b == 128 else 17925)
            self.assertEqual(sum(len(o['ids']) for o in p['operations']),p['tokens'])
            self.assertEqual(sum(o['api']=='restore' for o in p['operations']),2)
            trace(synthetic(b),b)

    def test_trace_faults(self):
        good = synthetic(256)
        for bad in (good.replace('CALL index=0','CALL index=1',1),
                    good.replace('RESULT index=0 status=0 length=0','RESULT index=0 status=0 length=1',1),
                    good.replace('ENDKV file=a.f32 prefix=4096 buffer=51 sha256='+'a'*64,
                                 'ENDKV file=a.f32 prefix=4096 buffer=51 sha256='+'b'*64),
                    good.replace('REJECT test=0 count=0 status=-1','REJECT test=0 count=0 status=0',1)):
            with self.assertRaises(ValueError):
                trace(bad,256)

    def test_vectors_stale_and_nonfinite(self):
        good = struct.pack('<131072f', *([1.0]*131072))
        vector(good,good)
        for index in (0,65536,131071):
            bad = bytearray(good)
            struct.pack_into('<f',bad,index*4,2.0)
            with self.assertRaisesRegex(ValueError,'logits mismatch'):
                vector(bad,good)
        for value in (math.nan,math.inf,-math.inf):
            bad = bytearray(good)
            struct.pack_into('<f',bad,524284,value)
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                vector(bad,bad)
        with self.assertRaises(ValueError):
            vector(good[:-4],good)

    def test_changed_schedule_contract(self):
        a = struct.pack('<131072f', *([1.0001]*131072))
        b = struct.pack('<131072f', *([1.0]*131072))
        vector(a,b,'changed-schedule-frozen-tolerance')
        with self.assertRaises(ValueError):
            vector(a,b)
        with self.assertRaises(ValueError):
            vector(a,b,'tolerant')

    def test_compiled_model_free(self):
        import os
        binary = Path(os.environ['LONG_CONTEXT_BUILD'])/'runtime-probe'
        self.assertTrue(binary.is_file(), 'run build.py build-03 first')
        for flag in ('--dry-run','--planner','--badargs'):
            p = subprocess.run([str(binary),flag],capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,0,p.stderr)
        for args in ([],['--execute-authorized-B'],
                     ['--execute-authorized-LONG','/DO_NOT_OPEN','128','mps','resident-fp32','/tmp','--ack-model-load']):
            p = subprocess.run([str(binary),*args],capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,2,p.stderr)

if __name__ == '__main__':
    unittest.main()
