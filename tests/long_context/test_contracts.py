import unittest
import runner as r

class TimingTests(unittest.TestCase):
    def test_timing_schema(self):
        metrics={key:12.25 for key in r.MS_KEYS}
        metrics.update({key:42 for key in r.US_KEYS})
        metrics.update(decode_step_duration_us=[1,2], unknown_ms=999)
        r.timing_metrics(metrics)
        self.assertEqual(metrics['decode_step_duration_us'],[0,0])
        self.assertEqual(metrics['unknown_ms'],999)
        self.assertTrue(all(metrics[key]==0 for key in r.MS_KEYS | r.US_KEYS))
        nullable={key:None for key in r.NULL_KEYS}
        r.timing_metrics(nullable)
        self.assertEqual(nullable,{key:None for key in r.NULL_KEYS})

