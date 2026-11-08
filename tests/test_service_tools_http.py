"""SCRIPTED actual-socket integration; not inference or model-quality evidence."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from workload_client import Service
from real_workload import load_scenario, run

SERVER, TOKENIZER = map(str, map(Path.resolve, map(Path, sys.argv[1:3])))
del sys.argv[1:3]


class ObservedService(Service):
    def request(self, method, path, body=None):
        status, reply = super().request(method, path, body)
        self.replies.append((method, path, body, status, reply))
        return status, reply

    def generate(self, *args, **kwargs):
        reply = super().generate(*args, **kwargs)
        self.generations.append(reply)
        return reply


class ScriptedHTTP(unittest.TestCase):
    def test_calculator(self):
        with tempfile.TemporaryDirectory(prefix="scripted-tools-") as directory:
            service = ObservedService.__new__(ObservedService)
            service.replies, service.generations = [], []
            service.__init__(SERVER, Path(directory) / "child-output", native_args=[
                "--model", TOKENIZER, "--tokenizer", TOKENIZER,
                "--strategy", "resident-fp32", "--context", "4352",
                "--prefill-chunk", "256", "--prefill", "custom",
                "--tools", "calculator-v13"], request_seconds=10,
                expected_strategy="resident-fp32")
            try:
                scenario = load_scenario(Path(__file__).resolve().parents[1] /
                                         "tools/scenarios/calculator-v13.json", 4352)
                report = run(service, scenario)
                self.assertTrue(report["protocol_success"], report)
                self.assertTrue(report["task_success"], report)
                self.assertEqual(report["tool_loops"][0]["tool_calls"], 1)
                self.assertFalse(report["performance_evidence"])
                self.assertEqual(len(service.generations), 2)
                call, final = service.generations
                call_ids = call["request_metadata"]["token_ids"]
                final_ids = final["request_metadata"]["token_ids"]
                self.assertEqual(call_ids[0], 9)
                self.assertEqual(call_ids.count(32), 1)
                self.assertEqual(call_ids[-1], 2)
                self.assertEqual(final_ids, [1052, 1051, 2])
                self.assertEqual(final["text"].strip(), "43")
                before = call["session_metadata"]["token_ids"]
                self.assertEqual(before[:2], [1, 5])
                self.assertIn(6, before)
                self.assertEqual(before[-len(call_ids):], call_ids)
                accepted = [reply for method, path, body, status, reply in service.replies
                            if method == "POST" and isinstance(body, dict)
                            and body.get("role") == "tool" and status == 200]
                self.assertEqual(len(accepted), 1)
                self.assertEqual(accepted[0]["token_ids"], before + [7, 1052, 1051, 8])
                self.assertEqual(final["session_metadata"]["token_ids"],
                                 accepted[0]["token_ids"] + final_ids)
                for reply in (call, final):
                    self.assertEqual(reply["pending_tokens"], 1)
                    self.assertEqual(reply["committed_tokens"], reply["logical_tokens"] - 1)
            finally:
                # Service owns the child and has bounded terminate/kill/wait cleanup.
                shutdown = service.close()
            self.assertTrue(shutdown)
            self.assertEqual(service.process.returncode, 0)
            self.assertIn("SCRIPTED", (Path(directory) / "child-output/server.stderr").read_text())


if __name__ == "__main__":
    unittest.main()
