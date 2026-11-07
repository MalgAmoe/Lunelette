"""SCRIPTED native two-tool socket/client integration, not model evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from real_workload import Service, load_scenario, run
from native_workload_tools import load_corpus

SERVER, TOKENIZER = [str(Path(arg).resolve()) for arg in sys.argv[1:3]]
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


class ScriptedInvestigation(unittest.TestCase):
    def test_investigation(self):
        with tempfile.TemporaryDirectory(prefix="scripted-lookup-") as directory:
            service = ObservedService.__new__(ObservedService)
            service.replies, service.generations = [], []
            service.__init__(SERVER, Path(directory) / "output", native_args=[
                "--model", TOKENIZER, "--tokenizer", TOKENIZER,
                "--strategy", "resident-fp32", "--context", "4352",
                "--prefill-chunk", "256", "--prefill", "custom",
                "--tools", "investigation-v13"], request_seconds=10)
            try:
                scenario = load_scenario(Path(__file__).resolve().parents[1] /
                                         "tools/scenarios/investigation-v13.json", 4352)
                report = run(service, scenario)
                self.assertTrue(report["protocol_success"], report)
                self.assertTrue(report["task_success"], report)
                self.assertEqual(report["tool_loops"][0]["tool_calls"], 2)
                self.assertFalse(report["performance_evidence"])
                self.assertEqual(len(service.generations), 3)
                lookup, calculator, final = service.generations
                self.assertEqual(lookup["tool_call"]["name"], "lookup")
                self.assertEqual(calculator["tool_call"]["name"], "calculator")
                self.assertEqual(final["text"], "19 [LARCH-STOCK:P1]")
                accepted = [(body, reply) for method, path, body, status, reply in service.replies
                            if method == "POST" and isinstance(body, dict)
                            and body.get("role") == "tool" and status == 200]
                self.assertEqual(len(accepted), 2)
                self.assertEqual(json.loads(accepted[0][0]["content"]), load_corpus()["larch-stock"])
                self.assertEqual(accepted[1][0]["content"], "19")
                for index, call in enumerate((lookup, calculator)):
                    before = call["session_metadata"]["token_ids"]
                    body, reply = accepted[index]
                    self.assertEqual(body["tool_call_id"], call["id"])
                    suffix = reply["token_ids"][len(before):]
                    self.assertEqual(reply["token_ids"][:len(before)], before)
                    self.assertEqual((suffix[0], suffix[-1]), (7, 8))
                    self.assertTrue(all(t >= 1000 for t in suffix[1:-1]))
                    following = service.generations[index + 1]
                    self.assertEqual(following["session_metadata"]["token_ids"],
                                     reply["token_ids"] + following["request_metadata"]["token_ids"])
                for generation in service.generations:
                    self.assertEqual(generation["pending_tokens"], 1)
                    self.assertEqual(generation["committed_tokens"], generation["logical_tokens"] - 1)
            finally:
                shutdown = service.close()
            self.assertTrue(shutdown)
            self.assertEqual(service.process.returncode, 0)
            self.assertIn("SCRIPTED", (Path(directory) / "output/server.stderr").read_text())


if __name__ == "__main__":
    unittest.main()
