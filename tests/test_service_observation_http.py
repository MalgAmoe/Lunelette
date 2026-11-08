"""Bounded scripted native/session/HTTP test: no inference or client scenario changes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from workload_client import Service
from real_workload import load_scenario, run

SERVER, TOKENIZER = [str(Path(arg).resolve()) for arg in sys.argv[1:3]]
del sys.argv[1:3]


class ScriptedObservation(unittest.TestCase):
    def test_scenario_client(self):
        with tempfile.TemporaryDirectory(prefix="scripted-observation-client-") as directory:
            service = Service(SERVER, Path(directory) / "output", native_args=[
                "--model", TOKENIZER, "--tokenizer", TOKENIZER,
                "--strategy", "resident-fp32", "--context", "4352",
                "--prefill-chunk", "256", "--prefill", "custom",
                "--tools", "observation-v13"], request_seconds=10)
            try:
                scenario = load_scenario(Path(__file__).resolve().parents[1] /
                                         "tools/scenarios/observation-v13.json", 4352)
                report = run(service, scenario)
                self.assertTrue(report["protocol_success"], report)
                self.assertTrue(report["task_success"], report)
                self.assertEqual(report["tool_loops"][0]["tool_calls"], 1)
                self.assertEqual(report["tool_loops"][0]["answer"], "pump-A 84 ALERT")
                self.assertFalse(report["performance_evidence"])
            finally:
                shutdown = service.close()
            self.assertTrue(shutdown)
            self.assertEqual(service.process.returncode, 0)
            self.assertIn("SCRIPTED", (Path(directory) / "output/server.stderr").read_text())

    def test_observation(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/observation-v13.json").read_text())
        with tempfile.TemporaryDirectory(prefix="scripted-observation-") as directory:
            service = Service(SERVER, Path(directory) / "output", native_args=[
                "--model", TOKENIZER, "--tokenizer", TOKENIZER,
                "--strategy", "resident-fp32", "--context", "4352",
                "--prefill-chunk", "256", "--prefill", "custom",
                "--tools", "observation-v13"], request_seconds=10)
            try:
                status, info = service.request("GET", "/info")
                self.assertEqual(status, 200)
                self.assertEqual(info["conversation_protocol"], "native-observation-v13")
                self.assertTrue(info["tools"])
                self.assertEqual(service.request("POST", "/sessions", {})[0], 200)
                self.assertEqual(service.request("POST", "/sessions/1/checkpoint", {"id": "cp"})[0], 501)
                self.assertEqual(service.request("POST", "/sessions/1/messages", {
                    "role": "user", "content": fixture["prompt"]})[0], 200)
                call = service.generate(1, "obs", 64)
                self.assertEqual(call["tool_call"]["name"], "lookup")
                self.assertEqual(call["tool_call"]["arguments"], {"document_id": "pump-readings"})
                self.assertEqual(call["request_metadata"]["token_ids"], fixture["vectors"]["call1"])
                self.assertEqual(call["session_metadata"]["token_ids"],
                                 fixture["vectors"]["initial"] + fixture["vectors"]["call1"])
                body = {"role": "tool", "tool_call_id": "stale", "content": fixture["calls"][0]["result"]}
                self.assertEqual(service.request("POST", "/sessions/1/messages", body)[0], 409)
                body["tool_call_id"] = "obs"
                status, accepted = service.request("POST", "/sessions/1/messages", body)
                self.assertEqual(status, 200)
                self.assertEqual(accepted["token_ids"], fixture["vectors"]["full"])
                final = service.generate(1, "final", 64)
                self.assertEqual(final["text"], "pump-A 84 ALERT")
                self.assertEqual(final["session_metadata"]["token_ids"],
                                 accepted["token_ids"] + final["request_metadata"]["token_ids"])
                for generation in (call, final):
                    self.assertEqual(generation["pending_tokens"], 1)
                    self.assertEqual(generation["committed_tokens"], generation["logical_tokens"] - 1)
                self.assertEqual(service.request("POST", "/sessions/1/reset", {})[0], 200)
                self.assertEqual(service.request("POST", "/sessions/1/messages", body)[0], 409)
            finally:
                shutdown = service.close()
            self.assertTrue(shutdown)
            self.assertEqual(service.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
