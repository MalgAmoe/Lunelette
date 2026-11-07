"""Actual checkpoint stub socket orchestration: SERVER TOKENIZER arguments."""
import copy
import json
import subprocess
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from real_workload import Service, load_scenario, run


def check(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def exercise(server, tokenizer):
    scenario = load_scenario(Path(__file__).resolve().parents[1] / "tools/scenarios/checkpoint-repeat-v2.json")
    with tempfile.TemporaryDirectory() as directory:
        for policy in ("on", "off"):
            value = copy.deepcopy(scenario)
            value["prefix_reuse"] = policy
            # Instant CPU logits can intentionally overflow the bounded event queue
            # at 16 tokens. Two tokens exercise partial-state restore, not quality.
            for operation in value["operations"]:
                if operation["op"] == "generate":
                    operation["budget"] = 2
            service = Service(server, Path(directory) / policy, native_args=[
                "--model", tokenizer, "--tokenizer", tokenizer, "--strategy", "resident-fp32",
                "--context", "256", "--prefill-chunk", "256", "--prefill", "custom",
                "--prefix-reuse", policy])
            try:
                result = run(service, value)
                check(result["protocol_success"], result)
                check(result["completed_repetitions"] == 2, result)
                check(result["task_success"] is False, result)
                check(service.info["model_load_count"] == 1, service.info)
                check(result["prefix_reuse"] == policy, result)
                events = [json.loads(line) for line in (service.output / "client.ndjson").read_text().splitlines()]
                generations = [e for e in events if e["kind"] == "stream_result"]
                check(len(generations) == 4, generations)
                for event in generations:
                    work = event["metrics"]["work"]
                    check(work["prefill_calls"] == (policy == "off"), event)
                    check(work["prefill_committed"] == (8 if policy == "off" else 0), event)
                    check(event["request_metadata"]["token_ids"] == [22177, 22177], event)
                check([e["id"] for e in generations] == [
                    "r1-first", "r1-second", "r2-first", "r2-second"], generations)
                controls = [r["result"]["controls"] for r in result["repetitions"]]
                check(controls[0][0]["terminal"]["session_id"] != controls[1][0]["terminal"]["session_id"], controls)
                check(controls[0][0]["checkpoint_id"] != controls[1][0]["checkpoint_id"], controls)
                for save, restore in controls:
                    check(save["ledger"]["token_ids"] == restore["ledger"]["token_ids"], controls)
                    check(restore["terminal"]["cache_restored"] == (policy == "on"), controls)
            finally:
                original = sys.exc_info()[1]
                clean = service.close()
                if original is None:
                    check(clean, "shutdown failed")

            # Exercise CLI startup policy selection as well as the imported API.
            path = Path(directory) / (policy + ".json")
            path.write_text(json.dumps(value))
            output = Path(directory) / (policy + "-cli")
            command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] /
                       "tools/real_workload.py"), "--server", str(Path(server).resolve()),
                       "--scenario", str(path), "--output", str(output),
                       "--native-model", tokenizer, "--native-tokenizer", tokenizer,
                       "--native-prefill", "custom"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            check(completed.returncode == 0, completed.stderr)
            result = json.loads((output / "result.json").read_text())
            check(result["protocol_success"] and result["completed_repetitions"] == 2, result)
            check(result["task_success"] is False and result["prefix_reuse"] == policy, result)


if __name__ == "__main__":
    check(len(sys.argv) == 3, __doc__)
    exercise(sys.argv[1], str(Path(sys.argv[2]).resolve()))
    print("scenario on/off socket checks passed (CPU stubs, not model validation)")
