"""Exercise actual client parsers and Service launch boundaries without processes/models."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import real_workload
import interactive_workload


class StorageDefaults(unittest.TestCase):
    def test_native_launch_matrix(self):
        for module in (real_workload, interactive_workload):
            for selected in (None, "resident-fp32", "compact-fp8-arithmetic", "compact-fp8-lookup"):
                with self.subTest(client=module.__name__, strategy=selected):
                    argv = ["client", "--server", "unused", "--output", "unused",
                            "--native-model", "model", "--native-tokenizer", "tokenizer"]
                    if module is real_workload:
                        argv += ["--scenario", "unused"]
                    if selected:
                        argv += ["--native-strategy", selected]
                    expected = selected or "compact-fp8-arithmetic"
                    with patch.object(sys, "argv", argv), \
                         patch.object(module, "Service", side_effect=RuntimeError("launch seam")) as launch, \
                         patch.object(real_workload, "load_scenario", return_value={"mode": "free_greedy", "operations": []}), \
                         patch.object(interactive_workload.signal, "signal"), \
                         patch.object(interactive_workload, "Output"), \
                         contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(module.main(), 1)
                    self.assertEqual(launch.call_count, 1)
                    kwargs = launch.call_args.kwargs
                    native = kwargs["native_args"]
                    self.assertEqual(native, ["--model", "model", "--tokenizer", "tokenizer",
                                             "--strategy", expected, "--context", "256",
                                             "--prefill-chunk", "256", "--prefill", "mps"])
                    self.assertEqual(kwargs["expected_strategy"], expected)

    def test_rejected_selectors(self):
        for module in (real_workload, interactive_workload):
            for suffix in (["--native-strategy", "bad"], ["--native-strategy"]):
                argv = ["client", "--server", "unused", "--output", "unused"]
                if module is real_workload:
                    argv += ["--scenario", "unused"]
                with patch.object(sys, "argv", argv + suffix), \
                     patch.object(module, "Service") as launch, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.main()
                    self.assertEqual(error.exception.code, 2)
                    launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
