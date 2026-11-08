"""Shared validation ownership and bounds; no model or tokenizer required."""
import subprocess
import sys
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import tool_validation as shared
import workload_tools as legacy
import native_workload_tools as native


class ValidationTests(unittest.TestCase):
    def operation(self, **changes):
        return {"op": "tool_loop", "id": "calc", "budget": 256, "max_calls": 16,
                "max_generated": 4096, "wall_seconds": 120, "answer": 10**12,
                **changes}

    def test_compatibility_identity(self):
        for name in ("ProtocolError", "LimitError", "integer", "validate_operation"):
            self.assertIs(getattr(legacy, name), getattr(shared, name))
        self.assertIs(native.ProtocolError, shared.ProtocolError)
        self.assertIs(native.LimitError, shared.LimitError)
        self.assertIs(native.integer, shared.integer)
        self.assertIs(native.custom_validate, shared.validate_operation)

    def test_integer(self):
        for value in (-1000000, 0, 1000000):
            self.assertEqual(shared.integer(value), value)
        for value in (True, False, 1.0, "1", None, -1000001, 1000001):
            with self.subTest(value=value), self.assertRaises(shared.ProtocolError):
                shared.integer(value)
        self.assertEqual(shared.integer(-10**12, 10**12), -10**12)

    def test_operation_bounds(self):
        shared.validate_operation(self.operation())
        for key, upper in (("budget", 256), ("max_calls", 16),
                           ("max_generated", 4096), ("wall_seconds", 120)):
            shared.validate_operation(self.operation(**{key: 1}))
            for value in (0, -1, upper + 1, True, 1.0, "1", None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    shared.validate_operation(self.operation(**{key: value}))
        for value in (-10**12, 0, 10**12):
            shared.validate_operation(self.operation(answer=value))
        for value in (-10**12 - 1, 10**12 + 1, True, 43.0, "43"):
            with self.subTest(answer=value), self.assertRaises(shared.ProtocolError):
                shared.validate_operation(self.operation(answer=value))

    def test_fields_and_ids(self):
        for key in self.operation():
            value = self.operation()
            del value[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                shared.validate_operation(value)
        with self.assertRaises(ValueError):
            shared.validate_operation(self.operation(extra=1))
        for value in ("", "x" * 49, "space id", "é", None, 1):
            with self.subTest(id=value), self.assertRaises(ValueError):
                shared.validate_operation(self.operation(id=value))
        shared.validate_operation(self.operation(id="A_9-" + "x" * 44))
        # Profile/op-name policy stays with callers; extraction must not tighten it.
        shared.validate_operation(self.operation(op="caller-owned"))

    def test_native_import_does_not_load_legacy_executor(self):
        script = ("import sys; sys.path.insert(0, sys.argv[1]); "
                  "import native_workload_tools; "
                  "sys.exit('workload_tools' in sys.modules)")
        flags = ["-O"] if sys.flags.optimize else []
        result = subprocess.run([sys.executable, *flags, "-B", "-c", script, str(TOOLS)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
