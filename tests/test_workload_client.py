"""Shared client ownership and import-only contracts (also active under -O)."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import workload_client
import real_workload
import interactive_workload


class ClientOwnership(unittest.TestCase):
    def test_shared_identity_and_compatibility(self):
        for name in ("Service", "AdmissionError", "strict_object"):
            owned = getattr(workload_client, name)
            self.assertEqual(owned.__module__, "workload_client")
            self.assertIs(getattr(real_workload, name), owned)
            self.assertIs(getattr(interactive_workload, name), owned)

    def test_strict_object(self):
        self.assertEqual(json.loads('{"a":1}', object_pairs_hook=workload_client.strict_object), {"a": 1})
        with self.assertRaisesRegex(ValueError, "duplicate field a"):
            json.loads('{"a":1,"a":2}', object_pairs_hook=workload_client.strict_object)

    def test_dependency_direction(self):
        tree = ast.parse((TOOLS / "workload_client.py").read_text())
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertTrue(imports.isdisjoint({"real_workload", "interactive_workload", "workload_tools", "native_workload_tools", "argparse"}))
        for name in ("run", "run_once", "validate_scenario", "main"):
            self.assertFalse(hasattr(workload_client, name))

    def test_imports_do_not_launch_or_parse_cli(self):
        script = '''import argparse, subprocess, sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
with patch.object(subprocess, "Popen", side_effect=RuntimeError("launch on import")), patch.object(argparse.ArgumentParser, "parse_args", side_effect=RuntimeError("CLI on import")):
    import workload_client, interactive_workload, real_workload
'''
        for flags in ([], ["-O"]):
            result = subprocess.run([sys.executable, *flags, "-B", "-c", script, str(TOOLS)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
