# Lunelette project guidance

- Learning and understandable correctness come before optimization.
- Initial scope: Ministral 3 3B, text-only, portable C11 CPU baseline.
- Target development hardware: base Apple M5 with 32 GB unified memory.
- Future scope: Ministral 3 8B/14B, Mac acceleration, and portable deployment.
- Treat ../ds4/ as upstream reference material. Do not modify it as part of Lunelette.
- Do not assume architecture details or checkpoint compatibility: verify against
  the exact model configuration and trusted implementation.
- Python belongs in development/reference tooling, not the C runtime.
- Keep changes small; avoid speculative backend abstractions and dependencies.
- Preserve attribution and license notices when adapting upstream code.
- Run make test and, for numerical/memory changes, make sanitize.
- Never describe the scaffold as working model inference.
- Ask before large model downloads or dependency installation.
