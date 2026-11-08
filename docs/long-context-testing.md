# Optional reduced real-model regression

This tooling is separate from `make test`: Python is optional, and no model or network is needed to build. macOS with the existing Metal toolchain is required. No production defaults, guards, or checkpoint arithmetic are changed.

```
python3 tools/long_context.py --self-test
python3 tools/long_context.py --build /absolute/fresh/build
python3 tools/long_context.py --build /absolute/fresh/build \
  --weights /absolute/model.safetensors --tokenizer /absolute/tekken.json \
  --out /absolute/fresh/results
```

The final command loads weights: review all delivered changes before launching. No real-model execution is implied by a successful model-free test. Build and result directories must be fresh and exclusively owned; no resume, downloads, or dependency installation. Build receipts hash the original 104-file engine inventory plus all promoted tooling and record compiler/commands/products. Objects are rebuilt from current sources using Make's BUILD override, never copied from archived builds. Source drift invalidates an existing build. Paths and all ancestors are checked before creation without resolving away symlinks (on macOS, use `/private/tmp`, not the `/tmp` symlink). This assumes exclusive ownership; it is not protection against concurrent hostile filesystem replacement.

Build subprocesses use an allowlisted environment, an explicit xcrun-selected compiler and SDK, and explicit Make flags matching production Metal contraction settings (`-ffp-contract=off`). Inherited Make/compiler/toolchain flags are not used. `build.log` persists commands, compiler path/hash/version, SDK selection, environment and compiler output, including failed builds. Individual build commands have 30- or 180-second deadlines. Full Xcode is optional: an unavailable `xcodebuild -version` is recorded; Command Line Tools can build these probes.

Seven serial cells: short compact ON, resident ON, compact OFF, cancellation; long resident B256, compact B256, compact B128. Structural schedules, complete finite 131072-float logits, 52 KV-buffer hashes, restore and independent replay remain required. Resident/compact comparisons are bitwise; OFF partition and B128 use the unchanged reviewed changed-schedule tolerance. Timing normalization only removes explicit documented timing fields and cross-lane interleaving, not state events. Native parity is not an independent numerical oracle or task-quality approval.

The reviewed POSIX supervisor owns one child process group at a time and cleans that group on failure. Limits: 1200 seconds/cell, 14400 total, 256 MiB artifacts, 1 GiB free-space headroom, declared bounded stdout/stderr and sidecar writers. Real children have no RLIMIT_FSIZE (it can break Metal compiler internal files); this is trusted-writer accounting, not a hostile-process sandbox or filesystem quota. Preserve failed artifacts; do not refresh contracts or tolerances to erase failures.

## Provenance and deferred work

Promoted from Lunelette's reviewed M6 lean-validation short/long/suite probes and stage 3 controller observer/supervisor. These are project-authored helpers covered by the repository license/NOTICE; embedded implementation retains its existing notices. `contracts.json` retains reviewed normalized structural contracts, not model-generated numerical goldens. No developer-specific source paths are needed. The observer includes the original unused long controller scenarios: trimming is explicitly deferred to avoid changing assertions during promotion. A future trim should prove stub traces unchanged before acceptance. Model-free tests cover the long schedule, vector/checker faults, argument admission, and supervisor failures; the promoted seven-cell tool also passed one fresh real-model run during M6 packaging. Main independently rechecked all vectors/state contracts and 119 unchanged artifacts. This evidence is source-bound; it does not certify future builds. The actual stub runs use a synthetic tokenizer whose IDs differ from the official tokenizer: they do not establish equality to the frozen official-token contracts. Separate checker unit tests copy normalized contracts and supply synthetic finite vectors and stub input-state hashes. They exercise structural/hash/vector rejection, including refusal to classify stub hashes as native KV evidence; these are checker coverage only, not independently observed contract or numerical evidence.
