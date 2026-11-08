# Retired application-JSON tool loop: historical protocol

Application-JSON `tool_loop`, its calculator-v1 scenario and dedicated executor tests are retired. Old invocations are unsupported and rejected before service effects, not translated to native tools. Use the [native tool protocol](native-tool-loop.md) for the supported fixed calculator, lookup and observation profiles, and [testing](testing.md#shared-python-tool-validation) for the current validation owner.

The first bounded real-model application-JSON observation failed: fenced invented `tool_result` JSON was rejected, task score stayed null, zero tools executed and no result was appended. The appearance of 43 was not successful tool use. No prompt repair or retry made it pass. The old schema, execution bounds and original failed observation remain historical evidence, separate from native-tool evidence.

## Retrieve the exact original

The exact pre-closeout document is in Git baseline `989efb62c84433e1e2cccba23f2730722d4ac40c` at `docs/tool-loop-v1.md`:

```sh
git show 989efb62c84433e1e2cccba23f2730722d4ac40c:docs/tool-loop-v1.md
```

This requires that Git object to be available in your checkout; a source-only distribution or shallow clone may not contain it. Exact before-images were also preserved in a local development archive outside the repository, with hashes and original link bases. That archive and the external experiment payloads are **not shipped** here; Git retrieval supplies the document, not model files, fixtures or campaign artifacts.

## Historical section links

These compatibility anchors refer to sections of the exact original above, not current invocation instructions.

<a id="client-tool-loop-v1-diagnostic-only"></a>
<a id="protocol"></a>
<a id="scenario-operation-and-limits"></a>
<a id="first-bounded-real-model-observation"></a>
