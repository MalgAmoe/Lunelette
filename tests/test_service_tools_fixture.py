"""Optional frozen V13 parity: python this.py C-probe tekken.json.

Requires local mistral_common==1.13.1; no downloads, weights or fixture refresh.
"""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys

from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage, ToolMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Tool, ToolCall, FunctionCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
import mistral_common.tokens.tokenizers.instruct as instruct
import mistral_common.tokens.tokenizers.tekken as tekken


def equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: frozen V13 mismatch")


def official(tokenizer, fixture):
    tools = [Tool.model_validate(tool) for tool in fixture["declaration"]]
    user = UserMessage(content=fixture["prompt"])
    call = AssistantMessage(tool_calls=[ToolCall(id="call", function=FunctionCall(
        name="calculator", arguments=fixture["arguments"]))])
    result = ToolMessage(tool_call_id="call", content=fixture["result_content"])

    def encode(messages):
        return tokenizer.encode_chat_completion(ChatCompletionRequest(
            messages=messages, tools=tools)).tokens

    initial = encode([user])
    full = encode([user, call, result])
    raw = tokenizer.instruct_tokenizer.tokenizer
    # Official assistant encoder, not a hand-built call token sequence.
    call_ids = tokenizer.instruct_tokenizer.encode_assistant_message(
        call, is_before_last_user_message=False)
    result_ids = full[len(initial) + len(call_ids):]
    equal(initial + call_ids + result_ids, full, "official transcript decomposition")
    later = [3, *raw.encode(fixture["later_user"], bos=False, eos=False), 4]
    later_full = encode([user, call, result, AssistantMessage(content="Done."),
                         UserMessage(content=fixture["later_user"])])
    equal(later_full[-len(later):], later, "official later user")
    return dict(initial=initial, assistant_call=call_ids, result=result_ids, full=full,
                special_initial=encode([UserMessage(content=fixture["special_prompt"])]),
                later=later)


def main():
    fixture = json.loads((Path(__file__).parent / "fixtures/calculator-v13.json").read_text())
    provenance = fixture["provenance"]
    equal(importlib.metadata.version("mistral_common"),
          provenance["mistral_common_version"], "library version")
    equal(hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
          provenance["tokenizer_sha256"], "tokenizer hash")
    for module in (instruct, tekken):
        source = Path(inspect.getfile(module))
        equal(hashlib.sha256(source.read_bytes()).hexdigest(),
              provenance["source_sha256"][source.name], source.name)
    vectors = official(MistralTokenizer.from_file(sys.argv[2]), fixture)
    for name, vector in vectors.items():
        equal(vector, fixture[name], f"official {name}")
    lines = subprocess.check_output([sys.argv[1], sys.argv[2]], text=True).splitlines()
    declaration, initial, result, special, later = map(json.loads, lines)
    equal(declaration, fixture["declaration"], "C declaration")
    equal(lines[0], json.dumps(declaration, ensure_ascii=False), "declaration serialization")
    for name, vector in dict(initial=initial, result=result,
                             special_initial=special, later=later).items():
        equal(vector, fixture[name], f"C {name}")
    equal(initial + fixture["assistant_call"] + result, fixture["full"], "C full")
    damaged = list(fixture["full"])
    damaged[len(initial)] += 1
    try:
        equal(damaged, fixture["full"], "deliberate mismatch")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("comparator accepted deliberate mismatch")
    print("frozen official/C V13 vectors and deliberate-mismatch check passed")


if __name__ == "__main__":
    main()
