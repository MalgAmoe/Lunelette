"""Frozen official 1.13.1 observation lookup parity. No downloads or fixture refresh."""
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


def check(condition, label):
    if not condition:
        raise RuntimeError(label)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def official(tokenizer, fixture):
    tools = [Tool.model_validate(t) for t in fixture["declaration"]]
    messages = [UserMessage(content=fixture["prompt"])]
    def encode():
        return tokenizer.encode_chat_completion(ChatCompletionRequest(
            messages=messages, tools=tools)).tokens
    vectors = {"initial": encode()}
    for i, call in enumerate(fixture["calls"], 1):
        assistant = AssistantMessage(tool_calls=[ToolCall(id=f"call{i}", function=FunctionCall(
            name=call["name"], arguments=call["arguments"]))])
        call_ids = tokenizer.instruct_tokenizer.encode_assistant_message(
            assistant, is_before_last_user_message=False)
        before = encode()
        messages.extend([assistant, ToolMessage(tool_call_id=f"call{i}", content=call["result"])])
        after = encode()
        check(after[:len(before) + len(call_ids)] == before + call_ids, "observation fixture comparison failed")
        vectors[f"call{i}"] = call_ids
        vectors[f"result{i}"] = after[len(before) + len(call_ids):]
    vectors["full"] = encode()
    return vectors


def main():
    fixture = json.loads((Path(__file__).parent / "fixtures/observation-v13.json").read_text())
    check(importlib.metadata.version("mistral_common") == "1.13.1", "observation fixture comparison failed")
    check(hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest() == fixture["provenance"]["tokenizer_sha256"], "observation fixture comparison failed")
    for module in (instruct, tekken):
        path = Path(inspect.getfile(module))
        check(hashlib.sha256(path.read_bytes()).hexdigest() == fixture["provenance"]["source_sha256"][path.name], "observation fixture comparison failed")
    vectors = official(MistralTokenizer.from_file(sys.argv[2]), fixture)
    check(vectors == fixture["vectors"], "observation fixture comparison failed")
    for name, vector in vectors.items():
        check(digest(vector) == fixture["vector_sha256"][name], "observation fixture comparison failed")
    lines = subprocess.check_output([sys.argv[1], sys.argv[2]], text=True).splitlines()
    declaration, *arrays = map(json.loads, lines)
    check(declaration == fixture["declaration"], "observation fixture comparison failed")
    check(lines[0] == json.dumps(declaration), "observation fixture comparison failed")
    check(dict(zip(("initial", "call1", "result1", "full"), arrays)) == vectors, "observation fixture comparison failed")
    damaged = list(arrays[-1])
    damaged[0] += 1
    check(digest(damaged) != fixture["vector_sha256"]["full"], "observation fixture comparison failed")
    print("frozen official/C observation vectors and mismatch check passed")


if __name__ == "__main__":
    main()
