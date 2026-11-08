"""Official tokenizer and observation reader; no model-library imports."""
import hashlib
import json
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def check_version():
    try:
        actual = version('mistral_common')
    except PackageNotFoundError as error:
        raise RuntimeError('official tokenizer requires mistral_common==1.13.1; use explicit --python; no installation attempted') from error
    if actual != '1.13.1':
        raise RuntimeError('official tokenizer requires mistral_common==1.13.1, found ' + actual)


def environment():
    # Version census, not an immutable installed-package content pin.
    import sys
    from importlib.metadata import distributions
    check_version()
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                packages={d.metadata['Name']: d.version for d in distributions()},
                limitation='installed package contents are not pinned')


TOKENIZER_HASH = '600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843'


class Tokenizer:
    def __init__(self, path):
        check_version()
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_HASH:
            raise ValueError('tokenizer identity')
        self.tokenizer = Tekkenizer.from_file(path)

    def frame(self, text, initial):
        return ([1] if initial else []) + [3] + self.tokenizer.encode(
            text, bos=False, eos=False) + [4]

    def reset_decode(self):
        self.tail = b''

    def decode(self, ids):
        # EOS is a ledger special, never a Tekken byte piece. Like the service,
        # retain incomplete tails across generate requests; EOS with a tail fails.
        tail = getattr(self, 'tail', b'')
        text = ''
        for index, token in enumerate(ids):
            if token == 2:
                if tail or index != len(ids) - 1:
                    raise ValueError('EOS with incomplete UTF8 or subsequent IDs')
                continue
            raw = tail + self.tokenizer.id_to_byte_piece(token)
            complete = 0
            while complete < len(raw):
                first = raw[complete]
                width = (1 if first < 128 else 2 if 0xc2 <= first <= 0xdf else
                         3 if 0xe0 <= first <= 0xef else 4 if 0xf0 <= first <= 0xf4 else 0)
                if not width:
                    raise ValueError('invalid UTF8')
                chunk = raw[complete:complete + width]
                if len(chunk) < width:
                    if any(b & 0xc0 != 0x80 for b in chunk[1:]):
                        raise ValueError('invalid UTF8 tail')
                    break
                decoded = chunk.decode('utf-8', errors='strict')
                if '\0' in decoded:
                    raise ValueError('NUL output')
                text += decoded
                complete += width
            tail = raw[complete:]
        self.tail = tail
        return dict(text=text, incomplete_utf8=bool(tail))


class Evidence(Tokenizer):
    def __init__(self, tokenizer, trace, transport):
        super().__init__(tokenizer)
        self.trace = Path(trace)
        self.transport = transport

    def rows(self):
        if self.trace.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('trace cap')
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def calls(self, identity):
        return [r for r in self.rows() if r['kind'] == 'call' and r['request_id'] == identity]

    def memory_point(self, phase):
        before = len(self.rows())
        status, _ = self.transport.request('GET', '/info')
        rows = self.rows()[before:]
        if status != 200 or not rows or rows[-1]['kind'] != 'memory':
            raise ValueError('fresh idle observation unavailable')
        return dict(rows[-1], phase=phase)

    def finish(self):
        calls = [r for r in self.rows() if r['kind'] == 'call']
        model = [r for r in calls if r['api'] != 'restore']
        return dict(unattributed_calls=sum(not r['request_id'] for r in calls),
                    model_calls=len(model), model_tokens=sum(r['attempted'] for r in model))

