"""Explicit tokenizer-only check. Never opens weights or imports model libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driver import frozen, PREP, require
from provider import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer(args.tokenizer)
    for row in frozen().values():
        require(tokenizer.frame(row['text'], row['initial']) == row['ids'], 'official frozen framing')
    negative = json.loads((PREP/'negative.json').read_text())
    text = (PREP/'document.txt').read_text() + (PREP/'new-user.txt').read_text()
    ids = tokenizer.frame(text, True)
    require(ids == negative['ids'] and len(ids) == 4594, 'official capacity-negative IDs')
    require(hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() ==
            negative['ids_sha256'], 'official capacity-negative hash')
    ordinary = tokenizer.tokenizer.encode('café', bos=False, eos=False)
    require(tokenizer.decode(ordinary) == dict(text='café', incomplete_utf8=False), 'official byte decode')
    require(not any(n == 'torch' or n.startswith('transformers') for n in sys.modules), 'model library imported')
    print('official framing and decode: pass; weights never opened')


if __name__ == '__main__':
    main()
