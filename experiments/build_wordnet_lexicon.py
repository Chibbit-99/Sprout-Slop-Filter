"""Build a licensed, offline lexical lookup from wordnet-db 3.1.14.

Run after `npm install --no-save wordnet-db@3.1.14`:
  python3 experiments/build_wordnet_lexicon.py node_modules/wordnet-db/dict

Only the generated JSON is needed at runtime. It records WordNet lexicographer
classes (and a first-sense synonym only if already present in training).
It does NOT turn unlabelled WordNet words into labelled training examples.
The WordNet notice is in WORDNET_LICENSE.txt.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'node_modules/wordnet-db/dict'
# Freeze the pre-real-issue training vocabulary. Reading the mutable shipped
# model would change which synonyms/lemmas are selected after later retrains.
KNOWN = set(json.loads((ROOT / 'experiments/task_unanchored_model.json').read_text())['vocabulary'])
WORD = re.compile(r'[a-z]{3,20}\Z')
synsets = {}
for pos in ('noun', 'verb', 'adj', 'adv'):
    for line in (SOURCE / ('data.' + pos)).read_text().splitlines():
        if not line[:8].isdigit():
            continue
        fields = line.split('|')[0].split()
        count = int(fields[3], 16)
        lemmas = [fields[4 + 2 * i].lower() for i in range(count)]
        synsets[(pos[0], fields[0])] = (
            pos[0] + fields[1], [w for w in lemmas if WORD.fullmatch(w)]
        )
candidates = {}
for pos in ('noun', 'verb', 'adj', 'adv'):
    for line in (SOURCE / ('index.' + pos)).read_text().splitlines():
        fields = line.split()
        if not fields or not WORD.fullmatch(fields[0]):
            continue
        pointers = int(fields[3])
        tag_count = int(fields[5 + pointers])
        first_offset = fields[6 + pointers]
        record = synsets.get((pos[0], first_offset))
        if not record:
            continue
        word = fields[0]
        # Prefer a direct synonym in the most frequent sense, if known in the
        # labelled task training vocabulary. Never invent a meaning link.
        alias = next((w for w in record[1] if w != word and w in KNOWN), '')
        candidates.setdefault(word, []).append((tag_count, record[0], alias))
chosen = {word: max(senses, key=lambda s: s[0]) for word, senses in candidates.items()}
known_entries = {w: chosen[w] for w in KNOWN & chosen.keys()}
new = sorted(((word, info) for word, info in chosen.items() if word not in KNOWN),
             key=lambda item: (-item[1][0], len(item[0]), item[0]))[:31000]
lexicon = {word: [info[1], info[2]] for word, info in sorted(list(known_entries.items()) + new)}
recognised = len(KNOWN | lexicon.keys()) - 1  # <UNK> is not a word
assert len(new) == 31000 and recognised >= 34020, recognised
path = ROOT / 'experiments/wordnet_lexicon.json'
path.write_text(json.dumps(lexicon, ensure_ascii=False, separators=(',', ':')) + '\n')
print(f'{len(lexicon)} WordNet lemmas, {recognised} recognised tokens including {len(KNOWN)-1} training-observed tokens; {path.stat().st_size} bytes')
