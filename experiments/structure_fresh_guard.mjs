// Run before any fresh-set prediction. Content/label guard, no model loaded.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
import './structure_partition_guard.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const canonical = text => sprout.tokenise(sprout.normalise_text(text))
    .filter(word => /[\p{L}\p{N}]/u.test(word)).join(' ');
const oldFiles = ['../training.json', '../commit_training.json', 'task_baseline_training.json',
    'task_supplement.json', 'commit_supplement.json', 'real_issue_training.json',
    'issue_corrections_training.json', 'real_issue_validation.json', 'real_issue_holdout.json',
    'fresh_issue_selection.json', 'fresh_issue_test.json', 'blind_issue_audit.json',
    'structure_selection.json', 'structure_locked.json', 'structure_confirmation.json',
    'structure_v2_selection.json', 'unfamiliar_cases.json', 'other_contexts.json', 'domain_probes.json', 'transfer_cases.json'];
const old = new Set();
function visit(node) {
    if (!node || typeof node !== 'object') return;
    if (typeof node.text === 'string') old.add(canonical(node.text));
    else for (const v of Object.values(node)) visit(v);
}
for (const file of oldFiles) visit(read(file));
const raw = fs.readFileSync(new URL('structure_fresh_locked.json', import.meta.url));
assert.equal(createHash('sha256').update(raw).digest('hex'),
    '306d044a11ae556ad8d9b2cc229a6d941ec57c4690669601d7c5544976ebe598',
    'locked-set labels/titles changed after opening');
const rows = JSON.parse(raw);
const sources = {
    'golang/go/issues': [85, 49, 5, 31],
    'rails/rails/issues': [24, 20, 1, 3],
    'kubernetes/kubernetes/issues': [21, 13, 0, 8],
    'oven-sh/bun/commit': [100, 97, 3, 0]
};
assert.equal(rows.length, 230);
const urls = new Set(), seen = new Set();
for (const [source, [n, genuine, slop, unclear]] of Object.entries(sources)) {
    const found = rows.filter(row => row.url.startsWith('https://github.com/' + source + '/'));
    assert.equal(found.length, n);
    assert.deepEqual(['genuine', 'slop', 'unclear'].map(label => found.filter(row => row.label === label).length),
        [genuine, slop, unclear]);
    for (const row of found) {
        assert.ok(!urls.has(row.url)); urls.add(row.url);
        assert.equal(row.kind, source.endsWith('issues') ? 'issue' : 'commit');
        assert.match(row.url, source.endsWith('issues') ? /\/issues\/\d+$/ : /\/commit\/[a-f\d]{40}$/);
        if (row.label === 'unclear') { assert.ok(row.reason); continue; }
        const key = canonical(row.text);
        assert.ok(key && !old.has(key) && !seen.has(key), `cross-set overlap: ${row.url}`);
        seen.add(key);
    }
}
console.log('Fresh-set source, label, URL, and canonical exclusion checks passed; no scores opened.');
