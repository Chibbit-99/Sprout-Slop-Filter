// Data-only guard: source/label counts, URL provenance, protected-row and
// cross-partition canonical overlap. No model checkpoint is opened here.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = name => JSON.parse(fs.readFileSync(new URL(name, import.meta.url), 'utf8'));
const canonical = text => sprout.tokenise(sprout.normalise_text(text))
    .filter(word => /[\p{L}\p{N}]/u.test(word)).join(' ');
const prior = new Set(), priorFiles = ['../training.json', '../commit_training.json',
    'task_baseline_training.json', 'task_supplement.json', 'commit_supplement.json',
    'real_issue_training.json', 'issue_corrections_training.json',
    'real_issue_validation.json', 'real_issue_holdout.json',
    'fresh_issue_selection.json', 'fresh_issue_test.json', 'blind_issue_audit.json',
    'structure_selection.json', 'structure_locked.json', 'structure_confirmation.json',
    'structure_fresh_locked.json', 'structure_v2_selection.json',
    'unfamiliar_cases.json', 'other_contexts.json', 'domain_probes.json', 'transfer_cases.json'];
function visit(node) {
    if (!node || typeof node !== 'object') return;
    if (typeof node.text === 'string') prior.add(canonical(node.text));
    else for (const value of Object.values(node)) visit(value);
}
for (const file of priorFiles) visit(read(file));
const sourceCounts = {
    'relational_development.json': {
        'angular/angular/issues': [22, 20, 1, 1],
        'hashicorp/terraform/issues': [28, 22, 4, 2],
        'fastapi/fastapi/commit': [100, 19, 1, 80]
    },
    'relational_locked.json': {
        'pandas-dev/pandas/issues': [46, 42, 1, 3],
        'electron/electron/commit': [100, 93, 7, 0]
    }
};
const seen = new Set(), urls = new Set();
for (const [file, sources] of Object.entries(sourceCounts)) {
    const rows = read(file);
    assert.equal(rows.length, Object.values(sources).reduce((a, b) => a + b[0], 0));
    for (const [source, counts] of Object.entries(sources)) {
        const group = rows.filter(row => row.url.startsWith('https://github.com/' + source + '/'));
        assert.deepEqual([group.length, ...['genuine', 'slop', 'unclear'].map(label =>
            group.filter(row => row.label === label).length)], counts, source);
        for (const row of group) {
            assert.ok(!urls.has(row.url), 'duplicate URL: ' + row.url); urls.add(row.url);
            assert.equal(row.kind, source.endsWith('issues') ? 'issue' : 'commit');
            assert.match(row.url, source.endsWith('issues') ? /\/issues\/\d+$/ : /\/commit\/[a-f\d]{40}$/);
            if (row.label === 'unclear') { assert.ok(row.reason); continue; }
            const key = canonical(row.text);
            assert.ok(key && !prior.has(key) && !seen.has(key), 'duplicate ' + row.url + ': ' + key);
            seen.add(key);
        }
    }
}
for (const name of ['relational_development_pairs.json', 'relational_locked_pairs.json']) {
    const rows = read(name);
    assert.equal(rows.length, name.includes('development') ? 16 : 20);
    const pairs = new Map();
    for (const row of rows) {
        const key = canonical(row.text);
        assert.ok(key && !prior.has(key) && !seen.has(key), 'pair duplicate: ' + row.text);
        seen.add(key);
        if (!pairs.has(row.pair)) pairs.set(row.pair, new Set());
        pairs.get(row.pair).add(row.label);
    }
    assert.ok([...pairs.values()].every(labels => labels.size === 2 && labels.has('genuine') && labels.has('slop')));
}
const guards = read('relational_training_guards.json');
assert.equal(guards.length, 24);
assert.deepEqual(['genuine', 'slop'].map(label => guards.filter(row => row.label === label).length), [12, 12]);
for (const row of guards) {
    const key = canonical(row.text);
    assert.ok(key && !prior.has(key) && !seen.has(key), 'training guard overlapped evaluation: ' + row.text);
    seen.add(key);
}
// After the first evaluation these digests make changed labels fail loudly;
// they do not substitute for independent annotation or blind evaluation.
for (const [name, expected] of [
    ['relational_locked.json', 'c6c0205264f350ef05ad59e5d7abee728bb7763f111d20103342af0a6e1ce60b'],
    ['relational_locked_pairs.json', '1a5d40166e6978c795bbee5d2f4c17dd9750866d3098c5535bf8e713ab693492']
]) {
    assert.equal(createHash('sha256').update(fs.readFileSync(new URL(name, import.meta.url))).digest('hex'), expected);
}
console.log('Relational source, label, URL, canonical and pair partition checks passed; no scoring in guard.');
