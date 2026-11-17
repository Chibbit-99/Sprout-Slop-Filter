// Data-only guard: it was run before fitting/scoring the correction candidate.
// Re-running it does not score titles; the test has since been opened once.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const canonical = text => sprout.tokenise(sprout.normalise_text(text))
    .filter(word => /[\p{L}\p{N}]/u.test(word)).join(' ');
const historic = [
    '../training.json', '../commit_training.json', 'task_baseline_training.json',
    'task_supplement.json', 'commit_supplement.json', 'real_issue_training.json',
    'real_issue_validation.json', 'real_issue_holdout.json', 'unfamiliar_cases.json',
    'other_contexts.json', 'domain_probes.json', 'transfer_cases.json'
];
const prior = new Set();
function protect(node) {
    if (!node || typeof node !== 'object') return;
    if (typeof node.text === 'string') prior.add(canonical(node.text));
    else for (const value of Object.values(node)) protect(value);
}
for (const file of historic) protect(read(file));
assert.deepEqual(read('issue_corrections_training.json'),
    read('blind_issue_audit.json').filter(row => row.label !== 'unclear'),
    'corrections must retain their pre-scoring audit titles/labels');
const groups = [
    { file: 'issue_corrections_training.json', repos: ['fastify/fastify', 'eslint/eslint'],
        counts: { genuine: 90, slop: 21, unclear: 0 }, length: 111 },
    { file: 'fresh_issue_selection.json', repos: ['babel/babel'],
        counts: { genuine: 27, slop: 2, unclear: 5 }, length: 34 },
    { file: 'fresh_issue_test.json', repos: ['react/react', 'microsoft/TypeScript', 'microsoft/vscode'],
        counts: { genuine: 114, slop: 21, unclear: 26 }, length: 161 }
];
const seen = new Map(), valid = [];
for (const group of groups) {
    const rows = read(group.file), counts = { genuine: 0, slop: 0, unclear: 0 };
    assert.equal(rows.length, group.length, group.file);
    for (const row of rows) {
        assert.ok(group.repos.some(repo => row.url.startsWith('https://github.com/' + repo + '/issues/')),
            'source URL: ' + row.url);
        assert.ok(Object.hasOwn(counts, row.label), 'label: ' + row.url);
        counts[row.label]++;
        if (row.label === 'unclear') { assert.ok(row.reason, 'excluded reason: ' + row.url); continue; }
        const key = canonical(row.text);
        assert.ok(key && !prior.has(key) && !seen.has(key), 'cross-partition duplicate: ' + row.url);
        seen.set(key, group.file);
        valid.push({ ...row, group: group.file, terms: new Set(key.split(' ')) });
    }
    assert.deepEqual(counts, group.counts, 'frozen labels: ' + group.file);
}
// Conservative cross-partition near-duplicate guard. Short generic titles
// cannot be judged via Jaccard; those require human review of the URL lists.
for (let i = 0; i < valid.length; i++) {
    for (let j = 0; j < i; j++) {
        const a = valid[i], b = valid[j];
        if (a.group === b.group || a.terms.size < 4 || b.terms.size < 4) continue;
        let shared = 0;
        for (const token of a.terms) if (b.terms.has(token)) shared++;
        assert.ok(shared / (a.terms.size + b.terms.size - shared) < 0.85,
            'cross-partition near duplicate: ' + a.url + ' and ' + b.url);
    }
}
console.log('Issue training/selection/test URL, label, canonical/near-duplicate checks passed; no scoring in this guard (the test has since been inspected).');
