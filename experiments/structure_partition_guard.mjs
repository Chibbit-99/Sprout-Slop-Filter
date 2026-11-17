// Data-only guard for a prospective vocabulary-independent feature trial.
// It validates sources and leakage without loading/scoring any checkpoint.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const canonical = text => sprout.tokenise(sprout.normalise_text(text))
    .filter(token => /[\p{L}\p{N}]/u.test(token)).join(' ');
const priorFiles = ['../training.json', '../commit_training.json', 'task_baseline_training.json',
    'task_supplement.json', 'commit_supplement.json', 'real_issue_training.json',
    'real_issue_validation.json', 'real_issue_holdout.json', 'blind_issue_audit.json',
    'issue_corrections_training.json', 'fresh_issue_selection.json', 'fresh_issue_test.json',
    'unfamiliar_cases.json', 'other_contexts.json', 'domain_probes.json', 'transfer_cases.json'];
const prior = new Set();
function visit(node) {
    if (!node || typeof node !== 'object') return;
    if (typeof node.text === 'string') prior.add(canonical(node.text));
    else for (const value of Object.values(node)) visit(value);
}
for (const file of priorFiles) visit(read(file));
const configs = [
    ['structure_selection.json', {
        'vitest-dev/vitest/issues': [40, 36, 4, 0],
        'vitest-dev/vitest/commit': [50, 43, 3, 4]
    }],
    ['structure_locked.json', {
        'home-assistant/core/issues': [24, 19, 5, 0],
        'flutter/flutter/issues': [32, 27, 2, 3],
        'microsoft/playwright/issues': [36, 32, 4, 0],
        'denoland/deno/commit': [60, 56, 1, 3]
    }],
    ['structure_confirmation.json', {
        'rust-lang/rust/issues': [30, 28, 0, 2],
        'python/cpython/issues': [29, 27, 1, 1],
        'pallets/flask/commit': [50, 26, 3, 21]
    }]
];
const seen = new Set(), urls = new Set(), labelled = [];
for (const [file, groups] of configs) {
    const rows = read(file);
    assert.equal(rows.length, Object.values(groups).reduce((sum, counts) => sum + counts[0], 0));
    for (const [source, expected] of Object.entries(groups)) {
        const selected = rows.filter(row => row.url.startsWith('https://github.com/' + source + '/'));
        assert.equal(selected.length, expected[0], source);
        assert.deepEqual(['genuine', 'slop', 'unclear'].map(label => selected.filter(row => row.label === label).length),
            expected.slice(1), source);
        for (const row of selected) {
            assert.ok(!urls.has(row.url), 'duplicate URL: ' + row.url);
            urls.add(row.url);
            assert.equal(row.kind, source.endsWith('/issues') ? 'issue' : 'commit', row.url);
            assert.match(row.url, source.endsWith('/commit') ? /\/commit\/[a-f\d]{40}$/ : /\/issues\/\d+$/);
            if (row.label === 'unclear') { assert.ok(row.reason); continue; }
            const key = canonical(row.text);
            assert.ok(key && !prior.has(key) && !seen.has(key), 'duplicate: ' + row.url);
            seen.add(key);
            labelled.push({ ...row, source: file, terms: new Set(key.split(' ')) });
        }
    }
}
for (let i = 0; i < labelled.length; i++) for (let j = 0; j < i; j++) {
    const a = labelled[i], b = labelled[j];
    if (a.source === b.source || a.terms.size < 4 || b.terms.size < 4) continue;
    let common = 0;
    for (const token of a.terms) if (b.terms.has(token)) common++;
    assert.ok(common / (a.terms.size + b.terms.size - common) < 0.85,
        'near duplicate: ' + a.url + ' / ' + b.url);
}
console.log('Source, label count, canonical and cross-partition near-duplicate guards passed. No scoring occurred.');
