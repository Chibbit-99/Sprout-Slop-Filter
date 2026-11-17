// ONE-TIME release gate for the development-selected v3 recipe. The fresh
// URLs/titles/labels were frozen and partition-checked before fitting v3.
// Rules below are fixed before loading/scoring fresh rows. Both the original
// failed gate and a new, source-separated diagnostic must pass; no post-hoc
// anchor, threshold, cue-list or label changes on this opened set.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './structure_fresh_guard.mjs';
import { sprout } from './lib/sprout.mjs';
import { correctionReference } from './correction_reference.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const baseline = correctionReference();
const selected = read('structure_v3_selection.json');
assert.equal(selected.mix, 0.4);
// On a clean checkout model.json is the reproduced release candidate; local
// .cache is optional and must agree if present.
const candidate = load('model.json');
if (fs.existsSync('.cache/structure-v3-selected.json')) {
    const frozen = load('.cache/structure-v3-selected.json');
    assert.ok(candidate.weights.every((w, i) => Math.abs(w - frozen.weights[i]) < 1e-12));
    assert.ok(Math.abs(candidate.bias - frozen.bias) < 1e-12);
}
assert.equal(candidate.options.structureVersion, 3);
assert.equal(candidate.options.trainedFrom.anchorMix, selected.mix);
assert.equal(sprout.model_parameter_stats(candidate).trainable, 524289);
const score = (model, rows) => {
    const r = sprout.evaluate_model(model, rows);
    return { correct: r.truePositive + r.trueNegative, n: rows.length,
        fp: r.falsePositive, fn: r.falseNegative };
};
const compare = rows => ({ old: score(baseline, rows), candidate: score(candidate, rows) });
const historicalTask = sprout.split_dataset(read('../training.json'), { seed: 1 }).testSet;
const unfamiliar = Object.values(read('unfamiliar_cases.json').groups).flat();
const helpdesk = read('other_contexts.json').helpdesk.stress;
const first = read('structure_locked.json').filter(row => row.label !== 'unclear');
const priorIssue = compare(first.filter(row => row.kind === 'issue'));
const priorCommit = compare(first.filter(row => row.kind === 'commit'));
const priorRepos = ['home-assistant/core', 'flutter/flutter', 'microsoft/playwright'];
const priorGate = priorIssue.candidate.fp <= priorIssue.old.fp - 8 &&
    priorIssue.candidate.fn <= priorIssue.old.fn + 2 &&
    priorCommit.candidate.fp <= priorCommit.old.fp - 3 &&
    priorCommit.candidate.correct >= priorCommit.old.correct + 3 &&
    priorCommit.candidate.fn <= priorCommit.old.fn + 2 &&
    priorRepos.every(repo => {
        const x = compare(first.filter(row => row.url.includes('/' + repo + '/')));
        return x.candidate.correct >= x.old.correct - 2;
    }) && score(candidate, historicalTask).correct >= 168 &&
    score(candidate, unfamiliar).correct >= 35 && score(candidate, helpdesk).correct >= 12;
// The fresh-set gate was declared before its first prediction. The first
// held-out test was inspected previously, so the second set is an additional
// prospective diagnostic, not an independent estimate of deployment accuracy.
const fresh = read('structure_fresh_locked.json').filter(row => row.label !== 'unclear');
const freshIssues = compare(fresh.filter(row => row.kind === 'issue'));
const freshCommits = compare(fresh.filter(row => row.kind === 'commit'));
const freshAll = compare(fresh);
const repos = ['golang/go', 'rails/rails', 'kubernetes/kubernetes', 'oven-sh/bun'];
const project = Object.fromEntries(repos.map(repo => [repo,
    compare(fresh.filter(row => row.url.includes('/' + repo + '/')))]));
const freshGate = freshIssues.candidate.correct >= freshIssues.old.correct + 3 &&
    freshCommits.candidate.correct >= freshCommits.old.correct + 3 &&
    freshAll.candidate.fp <= freshAll.old.fp - 12 &&
    freshAll.candidate.fn <= freshAll.old.fn + 2 &&
    repos.every(repo => project[repo].candidate.correct >= project[repo].old.correct - 2);
console.log('HISTORICAL TASK', compare(historicalTask));
console.log('INSPECTED UNFAMILIAR', compare(unfamiliar));
console.log('INSPECTED HELPDESK', compare(helpdesk));
console.log('FIRST LOCKED ISSUE', priorIssue, 'COMMIT', priorCommit);
console.log('FRESH ISSUE', freshIssues, 'COMMIT', freshCommits, 'TOTAL', freshAll);
console.log('FRESH PROJECTS', project);
console.log('ORIGINAL RELEASE GATE', priorGate ? 'PASS' : 'FAIL');
console.log('NEW PREDECLARED GATE', freshGate ? 'PASS' : 'FAIL');
const pass = priorGate && freshGate;
console.log('OVERALL', pass ? 'PASS — integration may proceed' : 'FAIL — keep model.json unchanged');
if (!pass) process.exitCode = 1;
