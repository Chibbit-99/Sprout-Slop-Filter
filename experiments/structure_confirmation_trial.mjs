// Second, separately sourced confirmation for the frozen 0.40 candidate.
// Collected after the first locked trial failed its predeclared task floor.
// This diagnostic cannot override that failed release gate or ship a model.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './structure_partition_guard.mjs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const half = load('.cache/structure-candidate-050.json');
const anchor = load('experiments/task_baseline_model.json');
const baseline = load('experiments/task_corrections_unanchored_model.json');
sprout.blend_models(baseline, anchor, 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    baseline.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
baseline.options.lexicon = half.options.lexicon;
baseline.options.lexiconStrength = 1;
const chosen = JSON.parse(fs.readFileSync('.cache/structure-selected.json', 'utf8'));
assert.equal(chosen.mix, 0.4);
const mix = chosen.mix;
const candidate = { ...half,
    weights: half.weights.map((w, i) => (1 - mix) * (2 * w - anchor.weights[i]) + mix * anchor.weights[i]),
    bias: (1 - mix) * (2 * half.bias - anchor.bias) + mix * anchor.bias };
for (const category of Object.keys(read('lexicon_class_weights.json'))) {
    const id = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
    candidate.weights[id] = half.weights[id];
}
const rows = read('structure_confirmation.json').filter(row => row.label !== 'unclear');
const metric = (model, group) => {
    const m = sprout.evaluate_model(model, group);
    return { correct: m.truePositive + m.trueNegative, n: group.length,
        fp: m.falsePositive, fn: m.falseNegative };
};
const compare = group => ({ before: metric(baseline, group), after: metric(candidate, group) });
for (const repo of ['rust-lang/rust', 'python/cpython', 'pallets/flask']) {
    const group = rows.filter(row => row.url.includes('/' + repo + '/'));
    console.log(repo, compare(group));
    console.log('  genuine', compare(group.filter(row => row.label === 'genuine')));
    console.log('  vague', compare(group.filter(row => row.label === 'slop')));
}
const issues = compare(rows.filter(row => row.kind === 'issue'));
const commits = compare(rows.filter(row => row.kind === 'commit'));
const total = compare(rows);
console.log('SECOND CONFIRMATION ISSUES', issues);
console.log('SECOND CONFIRMATION COMMITS', commits);
console.log('SECOND CONFIRMATION TOTAL', total);
// The second partition was labelled before its scoring but after the first
// gate failed; its encouraging results are exploratory confirmation, not an
// alternative release criterion. The 4 vague titles are inadequate for
// assessing missed filler. No checkpoint is written by this diagnostic.
assert.equal(sprout.model_parameter_stats(candidate).trainable, 524289);
console.log('FIRST PREDECLARED RELEASE GATE: FAILED (168/170 task floor; candidate 167/170)');
console.log('SECOND PARTITION: descriptive confirmation only; no release');
