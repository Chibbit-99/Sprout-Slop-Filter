// Reproducible *inspected* diagnostics. Reconstruct the shipped corrected
// v4 and the rejected opt-in structural candidate at the same 524,289-scalar
// budget. Single-rater sampled titles are NOT a deployment estimate; never
// use these inspected sets to tune another candidate.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './structure_partition_guard.mjs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(new URL(file, import.meta.url), 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const current = load('../model.json');
const anchor = load('task_baseline_model.json');
const old = load('task_corrections_unanchored_model.json');
sprout.blend_models(old, anchor, 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    old.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
old.options.lexicon = current.options.lexicon;
old.options.lexiconStrength = 1;
const half = load('../.cache/structure-candidate-050.json');
const chosen = read('../.cache/structure-selected.json');
assert.equal(chosen.mix, 0.4);
const mix = chosen.mix;
const candidate = { ...half,
    weights: half.weights.map((weight, i) => (1 - mix) * (2 * weight - anchor.weights[i]) + mix * anchor.weights[i]),
    bias: (1 - mix) * (2 * half.bias - anchor.bias) + mix * anchor.bias };
for (const category of Object.keys(read('lexicon_class_weights.json'))) {
    const id = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
    candidate.weights[id] = half.weights[id];
}
assert.equal(current.options.structureVersion, 3);
assert.equal(old.options.structureVersion, undefined);
assert.equal(candidate.options.structureVersion, 1);
assert.equal(sprout.model_parameter_stats(candidate).trainable, 524289);
const metric = (model, rows) => {
    const result = sprout.evaluate_model(model, rows);
    return { correct: result.truePositive + result.trueNegative, n: rows.length,
        fp: result.falsePositive, fn: result.falseNegative };
};
const compare = rows => ({ before: metric(old, rows), after: metric(candidate, rows) });
const selected = read('structure_selection.json').filter(row => row.label !== 'unclear');
const locked = read('structure_locked.json').filter(row => row.label !== 'unclear');
const confirmation = read('structure_confirmation.json').filter(row => row.label !== 'unclear');
for (const [group, rows] of [['selection issues', selected.filter(row => row.kind === 'issue')],
    ['selection commits', selected.filter(row => row.kind === 'commit')],
    ['first held-out issues', locked.filter(row => row.kind === 'issue')],
    ['first held-out commits', locked.filter(row => row.kind === 'commit')],
    ['second confirmation issues', confirmation.filter(row => row.kind === 'issue')],
    ['second confirmation commits', confirmation.filter(row => row.kind === 'commit')]]) {
    console.log(group, compare(rows));
}
assert.deepEqual(metric(old, locked), { correct: 85, n: 146, fp: 56, fn: 5 });
assert.deepEqual(metric(candidate, locked), { correct: 112, n: 146, fp: 29, fn: 5 });
assert.deepEqual(metric(old, confirmation), { correct: 54, n: 85, fp: 30, fn: 1 });
assert.deepEqual(metric(candidate, confirmation), { correct: 67, n: 85, fp: 17, fn: 1 });
assert.deepEqual(metric(current, locked), metric(candidate, locked));
const known = new Set(current.vocabulary);
const novelFraction = row => {
    const words = sprout.tokenise(sprout.normalise_text(row.text));
    return words.filter(word => !known.has(word)).length / words.length;
};
for (const [label, rows] of [['<25% training-OOV', locked.concat(confirmation).filter(row => novelFraction(row) < 0.25)],
    ['>=25% training-OOV', locked.concat(confirmation).filter(row => novelFraction(row) >= 0.25)]]) {
    console.log(label, compare(rows));
}
for (const [name, rows] of [['old task test', sprout.split_dataset(read('../training.json'), { seed: 1 }).testSet],
    ['unfamiliar authored contexts', Object.values(read('unfamiliar_cases.json').groups).flat()],
    ['helpdesk hard zero-shot', read('other_contexts.json').helpdesk.stress]]) {
    console.log('historical ' + name, compare(rows));
}
console.log('All sources were selected by API pages and single-rater title/subject labels; commit informativeness does not verify the code diff or author.');
