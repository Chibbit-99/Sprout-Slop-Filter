// Select only using the previously inspected task/commit VALIDATION and the
// newly labelled Vitest issue/commit selection rows. Do not open the new
// Flutter/Home Assistant/Playwright/Deno locked set here.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const candidate = load('.cache/structure-candidate-050.json');
const anchor = load('experiments/task_baseline_model.json');
const baseline = load('experiments/task_corrections_unanchored_model.json');
sprout.blend_models(baseline, anchor, 0.5);
assert.equal(candidate.options.structureVersion, 1);
const head = read('lexicon_class_weights.json');
for (const [category, weight] of Object.entries(head)) {
    baseline.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
baseline.options.lexicon = candidate.options.lexicon;
baseline.options.lexiconStrength = 1;
const slots = Object.keys(head).map(category => 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536));
const selected = read('structure_selection.json').filter(row => row.label !== 'unclear');
const groups = {
    taskVal: sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 }).validationSet,
    commitVal: sprout.split_dataset(read('../commit_training.json'), { seed: 1 }).validationSet,
    issue: selected.filter(row => row.kind === 'issue'),
    commit: selected.filter(row => row.kind === 'commit')
};
const metric = (model, rows) => {
    const m = sprout.evaluate_model(model, rows);
    return { right: m.truePositive + m.trueNegative, fp: m.falsePositive, fn: m.falseNegative };
};
const evaluate = model => Object.fromEntries(Object.entries(groups).map(([key, rows]) => [key, metric(model, rows)]));
const reference = evaluate(baseline);
console.log('baseline selection:', reference);
let best = null;
for (const mix of [0.3, 0.4, 0.5, 0.6, 0.7]) {
    const model = { ...candidate,
        weights: candidate.weights.map((value, i) => (1 - mix) * (2 * value - anchor.weights[i]) + mix * anchor.weights[i]),
        bias: (1 - mix) * (2 * candidate.bias - anchor.bias) + mix * anchor.bias };
    for (const slot of slots) model.weights[slot] = candidate.weights[slot];
    const result = evaluate(model);
    console.log('mix=' + mix, result);
    // Gate fixed before this run. The candidate must improve *both* new
    // selection contexts, lower their combined FP by >=3 without more than
    // one new FN each, and preserve old task validation and near-preserve
    // commit validation. Otherwise no new test will be opened.
    const acceptable = result.taskVal.right >= reference.taskVal.right && result.taskVal.fp <= reference.taskVal.fp &&
        result.commitVal.right >= reference.commitVal.right - 1 &&
        result.issue.right >= reference.issue.right + 1 &&
        result.commit.right >= reference.commit.right + 1 &&
        result.issue.fp + result.commit.fp <= reference.issue.fp + reference.commit.fp - 3 &&
        result.issue.fn <= reference.issue.fn + 1 && result.commit.fn <= reference.commit.fn + 1;
    if (!acceptable) continue;
    const improvement = result.issue.right + result.commit.right - reference.issue.right - reference.commit.right;
    if (!best || improvement > best.improvement || (improvement === best.improvement && mix > best.mix)) {
        best = { mix, improvement, result };
    }
}
if (best) {
    console.log('SELECTED without opening new test', best);
    fs.writeFileSync('.cache/structure-selected.json', JSON.stringify(best, null, 2) + '\n');
} else {
    console.log('REJECTED: no validation-compatible structural candidate; keep model.json unchanged.');
}
