// Formerly locked, one-time post-selection test (opened AFTER fixing the
// 0.50 anchor on task/Axios/Babel validation). It is NOW INSPECTED: future
// tuning on these rows would invalidate a claim of independent improvement.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './issue_partition_guard.mjs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const raw = fs.readFileSync(new URL(file, import.meta.url), 'utf8');
    const result = sprout.load_model(file, { storage: () => raw });
    assert.ok(result.success, result.error);
    return result.model;
};
const current = load('../model.json');
const before = load('task_real_unanchored_model.json');
sprout.blend_models(before, load('task_baseline_model.json'), 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    before.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
before.options.lexicon = current.options.lexicon;
before.options.lexiconStrength = 1;
const corrected = load('task_corrections_unanchored_model.json');
sprout.blend_models(corrected, load('task_baseline_model.json'), 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    corrected.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
corrected.options.lexicon = current.options.lexicon;
corrected.options.lexiconStrength = 1;
const models = { before, corrected, current };
const rows = read('fresh_issue_test.json').filter(row => row.label !== 'unclear');
const evaluate = (model, group) => {
    const m = sprout.evaluate_model(model, group);
    return { right: m.truePositive + m.trueNegative, n: group.length,
        fp: m.falsePositive, fn: m.falseNegative };
};
const compare = group => Object.fromEntries(Object.entries(models)
    .map(([name, model]) => [name, evaluate(model, group)]));
for (const repo of ['react/react', 'microsoft/TypeScript', 'microsoft/vscode']) {
    const group = rows.filter(row => row.url.includes('/' + repo + '/'));
    console.log(repo, compare(group));
    for (const label of ['genuine', 'slop']) {
        console.log('  ' + label, compare(group.filter(row => row.label === label)));
    }
}
const total = compare(rows);
console.log('ONCE-LOCKED TEST (NOW INSPECTED)', total);
assert.deepEqual(total.before, { right: 75, n: 135, fp: 58, fn: 2 });
assert.deepEqual(total.corrected, { right: 82, n: 135, fp: 50, fn: 3 });
console.log('Fastify/ESLint CORRECTION TRAINING rows', compare(read('issue_corrections_training.json')));
const frozen = sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 });
const expanded = sprout.split_dataset(read('../training.json'), { seed: 1 });
console.log('historical original task TEST', compare(frozen.testSet));
console.log('historical expanded task TEST', compare(expanded.testSet));
console.log('historical Axios selection', compare(read('real_issue_validation.json')));
console.log('historical SvelteKit/Express', compare(read('real_issue_holdout.json')));
console.log('historical unfamiliar contexts', compare(Object.values(read('unfamiliar_cases.json').groups).flat()));
console.log('These title-only, single-rater, page-selected rows are no longer an independent test.');
