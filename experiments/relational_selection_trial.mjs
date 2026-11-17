// Development-only mix selection. The Pandas/Electron and authored locked
// partitions are intentionally never imported here. Training rows are still
// the original 3,928; the new Angular/Terraform/FastAPI rows select only.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './relational_partition_guard.mjs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const baseline = load('model.json');
const variant = process.argv.includes('--guards') ? 'guarded' : 'plain';
const half = load(`.cache/relational-${variant}-half.json`);
const anchor = load('experiments/task_baseline_model.json');
assert.equal(baseline.options.structureVersion, 3);
assert.equal(half.options.structureVersion, 4);
const rows = read('relational_development.json').filter(row => row.label !== 'unclear');
const groups = {
    taskValidation: sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 }).validationSet,
    commitValidation: sprout.split_dataset(read('../commit_training.json'), { seed: 1 }).validationSet,
    issue: rows.filter(row => row.kind === 'issue'),
    commit: rows.filter(row => row.kind === 'commit'),
    pairs: read('relational_development_pairs.json')
};
const metric = (model, group) => {
    const r = sprout.evaluate_model(model, group);
    return { correct: r.truePositive + r.trueNegative, n: group.length,
        fp: r.falsePositive, fn: r.falseNegative };
};
const score = model => Object.fromEntries(Object.entries(groups).map(([name, group]) =>
    [name, metric(model, group)]));
const original = score(baseline);
console.log('baseline selection:', original);
const head = read('lexicon_class_weights.json');
function interpolated(mix) {
    const model = { ...half,
        weights: half.weights.map((v, i) => (1 - mix) * (2 * v - anchor.weights[i]) + mix * anchor.weights[i]),
        bias: (1 - mix) * (2 * half.bias - anchor.bias) + mix * anchor.bias,
        options: { ...half.options, trainedFrom: { ...half.options.trainedFrom, anchorMix: mix } } };
    for (const category of Object.keys(head)) {
        const slot = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
        model.weights[slot] = half.weights[slot];
    }
    return model;
}
let selected = null;
for (const mix of [0.2, 0.3, 0.4, 0.5, 0.6]) {
    const result = score(interpolated(mix));
    console.log('mix', mix, result);
    // Declared before this development run. Accept only simultaneous issue
    // and commit gains, class-wise bounds, ordinary-request stability and
    // unchanged task validation; never select on an old or new locked set.
    const pass = result.taskValidation.correct >= original.taskValidation.correct &&
        result.taskValidation.fp <= original.taskValidation.fp &&
        result.commitValidation.correct >= original.commitValidation.correct - 1 &&
        result.issue.correct >= original.issue.correct + 2 &&
        result.commit.correct >= original.commit.correct + 2 &&
        result.issue.fn <= original.issue.fn + 1 &&
        result.commit.fn <= original.commit.fn + 1 &&
        result.issue.fp + result.commit.fp <= original.issue.fp + original.commit.fp - 4 &&
        result.pairs.correct >= original.pairs.correct && result.pairs.fp <= original.pairs.fp;
    if (!pass) continue;
    const gain = result.issue.correct + result.commit.correct -
        original.issue.correct - original.commit.correct;
    if (!selected || gain > selected.gain || (gain === selected.gain && mix > selected.mix)) {
        selected = { mix, gain, result };
    }
}
if (!selected) {
    console.log('REJECT DEVELOPMENT: no mix passes; do not open locked partition');
    process.exitCode = 1;
} else {
    console.log('FROZEN DEVELOPMENT SELECTION', selected);
    fs.writeFileSync(`.cache/relational-${variant}-selection.json`, JSON.stringify(selected, null, 2) + '\n');
    const saved = sprout.save_model(interpolated(selected.mix), `.cache/relational-${variant}-selected.json`, {
        storage: payload => fs.writeFileSync(`.cache/relational-${variant}-selected.json`, payload) });
    assert.ok(saved.success, saved.error);
}
