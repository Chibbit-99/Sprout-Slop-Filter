// Development-only selector for a NEW evidence-gated feature recipe.
// No fresh locked set or historical task test is loaded here.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
import { correctionReference } from './correction_reference.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const version = Number(process.argv[2] || 2);
assert.ok([2, 3].includes(version));
const half = load(`.cache/structure-v${version}-half.json`);
const anchor = load('experiments/task_baseline_model.json');
const baseline = correctionReference();
assert.equal(baseline.options.structureVersion, undefined);
assert.equal(half.options.structureVersion, version);
const selection = read('structure_selection.json').filter(row => row.label !== 'unclear');
const groups = {
    task: sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 }).validationSet,
    commitValidation: sprout.split_dataset(read('../commit_training.json'), { seed: 1 }).validationSet,
    ordinary: read('structure_v2_selection.json'),
    issue: selection.filter(row => row.kind === 'issue'),
    commit: selection.filter(row => row.kind === 'commit')
};
const metrics = model => Object.fromEntries(Object.entries(groups).map(([name, rows]) => {
    const m = sprout.evaluate_model(model, rows);
    return [name, { correct: m.truePositive + m.trueNegative, fp: m.falsePositive, fn: m.falseNegative }];
}));
const baselineResults = metrics(baseline);
console.log('baseline', baselineResults);
const head = read('lexicon_class_weights.json');
function interpolate(mix) {
    const candidate = { ...half, weights: half.weights.map((v, i) =>
        (1 - mix) * (2 * v - anchor.weights[i]) + mix * anchor.weights[i]),
        bias: (1 - mix) * (2 * half.bias - anchor.bias) + mix * anchor.bias };
    for (const category of Object.keys(head)) {
        const id = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
        candidate.weights[id] = half.weights[id];
    }
    candidate.options = { ...half.options, trainedFrom: { ...half.options.trainedFrom, anchorMix: mix } };
    return candidate;
}
let chosen = null;
for (const mix of [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) {
    const scores = metrics(interpolate(mix));
    console.log(mix, scores);
    // This development rule was fixed before scoring any fresh-project rows.
    const pass = scores.task.correct >= baselineResults.task.correct &&
        scores.task.fp <= baselineResults.task.fp &&
        scores.ordinary.correct >= baselineResults.ordinary.correct - 1 &&
        scores.commitValidation.correct >= baselineResults.commitValidation.correct - 1 &&
        scores.issue.correct >= baselineResults.issue.correct + 2 &&
        scores.commit.correct >= baselineResults.commit.correct + 1 &&
        scores.issue.fn <= baselineResults.issue.fn + 1 &&
        scores.commit.fn <= baselineResults.commit.fn + 1;
    if (!pass) continue;
    const gain = (scores.issue.correct - baselineResults.issue.correct) +
        (scores.commit.correct - baselineResults.commit.correct);
    if (!chosen || gain > chosen.gain || (gain === chosen.gain && mix > chosen.mix)) {
        chosen = { mix, gain, scores };
    }
}
if (chosen) {
    const model = interpolate(chosen.mix);
    const saved = sprout.save_model(model, `.cache/structure-v${version}-selected.json`, {
        storage: payload => fs.writeFileSync(`.cache/structure-v${version}-selected.json`, payload)
    });
    assert.ok(saved.success, saved.error);
    fs.writeFileSync(`.cache/structure-v${version}-selection.json`, JSON.stringify(chosen, null, 2) + '\n');
    console.log('FROZEN NEW CANDIDATE', chosen);
} else {
    console.log('NO DEVELOPMENT-COMPATIBLE CANDIDATE; DO NOT OPEN THE FRESH TEST');
    process.exitCode = 1;
}
