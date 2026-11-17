// Select a candidate using ONLY development/validation. It never opens the
// newly labelled React/TypeScript/VS Code test data. The pre-released
// Fastify/ESLint audit is now correction TRAINING, not a held-out score.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const raw = fs.readFileSync(file, 'utf8');
    const result = sprout.load_model(file, { storage: () => raw });
    assert.ok(result.success, result.error);
    return result.model;
};
const candidate = load('.cache/candidate-050.json');
const anchor = load('experiments/task_baseline_model.json');
const shipped = load('experiments/task_real_unanchored_model.json');
sprout.blend_models(shipped, anchor, 0.5);
const head = read('lexicon_class_weights.json');
for (const [category, weight] of Object.entries(head)) {
    shipped.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
shipped.options.lexicon = candidate.options.lexicon;
shipped.options.lexiconStrength = 1;
const classSlots = Object.keys(head).map(category =>
    7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536));
const frozen = sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 }).validationSet;
const axios = read('real_issue_validation.json');
const babel = read('fresh_issue_selection.json').filter(row => row.label !== 'unclear');
const groups = { frozen, axios, babel };
const metrics = (model, rows) => {
    const result = sprout.evaluate_model(model, rows);
    return { right: result.truePositive + result.trueNegative,
        fp: result.falsePositive, fn: result.falseNegative };
};
const reference = Object.fromEntries(Object.entries(groups).map(([key, rows]) => [key, metrics(shipped, rows)]));
console.log('shipped validation:', reference);
let selected = null;
for (const mix of [0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8]) {
    // Recover the unanchored correction-trained checkpoint from the 50%
    // training-time blend, then interpolate with the SAME old task anchor.
    // The 45 WordNet slots were grafted after the blend, so restore them.
    const model = { ...candidate, weights: candidate.weights.map((weight, i) =>
        (1 - mix) * (2 * weight - anchor.weights[i]) + mix * anchor.weights[i]),
        bias: (1 - mix) * (2 * candidate.bias - anchor.bias) + mix * anchor.bias };
    for (const slot of classSlots) model.weights[slot] = candidate.weights[slot];
    const result = Object.fromEntries(Object.entries(groups).map(([key, rows]) => [key, metrics(model, rows)]));
    console.log(`mix=${mix}:`, result);
    // Do not trade the previously shipped task validation or Axios quality
    // for a source-specific improvement. Babel has few vague titles; allow
    // at most one new mistake, then rank balanced labels on both real sets.
    const eligible = result.frozen.right >= reference.frozen.right && result.frozen.fp <= reference.frozen.fp &&
        result.axios.right >= reference.axios.right && result.axios.fp <= reference.axios.fp &&
        result.babel.right >= reference.babel.right - 1 && result.babel.fp <= reference.babel.fp + 1;
    if (!eligible) continue;
    const score = result.axios.right / axios.length + result.babel.right / babel.length;
    const fp = result.axios.fp + result.babel.fp;
    if (!selected || score > selected.score || (score === selected.score && fp < selected.fp) ||
        (score === selected.score && fp === selected.fp && mix > selected.mix)) {
        selected = { mix, score, fp, result };
    }
}
if (!selected) {
    console.log('No validation-compatible correction candidate; keep shipped checkpoint.');
    process.exitCode = 1;
} else {
    console.log('Selected WITHOUT opening new locked test:', selected);
    fs.writeFileSync('.cache/issue-selected.json', JSON.stringify(selected, null, 2) + '\n');
}
