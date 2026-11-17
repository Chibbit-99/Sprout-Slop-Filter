// Curated GitHub issue-title diagnostics: real text, manually selected and
// labelled by title ONLY. Cross-repository separation is not a random sample
// or a production accuracy estimate. Never train on validation/holdout rows.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () =>
        fs.readFileSync(new URL(file, import.meta.url), 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const old = load('task_unanchored_model.json');
sprout.blend_models(old, load('task_baseline_model.json'), 0.45);
const shipped = load('../model.json');
const head = read('lexicon_class_weights.json');
for (const [category, weight] of Object.entries(head)) {
    old.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
old.options.lexicon = shipped.options.lexicon;
old.options.lexiconStrength = 1;
const realTraining = read('real_issue_training.json');
const realValidation = read('real_issue_validation.json');
const realHoldout = read('real_issue_holdout.json');
const sources = [
    [realTraining, ['vitejs/vite', 'nodejs/node']],
    [realValidation, ['axios/axios']],
    [realHoldout, ['sveltejs/kit', 'expressjs/express']]
];
const canonical = text => sprout.tokenise(sprout.normalise_text(text))
    .filter(word => /[\p{L}\p{N}]/u.test(word)).join(' ');
const seen = new Set();
const base = read('../training.json');
const supplement = read('task_supplement.json');
for (const row of base.concat(supplement)) seen.add(canonical(row.text));
for (const [rows, repos] of sources) {
    for (const row of rows) {
        assert.ok(repos.some(repo => row.url.startsWith('https://github.com/' + repo + '/issues/')),
            'unexpected source ' + row.url);
        assert.ok(['genuine', 'slop'].includes(row.label));
        const key = canonical(row.text);
        assert.ok(key && !seen.has(key), 'cross-partition duplicate ' + row.url);
        seen.add(key);
    }
}
assert.equal(realTraining.length, 131);
assert.equal(realValidation.length, 46);
assert.equal(realHoldout.length, 30);
const full = sprout.split_dataset(base, { seed: 1 });
const frozen = sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 });
const groups = { originalValidation: frozen.validationSet, expandedValidation: full.validationSet,
    axiosValidation: realValidation, svelteExpressHoldout: realHoldout,
    originalTest: frozen.testSet, expandedTest: full.testSet,
    unfamiliar: Object.values(read('unfamiliar_cases.json').groups).flat(),
    helpdeskHard: read('other_contexts.json').helpdesk.stress,
    agendaHard: read('other_contexts.json').meeting_agendas.stress };
const metrics = (model, rows) => {
    const result = sprout.evaluate_model(model, rows);
    return { correct: result.truePositive + result.trueNegative, count: result.count,
        fp: result.falsePositive, fn: result.falseNegative };
};
console.log('Curated source-separated titles (training Vite/Node, validation Axios, held-out Svelte/Express):');
for (const [name, rows] of Object.entries(groups)) {
    const before = metrics(old, rows), after = metrics(shipped, rows);
    console.log(`${name.padEnd(23)} ${before.correct}/${before.count} FP${before.fp} FN${before.fn} → ` +
        `${after.correct}/${after.count} FP${after.fp} FN${after.fn}`);
    if (name === 'svelteExpressHoldout') {
        assert.deepEqual([before.correct, after.correct], [12, 23]);
    }
}
console.log('Titles were selected/labelled manually, not randomly sampled; results are not deployment estimates.');
