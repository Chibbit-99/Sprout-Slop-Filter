// Historical audit, now correction TRAINING data. Its title-only labels
// were fixed before the old model was scored, but it is NOT a held-out test
// of the new model. No production accuracy claim follows from these rows.
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
const current = load('../model.json');
const beforeCorrections = load('task_real_unanchored_model.json');
sprout.blend_models(beforeCorrections, load('task_baseline_model.json'), 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    beforeCorrections.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
beforeCorrections.options.lexicon = current.options.lexicon;
beforeCorrections.options.lexiconStrength = 1;
const prior = load('task_unanchored_model.json');
sprout.blend_models(prior, load('task_baseline_model.json'), 0.45);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    prior.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
prior.options.lexicon = current.options.lexicon;
prior.options.lexiconStrength = 1;
const rows = read('blind_issue_audit.json');
assert.equal(rows.length, 120);
const normalize = text => sprout.tokenise(sprout.normalise_text(text))
    .filter(token => /[\p{L}\p{N}]/u.test(token)).join(' ');
// Guard against leaks from *all* previously checked-in labelled training
// and diagnostics, not merely the three real-issue splits. Nested probe
// files use a different schema, so walk their labelled rows recursively.
const protectedFiles = [
    '../training.json', '../commit_training.json', 'task_baseline_training.json',
    'task_supplement.json', 'commit_supplement.json', 'real_issue_training.json',
    'real_issue_validation.json', 'real_issue_holdout.json', 'unfamiliar_cases.json',
    'other_contexts.json', 'domain_probes.json', 'transfer_cases.json'
];
const forbidden = new Set();
function protect(node) {
    if (node === null || typeof node !== 'object') return;
    if (typeof node.text === 'string') forbidden.add(normalize(node.text));
    else for (const value of Object.values(node)) protect(value);
}
for (const file of protectedFiles) protect(read(file));
const seen = new Set();
for (const row of rows) {
    assert.match(row.url, /^https:\/\/github\.com\/(fastify\/fastify|eslint\/eslint)\/issues\/\d+$/);
    assert.ok(['genuine', 'slop', 'unclear'].includes(row.label));
    const key = normalize(row.text);
    assert.ok(key && !forbidden.has(key) && !seen.has(key), 'overlap: ' + row.url);
    seen.add(key);
}
for (const [repo, expected] of [
    ['fastify/fastify', { genuine: 44, slop: 12, unclear: 4 }],
    ['eslint/eslint', { genuine: 46, slop: 9, unclear: 5 }]
]) {
    const group = rows.filter(row => row.url.includes('/' + repo + '/'));
    assert.equal(group.length, 60, repo + ' issue count');
    const actual = group.reduce((counts, row) => {
        counts[row.label]++;
        return counts;
    }, { genuine: 0, slop: 0, unclear: 0 });
    assert.deepEqual(actual, expected, repo + ' label counts');
}
const results = (model, group) => {
    let correct = 0, fp = 0, fn = 0, reviewed = 0, accepted = 0, acceptedCorrect = 0;
    let oov = 0, total = 0;
    const known = new Set(model.vocabulary);
    for (const row of group) {
        const score = sprout.make_prediction_for_training(row.text, model).score;
        const right = (score >= 0.5) === (row.label === 'slop');
        correct += +right;
        fp += +(score >= 0.5 && row.label === 'genuine');
        fn += +(score < 0.5 && row.label === 'slop');
        const tokens = sprout.tokenise(sprout.normalise_text(row.text));
        const novel = tokens.filter(word => !known.has(word)).length;
        oov += novel;
        total += tokens.length;
        // Fixed risk policy defined before scoring this audit: review near the
        // 0.5 decision boundary OR when at least half the tokens are OOV.
        const review = (score >= 0.4 && score <= 0.6) || (tokens.length > 0 && novel / tokens.length >= 0.5);
        reviewed += +review;
        accepted += +!review;
        acceptedCorrect += +(!review && right);
    }
    return { correct, totalRows: group.length, fp, fn, oov, total,
        reviewed, accepted, acceptedCorrect };
};
for (const repo of ['fastify/fastify', 'eslint/eslint']) {
    const set = rows.filter(row => row.url.includes('/' + repo + '/') && row.label !== 'unclear');
    for (const label of ['genuine', 'slop']) {
        const group = set.filter(row => row.label === label);
        const historical = results(prior, group), frozen = results(beforeCorrections, group), fitted = results(current, group);
        console.log(`${repo} ${label} (${group.length} titles): historical ${historical.correct}/${group.length}; ` +
            `pre-correction ${frozen.correct}/${group.length} FP${frozen.fp} FN${frozen.fn}; ` +
            `now TRAINING fit ${fitted.correct}/${group.length} FP${fitted.fp} FN${fitted.fn}; ` +
            `${frozen.oov}/${frozen.total} original-training-OOV tokens`);
    }
    const frozen = results(beforeCorrections, set), fitted = results(current, set);
    console.log(`  pre-correction ${frozen.correct}/${set.length}, FP${frozen.fp} FN${frozen.fn}; ` +
        `after training ${fitted.correct}/${set.length}, FP${fitted.fp} FN${fitted.fn}; ` +
        `PRE-correction fixed review rule: ${frozen.reviewed} reviewed, ${frozen.acceptedCorrect}/${frozen.accepted} accepted correct`);
}
// Historical post-hoc slices explain pre-correction failure modes only.
const evaluated = rows.filter(row => row.label !== 'unclear');
const vocabulary = new Set(beforeCorrections.vocabulary);
for (const label of ['genuine', 'slop']) {
    const items = evaluated.filter(row => row.label === label).map(row => {
        const tokens = sprout.tokenise(sprout.normalise_text(row.text));
        const unknown = tokens.filter(token => !vocabulary.has(token)).length;
        const score = sprout.make_prediction_for_training(row.text, beforeCorrections).score;
        return { length: tokens.length, oov: unknown / tokens.length,
            wrong: (score >= 0.5) !== (label === 'slop') };
    });
    for (const [name, test] of [
        ['short (<6 tokens)', row => row.length < 6],
        ['long (>=6 tokens)', row => row.length >= 6],
        ['low OOV (<25%)', row => row.oov < 0.25],
        ['higher OOV (>=25%)', row => row.oov >= 0.25]
    ]) {
        const slice = items.filter(test);
        console.log(`  ${label} ${name}: ${slice.filter(row => row.wrong).length}/${slice.length} wrong`);
    }
}
console.log('These single-rater audit titles are NOW IN TRAINING; only the pre-correction scores are a past diagnostic, never a new test.');
