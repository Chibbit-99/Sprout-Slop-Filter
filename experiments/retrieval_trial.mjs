// Experimental offline issue-title retrieval head. NOT shipped: this adds a
// second inference component and has not demonstrated reliable generalisation.
// Choose only on previously inspected ORIGINAL TASK and AXIOS validation;
// the inspected Fastify/ESLint audit is diagnostic, NOT an independent test.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';

const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
// Recreate the frozen pre-correction model exactly, avoiding an extra 2.5 MB
// checkpoint in the repo. This script remains a *historical* failed trial.
const model = load('experiments/task_real_unanchored_model.json');
sprout.blend_models(model, load('experiments/task_baseline_model.json'), 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    model.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
model.options.lexicon = load('model.json').options.lexicon;
model.options.lexiconStrength = 1;
const train = read('real_issue_training.json'); // Vite and Node titles only
const axios = read('real_issue_validation.json');
const original = sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 }).validationSet;

function terms(text) {
    const words = text.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
    const out = new Map();
    const add = (term, value) => out.set(term, (out.get(term) || 0) + value);
    for (let i = 0; i < words.length; i++) {
        const word = words[i].replace(/\d/g, '0');
        add('w:' + word, 1);
        if (i) add('p:' + words[i - 1].replace(/\d/g, '0') + ' ' + word, 0.7);
        const padded = '^' + word + '$';
        for (let j = 0; j + 3 <= padded.length; j++) add('c:' + padded.slice(j, j + 3), 0.12);
    }
    return out;
}
const termsByTrainingRow = train.map(row => terms(row.text));
const documentFrequency = new Map();
for (const row of termsByTrainingRow) {
    for (const term of row.keys()) documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
}
const idf = term => 1 + Math.log((1 + train.length) / (1 + (documentFrequency.get(term) || 0)));
function vector(text) {
    const result = terms(text);
    let squaredLength = 0;
    for (const [term, weight] of result) {
        const weighted = weight * idf(term);
        result.set(term, weighted);
        squaredLength += weighted * weighted;
    }
    const length = Math.sqrt(squaredLength) || 1;
    for (const [term, weight] of result) result.set(term, weight / length);
    return result;
}
const trainingVectors = train.map(row => vector(row.text));
const cache = new Map();
function ranked(text) {
    if (!cache.has(text)) {
        const query = vector(text);
        const hits = trainingVectors.map((candidate, i) => {
            let similarity = 0;
            for (const [term, weight] of query) {
                if (candidate.has(term)) similarity += weight * candidate.get(term);
            }
            return { similarity, i };
        });
        cache.set(text, hits.sort((a, b) => b.similarity - a.similarity));
    }
    return cache.get(text);
}
function score(text, config) {
    const base = sprout.make_prediction_for_training(text, model).score;
    const hits = ranked(text).filter(hit => hit.similarity >= config.min).slice(0, config.k);
    if (!hits.length) return base;
    const sum = hits.reduce((value, hit) => value + hit.similarity, 0);
    const signedEvidence = hits.reduce((value, hit) =>
        value + hit.similarity * (train[hit.i].label === 'slop' ? 1 : -1), 0) / sum;
    const logit = Math.log(base / (1 - base)) + hits[0].similarity * config.weight * signedEvidence;
    return 1 / (1 + Math.exp(-logit));
}
function metrics(rows, config) {
    let fp = 0, fn = 0, loss = 0;
    for (const row of rows) {
        const p = score(row.text, config), y = row.label === 'slop';
        fp += +(p >= 0.5 && !y);
        fn += +(p < 0.5 && y);
        loss += -(+y) * Math.log(Math.max(p, 1e-9)) - (+!y) * Math.log(Math.max(1 - p, 1e-9));
    }
    return { correct: rows.length - fp - fn, fp, fn, loss: +(loss / rows.length).toFixed(4) };
}
const baseline = { k: 1, min: 1, weight: 0 };
console.log('baseline original, Axios:', metrics(original, baseline), metrics(axios, baseline));
const eligible = [];
for (const k of [1, 3, 5]) {
    for (const min of [0.1, 0.2, 0.3, 0.4]) {
        for (const weight of [0.25, 0.5, 1, 1.5, 2, 3]) {
            const config = { k, min, weight };
            const task = metrics(original, config), real = metrics(axios, config);
            if (task.correct >= 97 && task.fp <= 2) eligible.push({ config, task, real });
        }
    }
}
eligible.sort((a, b) => b.real.correct - a.real.correct || a.real.fp - b.real.fp ||
    a.config.weight - b.config.weight || b.config.min - a.config.min);
console.log('top validation candidates:', eligible.slice(0, 3));
const selected = eligible[0];
if (selected) {
    const audit = read('blind_issue_audit.json').filter(row => row.label !== 'unclear');
    console.log('selected on original/Axios validation only:', selected);
    for (const repo of ['fastify/fastify', 'eslint/eslint']) {
        const group = audit.filter(row => row.url.includes('/' + repo + '/'));
        console.log('previously inspected audit ' + repo, metrics(group, baseline),
            '->', metrics(group, selected.config));
    }
    console.log('previously inspected audit total', metrics(audit, baseline),
        '->', metrics(audit, selected.config));
    const oldHoldout = read('real_issue_holdout.json');
    console.log('historical inspected holdout', metrics(oldHoldout, baseline),
        '->', metrics(oldHoldout, selected.config));
}
console.log('Do not select a release using the already inspected issue audit; this retrieval head is NOT shipped.');
