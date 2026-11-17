// Add a compact, trained WordNet class head and offline lexicon to an already
// trained v4 task model. This does NOT append fake training-observed words.
// Reuses 45 existing slots in the affix bank, retaining 524,289 parameters.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { sprout } from './lib/sprout.mjs';

const source = process.argv[2] || 'model.json';
const destination = process.argv[3] || source;
const load = file => {
    const raw = fs.readFileSync(file, 'utf8');
    const result = sprout.load_model(file, { storage: () => raw });
    assert.ok(result.success, result.error);
    return result.model;
};
const model = load(path.resolve(source));
const lexicon = JSON.parse(fs.readFileSync(new URL('wordnet_lexicon.json', import.meta.url), 'utf8'));
const head = JSON.parse(fs.readFileSync(new URL('lexicon_class_weights.json', import.meta.url), 'utf8'));
assert.equal(model.version, 4);
assert.equal(model.options.domain, 'training.json');
assert.equal(Object.keys(lexicon).length, 34037);
const categories = new Set(Object.values(lexicon).map(record => record[0]));
assert.deepEqual([...categories].sort(), Object.keys(head).sort());
for (const category of categories) {
    const id = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
    assert.ok(Number.isFinite(head[category]) && head[category] !== 0);
    model.weights[id] = head[category];
}
model.options.lexicon = lexicon;
model.options.lexiconStrength = 1;
model.options.lexiconSource = 'WordNet 3.1, Princeton University';
model.options.lexiconLicense = fs.readFileSync(new URL('WORDNET_LICENSE.txt', import.meta.url), 'utf8');
const recognized = new Set([...model.vocabulary.slice(1), ...Object.keys(lexicon)]);
assert.ok([...recognized].filter(word => /^\p{L}+$/u.test(word)).length >= 34020);
const saved = sprout.save_model(model, destination, { storage: payload =>
    fs.writeFileSync(path.resolve(destination), payload) });
assert.ok(saved.success, saved.error);
console.log(`${[...recognized].filter(word => /^\p{L}+$/u.test(word)).length} recognised alphabetic words; ` +
    `${model.vocabulary.filter(word => /^\p{L}+$/u.test(word)).length} observed in training; ` +
    `${sprout.model_parameter_stats(model).trainable} parameters; ${fs.statSync(path.resolve(destination)).size} bytes`);
