// Rebuild the small, checked-in class head from a separately trained v4
// WordNet-cue trial. This is a training-time step, not required for inference.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';

const file = process.argv[2] || 'experiments/task_lexicon_trial_model.json';
const raw = fs.readFileSync(file, 'utf8');
const result = sprout.load_model(file, { storage: () => raw });
assert.ok(result.success, result.error);
const model = result.model;
assert.equal(model.version, 4);
assert.equal(model.options.domain, 'training.json');
const categories = [...new Set(Object.values(model.options.lexicon).map(record => record[0]))].sort();
const head = Object.fromEntries(categories.map(category => [category,
    model.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)]]));
assert.equal(categories.length, 45);
assert.ok(Object.values(head).every(weight => Number.isFinite(weight) && weight !== 0));
const target = new URL('lexicon_class_weights.json', import.meta.url);
fs.writeFileSync(target, JSON.stringify(head, null, 2) + '\n');
console.log(`Saved ${categories.length} lexical class weights from ${file}`);
