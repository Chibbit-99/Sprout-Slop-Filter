// Rebuild the pre-structural, correction-trained v4 reference without relying
// on whichever checkpoint is currently shipped in model.json.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(new URL(file, import.meta.url), 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
export function correctionReference() {
    const model = load('task_corrections_unanchored_model.json');
    sprout.blend_models(model, load('task_baseline_model.json'), 0.5);
    for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
        const id = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
        model.weights[id] = weight;
    }
    model.options.lexicon = read('wordnet_lexicon.json');
    model.options.lexiconStrength = 1;
    assert.equal(model.options.structureVersion, undefined);
    return model;
}
