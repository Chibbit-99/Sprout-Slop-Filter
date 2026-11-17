// Diagnostic comparison, NOT a deployment accuracy estimate. The historical
// validation, held-out rows and handcrafted probes have been inspected.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(new URL(file, import.meta.url), 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const base = load('task_structure_v3_unanchored_model.json');
sprout.blend_models(base, load('task_baseline_model.json'), 0.4);
const model = load('../model.json');
const full = sprout.split_dataset(read('../training.json'), { seed: 1 });
const frozen = sprout.split_dataset(read('task_baseline_training.json'), { seed: 1 });
const diagnostic = read('unfamiliar_cases.json').groups;
const other = read('other_contexts.json');
const groups = { originalValidation: frozen.validationSet, expandedValidation: full.validationSet,
    expandedHeldout: full.testSet, originalHeldout: frozen.testSet,
    ...diagnostic, helpdeskHard: other.helpdesk.stress, agendaHard: other.meeting_agendas.stress };
const metric = (m, rows) => {
    const r = sprout.evaluate_model(m, rows);
    return `${r.truePositive + r.trueNegative}/${r.count} FP ${r.falsePositive} FN ${r.falseNegative}`;
};
console.log('                     base                 lexical');
for (const [name, rows] of Object.entries(groups)) {
    console.log(`${name.padEnd(23)}${metric(base, rows).padEnd(21)}${metric(model, rows)}`);
}
for (const text of ['Verify the ammeter before Monday', 'Repaint the drawbar with epoxy',
    'Retest the photometer at noon']) {
    const prior = sprout.make_prediction_for_training(text, base).score;
    const next = sprout.make_prediction_for_training(text, model).score;
    assert.notEqual(prior, next, `lexical cue inactive for ${text}`);
    console.log(`${text}: ${prior.toFixed(4)} → ${next.toFixed(4)} (authored example, not an accuracy measurement)`);
}
assert.equal(sprout.model_parameter_stats(base).trainable, sprout.model_parameter_stats(model).trainable);
console.log('This isolates the lexical head within the correction-trained checkpoint; it does not measure the effect of real training rows.');
