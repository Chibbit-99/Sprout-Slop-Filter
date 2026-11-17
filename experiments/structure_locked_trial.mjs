// ONE-TIME post-selection diagnostic. The natural issue titles and real
// commit subjects were labelled and partition-checked before any scoring.
// This opens the locked set ONLY for the validation-selected 0.4 mix.
// Once run, it becomes inspected and may not select a further candidate.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './structure_partition_guard.mjs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const half = load('.cache/structure-candidate-050.json');
const anchor = load('experiments/task_baseline_model.json');
const baseline = load('experiments/task_corrections_unanchored_model.json');
sprout.blend_models(baseline, anchor, 0.5);
for (const [category, weight] of Object.entries(read('lexicon_class_weights.json'))) {
    baseline.weights[7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536)] = weight;
}
baseline.options.lexicon = half.options.lexicon;
baseline.options.lexiconStrength = 1;
const chosen = JSON.parse(fs.readFileSync('.cache/structure-selected.json', 'utf8'));
assert.equal(chosen.mix, 0.4, 'only the preselected fraction may be opened');
const fraction = chosen.mix;
const candidate = { ...half, weights: half.weights.map((weight, i) =>
    (1 - fraction) * (2 * weight - anchor.weights[i]) + fraction * anchor.weights[i]),
    bias: (1 - fraction) * (2 * half.bias - anchor.bias) + fraction * anchor.bias };
for (const category of Object.keys(read('lexicon_class_weights.json'))) {
    const id = 7 * 65536 + sprout.hashed_feature_index('lex:' + category, 65536);
    candidate.weights[id] = half.weights[id];
}
assert.equal(sprout.model_parameter_stats(candidate).trainable, 524289);
const rows = read('structure_locked.json').filter(row => row.label !== 'unclear');
const metrics = (model, group) => {
    const m = sprout.evaluate_model(model, group);
    return { correct: m.truePositive + m.trueNegative, n: group.length,
        fp: m.falsePositive, fn: m.falseNegative };
};
const compare = group => ({ before: metrics(baseline, group), after: metrics(candidate, group) });
const issue = rows.filter(row => row.kind === 'issue');
const commit = rows.filter(row => row.kind === 'commit');
for (const repo of ['home-assistant/core', 'flutter/flutter', 'microsoft/playwright', 'denoland/deno']) {
    const group = rows.filter(row => row.url.includes('/' + repo + '/'));
    console.log(repo, compare(group));
    console.log('  genuine:', compare(group.filter(row => row.label === 'genuine')));
    console.log('  vague:', compare(group.filter(row => row.label === 'slop')));
}
const issues = compare(issue), commits = compare(commit);
console.log('LOCKED ISSUES', issues);
console.log('LOCKED COMMITS', commits);
const originalTask = sprout.split_dataset(read('../training.json'), { seed: 1 }).testSet;
const unfamiliar = Object.values(read('unfamiliar_cases.json').groups).flat();
const helpdesk = read('other_contexts.json').helpdesk.stress;
console.log('previously inspected task test', compare(originalTask));
console.log('previously inspected unfamiliar', compare(unfamiliar));
console.log('previously inspected helpdesk-hard', compare(helpdesk));
// Predeclared go/no-go rule: >=8 fewer natural-issue FPs with <=2 extra
// FNs, >=3 fewer commit FPs and >=3 more commit correct with <=2 extra FNs;
// no issue project loses more than 2 correct; historical task/unfamiliar/
// helpdesk diagnostics must remain at least 168/170, 35/40, 12/16.
const pass = issues.after.fp <= issues.before.fp - 8 && issues.after.fn <= issues.before.fn + 2 &&
    commits.after.fp <= commits.before.fp - 3 && commits.after.correct >= commits.before.correct + 3 &&
    commits.after.fn <= commits.before.fn + 2 &&
    ['home-assistant/core', 'flutter/flutter', 'microsoft/playwright'].every(repo => {
        const r = compare(issue.filter(row => row.url.includes('/' + repo + '/')));
        return r.after.correct >= r.before.correct - 2;
    }) && metrics(candidate, originalTask).correct >= 168 &&
    metrics(candidate, unfamiliar).correct >= 35 && metrics(candidate, helpdesk).correct >= 12;
console.log('PREDECLARED RELEASE GATE:', pass ? 'PASS' : 'FAIL — keep existing model');
if (pass) {
    candidate.options.trainedFrom.anchorMix = fraction;
    fs.writeFileSync('.cache/structure-locked-result.json', JSON.stringify({ issues, commits, pass }, null, 2) + '\n');
    const saved = sprout.save_model(candidate, '.cache/structure-selected-model.json', {
        storage: data => fs.writeFileSync('.cache/structure-selected-model.json', data)
    });
    assert.ok(saved.success, saved.error);
}
