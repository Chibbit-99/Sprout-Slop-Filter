// Single opening of the Pandas/Electron titles and authored contrast pairs.
// This gate is declared after development selection but BEFORE scoring these
// partitions. A failure leaves the shipped version-3 checkpoint unchanged.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import './relational_partition_guard.mjs';
import { sprout } from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), 'utf8'));
const load = file => {
    const result = sprout.load_model(file, { storage: () => fs.readFileSync(file, 'utf8') });
    assert.ok(result.success, result.error);
    return result.model;
};
const baseline = load('model.json');
assert.equal(baseline.options.structureVersion, 3);
const chosen = read('../.cache/relational-guarded-selection.json');
assert.equal(chosen.mix, 0.2);
const candidate = load('.cache/relational-guarded-selected.json');
assert.equal(candidate.options.structureVersion, 4);
assert.equal(candidate.options.trainedFrom.anchorMix, 0.2);
assert.equal(candidate.options.trainedFrom.feedbackRows, 266);
assert.equal(sprout.model_parameter_stats(candidate).trainable, 524289);
const metric = (model, group) => {
    const r = sprout.evaluate_model(model, group);
    return { correct: r.truePositive + r.trueNegative, n: group.length,
        fp: r.falsePositive, fn: r.falseNegative };
};
const compare = group => ({ before: metric(baseline, group), after: metric(candidate, group) });
// Previous gates are now inspected regression diagnostics, not new tests.
const task = compare(sprout.split_dataset(read('../training.json'), { seed: 1 }).testSet);
const unfamiliar = compare(Object.values(read('unfamiliar_cases.json').groups).flat());
const helpdesk = compare(read('other_contexts.json').helpdesk.stress);
const former = read('structure_fresh_locked.json').filter(row => row.label !== 'unclear');
const formerScore = compare(former);
const historicGate = task.after.correct >= 168 && task.after.fp === 0 &&
    unfamiliar.after.correct >= 36 && helpdesk.after.correct >= 12 &&
    formerScore.after.correct >= formerScore.before.correct - 2 &&
    formerScore.after.fn <= formerScore.before.fn + 1;
// New, source-separated clear rows and balanced *authored* counterfactual
// probes are both evaluated but never combined into a deployment estimate.
const rows = read('relational_locked.json').filter(row => row.label !== 'unclear');
const issue = compare(rows.filter(row => row.kind === 'issue'));
const commit = compare(rows.filter(row => row.kind === 'commit'));
const total = compare(rows);
const pandas = compare(rows.filter(row => row.url.includes('/pandas-dev/pandas/')));
const electron = compare(rows.filter(row => row.url.includes('/electron/electron/')));
const probeRows = read('relational_locked_pairs.json');
const probes = compare(probeRows);
const rank = model => new Set(probeRows.map(row => row.pair)).size -
    [...new Set(probeRows.map(row => row.pair))].filter(pair => {
        const genuine = probeRows.find(row => row.pair === pair && row.label === 'genuine');
        const slop = probeRows.find(row => row.pair === pair && row.label === 'slop');
        return sprout.make_prediction_for_training(genuine.text, model).score >=
            sprout.make_prediction_for_training(slop.text, model).score;
    }).length;
// Predeclared fresh gate: genuine titles improve without trading away the
// rare slop titles; neither project deteriorates, and condition-vs-generic
// contrast pairs do not regress in accuracy or ordering.
const freshGate = issue.after.correct >= issue.before.correct + 2 &&
    commit.after.correct >= commit.before.correct + 2 &&
    total.after.correct >= total.before.correct + 6 &&
    total.after.fp <= total.before.fp - 5 &&
    issue.after.fn <= issue.before.fn + 1 && commit.after.fn <= commit.before.fn + 1 &&
    pandas.after.correct >= pandas.before.correct - 2 &&
    electron.after.correct >= electron.before.correct - 2 &&
    probes.after.correct >= probes.before.correct && rank(candidate) >= rank(baseline);
console.log('INSPECTED TASK', task, 'UNFAMILIAR', unfamiliar, 'HELPDESK', helpdesk);
console.log('INSPECTED FOUR-PROJECT TITLE SET', formerScore);
console.log('NEW PANDAS ISSUES', issue, 'NEW ELECTRON COMMITS', commit, 'TOTAL', total);
console.log('NEW AUTHORED PAIRS', probes, 'correct ordering', rank(baseline), '→', rank(candidate));
console.log('INSPECTED REGRESSION GATE:', historicGate ? 'PASS' : 'FAIL');
console.log('PREDECLARED FRESH GATE:', freshGate ? 'PASS' : 'FAIL');
console.log('DECISION:', historicGate && freshGate ? 'PASS — eligible for integration' : 'FAIL — keep version 3');
if (historicGate && freshGate) fs.writeFileSync('.cache/relational-locked-result.json',
    JSON.stringify({ task, unfamiliar, helpdesk, formerScore, issue, commit, total, probes,
        rankBefore: rank(baseline), rankAfter: rank(candidate) }, null, 2) + '\n');
else process.exitCode = 1;
