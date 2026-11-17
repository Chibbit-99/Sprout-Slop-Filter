// CLI corrections belong to training only; neither seed-1 evaluation split
// nor the 2,900 supplemental rows may be lost or silently relabelled.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { sprout, buildVocabulary } from './lib/sprout.mjs';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sprout-feedback-'));
const rows = [{ text: 'Rebuild core_runtime.so for arm64', label: 'genuine' }];
const input = path.join(tmp, 'feedback.json');
const out = path.join(tmp, 'model.json');
const run = args => spawnSync(process.execPath, ['train.mjs',
    '--structure-version', '3', '--extra', 'experiments/task_supplement.json', '--epochs', '1', '--quiet',
    '--anchor-model', 'experiments/task_baseline_model.json', '--anchor-mix', '0.4',
    ...args, '--out', out], { encoding: 'utf8' });
try {
    fs.writeFileSync(input, JSON.stringify(rows));
    const real = 'experiments/real_issue_training.json';
    const corrections = 'experiments/issue_corrections_training.json';
    const realRows = JSON.parse(fs.readFileSync(real));
    const correctionRows = JSON.parse(fs.readFileSync(corrections));
    const result = run(['--feedback', real, '--feedback', corrections, '--feedback', input]);
    assert.equal(result.status, 0, result.stderr);
    const loaded = sprout.load_model(out, { storage: () => fs.readFileSync(out, 'utf8') });
    assert.ok(loaded.success, loaded.error);
    const model = loaded.model;
    assert.equal(model.options.structureVersion, 3);
    const expected = sprout.split_dataset(JSON.parse(fs.readFileSync('training.json')), { seed: 1 });
    const extras = JSON.parse(fs.readFileSync('experiments/task_supplement.json'));
    assert.equal(model.options.trainedFrom.trainSet,
        expected.trainSet.length + extras.length + realRows.length + correctionRows.length + rows.length);
    assert.equal(model.options.trainedFrom.testSet, expected.testSet.length);
    assert.equal(model.options.trainedFrom.validationSet, expected.validationSet.length);
    assert.equal(model.options.trainedFrom.feedbackRows, 243);
    assert.equal(model.options.trainedFrom.extraRows, 2900);
    assert.deepEqual(model.vocabulary.slice(1),
        Array.from(buildVocabulary(expected.trainSet.concat(extras, realRows, correctionRows, rows))));
    for (const text of ['review the data!', extras[0].text + '!']) {
        fs.writeFileSync(input, JSON.stringify([{ text, label: 'slop' }]));
        const rejected = run(['--feedback', real, '--feedback', corrections, '--feedback', input]);
        assert.notEqual(rejected.status, 0, 'duplicate evaluation/supplement row accepted');
        assert.match(rejected.stderr, /Feedback duplicates/);
    }
    fs.writeFileSync(input, JSON.stringify([{ text: realRows[0].text, label: 'slop' }]));
    const realDuplicate = run(['--feedback', real, '--feedback', corrections, '--feedback', input]);
    assert.notEqual(realDuplicate.status, 0);
    assert.match(realDuplicate.stderr, /Feedback duplicates/);
    fs.writeFileSync(input, JSON.stringify([{ text: correctionRows[0].text, label: 'slop' }]));
    const correctionDuplicate = run(['--feedback', real, '--feedback', corrections, '--feedback', input]);
    assert.notEqual(correctionDuplicate.status, 0);
    assert.match(correctionDuplicate.stderr, /Feedback duplicates/);
    const changedSplit = run(['--split-seed', '2', '--feedback', input]);
    assert.notEqual(changedSplit.status, 0);
    assert.match(changedSplit.stderr, /requires split seed 1/);
    console.log('CLI feedback keeps 3,929 training rows and the frozen 168 validation and 170 test rows; real correction duplicate and changed-split guards passed.');
} finally {
    fs.rmSync(tmp, { recursive: true, force: true });
}
