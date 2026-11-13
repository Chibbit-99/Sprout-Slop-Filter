// Rebuild the expanded default task corpus from the frozen 684-row baseline.
// The 1,600 two- to four-word artifact names are a selected, attributed subset
// of Princeton WordNet 3.1 noun.artifact lemmas (see WORDNET_LICENSE.txt).
// These are TEMPLATE-GENERATED examples, not independently labelled user text.
// Each term occurs in both labels so a rare noun is not itself a label cue.
import assert from "node:assert/strict";
import fs from "node:fs";
import { sprout, buildVocabulary } from "./lib/sprout.mjs";

const read = file => JSON.parse(fs.readFileSync(new URL(file, import.meta.url), "utf8"));
const write = (file, rows) => fs.writeFileSync(new URL(file, import.meta.url), JSON.stringify(rows, null, 2) + "\n");
const terms = read("task_artifact_terms.json");
const base = read("task_baseline_training.json");
const originalExtras = read("task_supplement.json").slice(0, 140);
const transfer = read("transfer_cases.json").tasks;
const unfamiliar = Object.values(read("unfamiliar_cases.json").groups).flat();
const blockedWords = new Set(unfamiliar.flatMap(row => row.unseen).concat(["glimmering", "fluxing"]));
const blockedTexts = new Set(Object.values(transfer).flat().map(sprout.normalise_text));
for (const row of unfamiliar) blockedTexts.add(sprout.normalise_text(row.text));
assert.equal(base.length, 684);
assert.equal(originalExtras.length, 140);
assert.equal(terms.length, 1600);
assert.equal(new Set(terms).size, terms.length);

const genuine = [
    term => `Photograph the ${term} for the equipment inventory`,
    term => `Record the condition of the ${term} before the site visit`,
    term => `Locate the ${term} in the storage register`,
    term => `Check the ${term} for damage during inspection`,
    term => `Add the ${term} to the asset catalogue`,
    term => `Write an inspection note for the ${term}`,
    term => `Verify the ${term} is listed on the property inventory`,
    term => `Update the maintenance record for the ${term}`
];
const slop = [
    term => `Think about the ${term} situation sometime`,
    term => `Do something with the ${term} eventually`,
    term => `Improve the ${term} experience in general`,
    term => `Consider possibilities around the ${term} at a high level`,
    term => `Move the needle on the ${term} somehow`,
    term => `Discuss the ${term} and other things`,
    term => `Explore vague opportunities involving the ${term}`,
    term => `Make the ${term} situation better somehow`
];

const seen = new Set([...base, ...originalExtras].map(row => sprout.normalise_text(row.text)));
function makePair(term, i) {
    assert.match(term, /^[a-z]+(?: [a-z]+)+$/);
    for (const token of term.split(" ")) assert.ok(!blockedWords.has(token), `protected OOV token: ${token}`);
    const pair = [
        { text: genuine[i % genuine.length](term), label: "genuine" },
        { text: slop[i % slop.length](term), label: "slop" }
    ];
    for (const row of pair) {
        const normalized = sprout.normalise_text(row.text);
        assert.ok(!seen.has(normalized) && !blockedTexts.has(normalized), `duplicate/probe leak: ${row.text}`);
        seen.add(normalized);
    }
    return pair;
}
const baseAdded = terms.slice(0, 220).flatMap(makePair);
const extraAdded = terms.slice(220).flatMap((term, i) => makePair(term, i + 220));
// Invert the seed-1 shuffle so the original 684-row train/validation/test
// partitions remain *exactly* where they were in the enlarged corpus. An
// ordinary concatenation would leak many old test rows into the new train set.
const originalSplit = sprout.split_dataset(base, { seed: 1 });
const generatedSplit = sprout.split_dataset(baseAdded, { seed: 1 });
const expected = {
    trainSet: originalSplit.trainSet.concat(generatedSplit.trainSet),
    validationSet: originalSplit.validationSet.concat(generatedSplit.validationSet),
    testSet: originalSplit.testSet.concat(generatedSplit.testSet)
};
const shuffled = [...expected.trainSet, ...expected.validationSet, ...expected.testSet];
const positions = sprout.split_dataset(Array.from({ length: shuffled.length }, (_, i) => i), { seed: 1 });
const expanded = new Array(shuffled.length);
for (const [index, position] of [...positions.trainSet, ...positions.validationSet, ...positions.testSet].entries()) {
    expanded[position] = shuffled[index];
}
const extras = originalExtras.concat(extraAdded);
const split = sprout.split_dataset(expanded, { seed: 1 });
for (const part of ["trainSet", "validationSet", "testSet"]) {
    assert.deepEqual(split[part], expected[part], `original ${part} must stay frozen`);
}
const trainRows = split.trainSet.concat(extras);
const trainWords = buildVocabulary(trainRows);
assert.ok(expanded.length >= 1000 && trainRows.length >= 1000);
assert.ok(trainWords.length >= 3000, `only ${trainWords.length} observed training tokens`);
assert.equal(new Set(trainRows.map(row => sprout.normalise_text(row.text))).size, trainRows.length);
assert.equal(expanded.filter(row => row.label === "genuine").length, expanded.length / 2);
assert.equal(extras.filter(row => row.label === "genuine").length, extras.length / 2);
write("../training.json", expanded);
write("task_supplement.json", extras);
console.log(`${expanded.length} displayed examples; ${trainRows.length} actual training examples ` +
    `(${split.trainSet.length} base + ${extras.length} training-only); ${trainWords.length} observed training tokens`);
console.log("Generated labels are illustrative; the original held-out rows are now in task_baseline_training.json.");
