// Realistic-difficulty experiment.
//
// The first learning curve used easy data: keyboard mash ("asdfghjkl") versus
// concrete nouns. Real slop is harder — it is grammatical, uses ordinary
// English words, and often sounds plausible. This experiment tests that case:
//
//   Genuine — a specific verb acting on a specific, checkable object
//             ("rotate the api key", "email the landlord about the leak")
//   Slop    — fluent corporate filler with no checkable object
//             ("drive alignment", "leverage best practices", "do the needful")
//
// Both classes share function words and some content words, so the model
// cannot win by detecting junk characters. This is the regime that actually
// matters, and it is the one the README's "Do something -> depends" example
// warns about.
//
// Run: node experiments/realistic_hard.mjs
//
// Trains with the real subsystem trainer (`sprout.train_model`), which since
// the word-vector fix (FINDINGS.md section 1) trains word vectors alongside
// the classifier.

import { sprout } from "./lib/sprout.mjs";
import { makeRandom, buildVocabulary, metricsFor, average, standardDeviation, pct } from "./lib/data.mjs";

// Genuine: specific and checkable.
const GENUINE = [
    "rotate the api key", "email the landlord about the leak",
    "renew the domain before friday", "call the dentist back",
    "submit the expense claim", "fix the login redirect bug",
    "book the meeting room for tuesday", "update the emergency contact",
    "replace the smoke alarm battery", "pay the council rates",
    "print the boarding passes", "cancel the gym membership",
    "refund the duplicate charge", "upload the signed contract",
    "chase the supplier for the invoice", "change the brake pads",
    "draft the quarterly report", "review the pull request",
    "take the parcel to the post office", "water the tomato plants",
    "renew the passport photo", "test the backup restore",
    "sharpen the kitchen knives", "return the wrong size shoes",
    "clear the blocked drain", "renew the car insurance",
    "reply to the client about pricing", "insulate the loft hatch",
    "register the new laptop", "patch the staging server",
    "unsubscribe from the spam list", "measure the window for curtains"
];

// Slop: fluent, plausible, uncheckable.
const SLOP = [
    "drive alignment across teams", "leverage best practices",
    "do the needful", "improve things", "look into it",
    "focus on growth", "enhance the experience", "optimise the workflow",
    "streamline operations", "move the needle", "take it to the next level",
    "circle back on this", "touch base later", "add value",
    "synergise the deliverables", "empower stakeholders",
    "foster a culture of excellence", "unlock potential",
    "think outside the box", "get everyone on the same page",
    "action the takeaways", "socialise the idea", "pivot the strategy",
    "double down on core competencies", "right size the organisation",
    "create a seamless journey", "champion innovation",
    "run it up the flagpole", "boil the ocean", "land the plane",
    "peel the onion", "raise the bar", "shift the paradigm",
    "do more with less", "make it pop", "give it some love",
    "sort it out", "deal with it", "handle it", "improve the thing"
];

// Held out entirely from training — the model must cope with new wording.
const HELD_OUT_GENUINE = [
    "hoover the car boot", "descale the kettle", "reheel the boots",
    "prune the hedge", "defrost the freezer", "sand the decking",
    "unclog the plughole", "rehang the picture rail",
    "sweep the chimney", "wax the surfboard"
];

const HELD_OUT_SLOP = [
    "align the synergies", "leverage the paradigm",
    "unlock the ecosystem", "champion the roadmap",
    "foster the mindset", "action the alignment",
    "socialise the framework", "right size the narrative",
    "double down on the vibe", "peel the strategic onion"
];

const TRAIN_SIZES = [20, 40, 80, 160, 320, 640];
const SEEDS = [1, 2, 3, 4, 5];
const EPOCHS = 300;
const LEARNING_RATE = 0.2;

// Draw `n` examples alternating classes, sampling within each pool.
function sampleSet(n, random, genuinePool, slopPool) {
    const examples = [];
    for (let i = 0; i < n; i++) {
        if (i % 2 === 0) {
            examples.push({ text: genuinePool[Math.floor(random() * genuinePool.length)], label: "genuine" });
        } else {
            examples.push({ text: slopPool[Math.floor(random() * slopPool.length)], label: "slop" });
        }
    }
    return examples;
}

const testSeen = [
    ...GENUINE.map(text => ({ text, label: "genuine" })),
    ...SLOP.map(text => ({ text, label: "slop" }))
];

const testUnseen = [
    ...HELD_OUT_GENUINE.map(text => ({ text, label: "genuine" })),
    ...HELD_OUT_SLOP.map(text => ({ text, label: "slop" }))
];

console.log("=== Realistic-difficulty learning curve (fluent slop, shared vocabulary) ===");
console.log(`Verdict threshold: a false positive is a genuine task wrongly flagged as slop.`);
console.log("");
console.log("train | vocab | ex/word | trainAcc | SEEN acc (sd) | slopRecall | genuineRecall | falsePos | UNSEEN acc");
console.log("-".repeat(112));

for (const size of TRAIN_SIZES) {
    const seen = [], unseen = [], train = [], slopRecall = [], genuineRecall = [], fpr = [], vocab = [];

    for (const seed of SEEDS) {
        const random = makeRandom(seed * 7919 + size);
        // Sample without letting a single pool entry repeat often: cap at the
        // pool size so the model sees the whole distribution.
        const trainSet = sampleSet(size, random, GENUINE, SLOP);
        const model = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed });

        sprout.train_model(trainSet, [], model, { epochs: EPOCHS, learningRate: LEARNING_RATE });

        const s = metricsFor(model, testSeen);
        const u = metricsFor(model, testUnseen);
        const t = metricsFor(model, trainSet);

        seen.push(s.accuracy);
        unseen.push(u.accuracy);
        train.push(t.accuracy);
        slopRecall.push(s.slopRecall);
        genuineRecall.push(1 - s.falsePositiveRate);
        fpr.push(s.falsePositiveRate);
        vocab.push(model.vocabulary.length);
    }

    const meanVocab = Math.round(average(vocab));
    console.log(
        [
            String(size).padStart(5),
            String(meanVocab).padStart(5),
            (size / meanVocab).toFixed(1).padStart(7),
            pct(average(train)).padStart(8),
            `${pct(average(seen))} (${(standardDeviation(seen) * 100).toFixed(1)})`.padStart(15),
            pct(average(slopRecall)).padStart(10),
            pct(average(genuineRecall)).padStart(13),
            pct(average(fpr)).padStart(8),
            pct(average(unseen)).padStart(10)
        ].join(" | ")
    );
}

console.log("");
console.log("Note: the pools repeat, so beyond ~72 examples (the pool size) the model");
console.log("starts re-seeing identical sentences. 'ex/word' is training examples per");
console.log("distinct vocabulary word — the ratio that drives whether a word's polarity");
console.log("can be learned from the data available.");
