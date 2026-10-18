// How much data do you need when some labels are genuinely debatable?
//
// The earlier curves used cleanly separable classes, which is optimistic.
// The README itself flags the hard case: "Do something" is "possibly slop"
// depending on the intended definition. Real task trackers contain text that
// reasonable people would label differently, and that creates irreducible
// error that no amount of data removes.
//
// This experiment mixes in an ambiguous pool: phrases that appear in BOTH
// classes with a random label. As `ambiguousShare` rises, the ceiling falls
// and the amount of data needed to approach that ceiling rises.
//
// Run: node experiments/ambiguity.mjs
//
// Trains with the real subsystem trainer (`sprout.train_model`), which since
// the word-vector fix (FINDINGS.md section 1) trains word vectors alongside
// the classifier.

import { sprout } from "./lib/sprout.mjs";
import { makeRandom, buildVocabulary, metricsFor, average, standardDeviation, pct } from "./lib/data.mjs";

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

// Genuinely debatable: short, vague, but could be a real placeholder task.
const AMBIGUOUS = [
    "do something", "check things", "review stuff", "tidy up",
    "improve the process", "clean things up", "sort the list",
    "handle the emails", "look at the numbers", "fix the thing",
    "update the docs", "organise the folder", "plan the week",
    "follow up", "look into options", "investigate the issue"
];

// Held out from training entirely.
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

const HELD_OUT_AMBIGUOUS = [
    "do the thing", "check the stuff", "review the bits",
    "tidy the area", "improve the situation", "sort the things"
];

const TRAIN_SIZES = [40, 80, 160, 320, 640, 1280];
// Contradictory labels make per-seed results noisy, so this experiment uses
// more seeds than the others (it originally ran with 3).
const SEEDS = [1, 2, 3, 4, 5];
const EPOCHS = 300;
const LEARNING_RATE = 0.2;

function pick(array, random) {
    return array[Math.floor(random() * array.length)];
}

// Build a training set where `ambiguousShare` of examples come from the
// ambiguous pool with a coin-flip label, and the rest are clean examples.
function sampleSet(n, ambiguousShare, random) {
    const examples = [];
    for (let i = 0; i < n; i++) {
        if (random() < ambiguousShare) {
            examples.push({
                text: pick(AMBIGUOUS, random),
                label: random() < 0.5 ? "genuine" : "slop"
            });
        } else if (i % 2 === 0) {
            examples.push({ text: pick(GENUINE, random), label: "genuine" });
        } else {
            examples.push({ text: pick(SLOP, random), label: "slop" });
        }
    }
    return examples;
}

const testSet = [
    ...HELD_OUT_GENUINE.map(text => ({ text, label: "genuine" })),
    ...HELD_OUT_SLOP.map(text => ({ text, label: "slop" })),
    ...HELD_OUT_AMBIGUOUS.map(text => ({ text, label: "genuine" })),
    ...HELD_OUT_AMBIGUOUS.map(text => ({ text, label: "slop" }))
];

const SHARES = [0, 0.1, 0.25, 0.5];

console.log("=== Data needed versus label ambiguity ===");
console.log("Held-out test set includes 12 deliberately debatable items.");
console.log("");
console.log("ambiguous |  train | seenAcc (sd) | slopRecall | falsePosRate");
console.log("-".repeat(66));

for (const share of SHARES) {
    for (const size of TRAIN_SIZES) {
        const seen = [], recall = [], fpr = [];

        for (const seed of SEEDS) {
            const random = makeRandom(seed * 7919 + size + Math.round(share * 1000));
            const trainSet = sampleSet(size, share, random);
            const model = sprout.initialise_model({ vocabulary: buildVocabulary(trainSet), seed });

            sprout.train_model(trainSet, [], model, { epochs: EPOCHS, learningRate: LEARNING_RATE });

            const m = metricsFor(model, testSet);
            seen.push(m.accuracy);
            recall.push(m.slopRecall);
            fpr.push(m.falsePositiveRate);
        }

        console.log(
            [
                pct(share).padStart(9),
                String(size).padStart(6),
                `${pct(average(seen))} (${(standardDeviation(seen) * 100).toFixed(1)})`.padStart(15),
                pct(average(recall)).padStart(10),
                pct(average(fpr)).padStart(12)
            ].join(" | ")
        );
    }
    console.log("");
}
