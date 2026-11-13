// Headless smoke test for the browser app in `index.html`.
//
// It loads the real page in jsdom against the real static server, waits for
// the page to boot, and then does what a user would do: read the dataset,
// score a sentence, score a list, train a model, read the metrics, change
// the softening, add an example, and save and reload the model.
//
// jsdom is not a dependency of this project. Install it once to run this:
//
//   npm install --no-save jsdom
//   node experiments/app_smoke.mjs
//
// Without jsdom the test explains itself and exits successfully, so it never
// blocks anything else.

import { startServer } from "../serve.mjs";

let JSDOM;
let VirtualConsole;

try {
    ({ JSDOM, VirtualConsole } = await import("jsdom"));
} catch (error) {
    console.log("jsdom is not installed — skipping the app smoke test.");
    console.log("Run: npm install --no-save jsdom && node experiments/app_smoke.mjs");
    process.exit(0);
}

const failures = [];
let total = 0;

function check(name, condition, detail) {
    total++;
    const ok = Boolean(condition);
    if (!ok) failures.push(name);
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
}

const { server, ready } = startServer(0, "127.0.0.1");
const { url } = await ready;

const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", error => {
    // jsdom has no canvas; the page guards against that.
    if (/getContext|Not implemented/i.test(error.message)) return;
    console.log("JSDOM ERROR:", error.message);
});
virtualConsole.on("error", (...args) => console.log("PAGE ERROR:", ...args));

const dom = await JSDOM.fromURL(url, {
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    virtualConsole
});

const { window } = dom;

// jsdom ships no fetch of its own — hand the page Node's, resolving the
// relative URLs it uses against the local server.
window.fetch = (input, init) => {
    const target = typeof input === "string" ? new URL(input, url).href : input;
    return fetch(target, init);
};

const $ = id => window.document.getElementById(id);

function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitFor(predicate, timeout = 20000) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
        if (predicate()) return true;
        await wait(50);
    }
    return false;
}

console.log(`=== App smoke test: ${url} ===`);
console.log("");

const booted = await waitFor(() => $("pillModel").textContent.includes("model.json"));
check("the page boots and loads the shipped model", booted, $("pillModel").textContent.trim());
check("the browser reports v4 parameter capacity and active weights",
    $("modelDetails").textContent.includes("524,289 trainable parameters") &&
    $("modelDetails").textContent.includes("nonzero weights"),
    $("modelDetails").textContent.trim());
check("browser counts actual training words and rows, not padded capacity",
    $("pillVocab").textContent.includes("3402 words") &&
    $("modelDetails").textContent.includes("3424 observed tokens") &&
    $("modelDetails").textContent.includes("trained on 3686 examples"));

const datasetLoaded = await waitFor(() => $("statExamples").textContent !== "0");
check("training.json is loaded", datasetLoaded, `${$("statExamples").textContent} examples`);
check("task training-only vocabulary set loads", $("supplementCount").textContent === "2900 extra rows");
check("the dataset is balanced",
    $("statGenuine").textContent === $("statSlop").textContent,
    `${$("statGenuine").textContent} genuine / ${$("statSlop").textContent} slop`);

const scored = await waitFor(() => $("scoreValue").textContent !== "—");
check("a sentence is scored on load", scored, `${$("scoreValue").textContent} → ${$("scoreVerdict").textContent}`);
check("a genuine sentence scores below 0.5",
    parseFloat($("scoreValue").textContent) < 0.5, $("scoreValue").textContent);

$("testInput").value = "asdfghjkl";
$("btnScore").click();
await waitFor(() => $("scoreValue").textContent !== "—");
check("a slop sentence scores above 0.5",
    parseFloat($("scoreValue").textContent) > 0.5, $("scoreValue").textContent);

$("testInput").value = "Water the plants on the balcony";
$("btnScore").click();
await wait(300);
check("the tokens of the text are shown",
    $("tokenList").querySelectorAll(".token").length > 0,
    `${$("tokenList").querySelectorAll(".token").length} tokens`);

$("btnSampleList").click();
await waitFor(() => $("batchRows").children.length > 0);
check("a list of sentences is scored", $("batchRows").children.length === 8, `${$("batchRows").children.length} rows`);

// A short training run keeps the test quick.
$("cfgEpochs").value = "40";
$("cfgVectorSize").value = "8";
$("btnTrain").click();
const trained = await waitFor(() => $("trainStatus").textContent === "finished", 120000);
check("training runs and finishes", trained, $("trainStatus").textContent);
check("browser trains on supplement without leaking into held-out rows",
    $("trainLog").textContent.includes("786 base + 2900 supplemental"),
    $("trainLog").textContent.split("\n").find(line => line.includes("training data")));
check("browser retraining uses the same v4 parameter budget and anchor recipe",
    $("modelDetails").textContent.includes("524,289 trainable parameters") &&
    $("modelDetails").textContent.includes("45% baseline-anchored") &&
    $("trainLog").textContent.includes("anchor        : 45% old task checkpoint"));
check("the held-out split is evaluated", $("mTestAcc").textContent !== "—", `test accuracy ${$("mTestAcc").textContent}`);
check("the confusion matrix is rendered", $("matrix").children.length === 9, `${$("matrix").children.length} cells`);
check("every held-out example is listed", $("testRows").children.length > 0, `${$("testRows").children.length} rows`);

$("testInput").value = "drive alignment across teams";
$("btnScore").click();
await wait(300);
check("the freshly trained model scores text",
    $("scoreValue").textContent !== "—", `score ${$("scoreValue").textContent}`);

// Softening: the score moves, the decision does not.
$("testInput").value = "align stakeholders";
$("btnScore").click();
await wait(300);
const softenedScore = parseFloat($("scoreValue").textContent);
const softenedVerdict = $("scoreVerdict").textContent;

$("cfgTemperature").value = "8";
$("cfgTemperature").dispatchEvent(new window.Event("change"));
await wait(400);
check("softening changes the score",
    parseFloat($("scoreValue").textContent) !== softenedScore,
    `${softenedScore} → ${$("scoreValue").textContent}`);
check("softening does not change the decision",
    $("scoreVerdict").textContent === softenedVerdict,
    `${softenedVerdict} → ${$("scoreVerdict").textContent}`);

$("cfgTemperature").value = "4";
$("cfgTemperature").dispatchEvent(new window.Event("change"));
await wait(300);

const before = parseInt($("statExamples").textContent, 10);
$("newExampleText").value = "Water the allotment";
$("btnNewSlop").click();
$("btnAddExample").click();
await wait(200);
check("an example can be added and stale evaluation is cleared",
    parseInt($("statExamples").textContent, 10) === before + 1 &&
    $("mTestAcc").textContent === "—" &&
    $("resultsMessage").textContent.includes("earlier evaluation"),
    `${before} → ${$("statExamples").textContent}`);

$("btnSaveLocal").click();
await wait(200);
$("btnLoadLocal").click();
await wait(200);
check("the model can be saved and reloaded in the browser",
    $("modelMessage").textContent.includes("loaded from this browser"),
    $("modelMessage").textContent.trim());

// A single default workflow: there is no mode picker or automatic domain switching.
check("no situation selector or preset model switch exists",
    !$("datasetPreset") && !window.document.querySelector("[name=datasetPreset]") &&
    $("pillModel").textContent.includes("browser storage"));
check("the default policy is explained without claiming general truth detection",
    window.document.querySelector(".help").textContent.includes("actionable request") &&
    window.document.querySelector(".help").textContent.includes("task-trained model"));

// Reset the trained model without changing the dataset or its label policy.
$("btnResetModel").click();
await waitFor(() => $("pillModel").textContent.includes("model.json"));
check("reset restores the one shipped model and its examples",
    $("statExamples").textContent === "1125" &&
    $("pillModel").textContent.includes("model.json") &&
    $("supplementCount").textContent === "2900 extra rows");
$("btnReloadDataset").click();
await waitFor(() => $("statExamples").textContent === "1124");
check("reload restores the default corpus without a preset selector",
    $("statExamples").textContent === "1124" && !$("datasetPreset"));
$("testInput").value = "asdfghjkl";
$("btnScore").click();
await waitFor(() => $("scoreVerdict").textContent === "reads like slop");
check("the reset model still scores text", $("scoreVerdict").textContent === "reads like slop");

// Any compatible labelled JSON can be imported directly, with no mode selector.
const imported = Array.from({ length: 12 }, (_, i) => ({
    text: i < 6 ? `Repair the damaged panel number ${i}` : `Handle miscellaneous matters number ${i}`,
    label: i < 6 ? "genuine" : "slop"
}));
const input = $("fileImportDataset");
Object.defineProperty(input, "files", { configurable: true, value: [
    new window.File([JSON.stringify(imported)], "my-examples.json", { type: "application/json" })
] });
input.dispatchEvent(new window.Event("change"));
const importedOK = await waitFor(() => $("statExamples").textContent === "12");
check("custom JSON imports without selecting a situation",
    importedOK && $("datasetMessage").textContent.includes("my-examples.json") &&
    !$("datasetPreset"));
check("import warns when the current model uses another corpus",
    $("modelDetails").textContent.includes("WARNING") &&
    $("modelDetails").textContent.includes("my-examples.json"));
$("cfgEpochs").value = "10";
$("btnTrain").click();
const customTrained = await waitFor(() => $("trainStatus").textContent === "finished" &&
    $("pillModel").textContent.includes("trained in this page"), 120000);
check("custom dataset trains without task-only supplements or baseline anchor", customTrained &&
    $("trainLog").textContent.includes("0 supplemental") &&
    !$("trainLog").textContent.includes("anchor        :") &&
    $("modelDetails").textContent.includes("524,289 trainable parameters"));
$("btnReloadDataset").click();
await waitFor(() => $("statExamples").textContent === "1124");
check("returning to default corpus clears custom evaluation",
    $("mTestAcc").textContent === "—" &&
    $("resultsMessage").textContent.includes("earlier evaluation"));

console.log("");
console.log(`${total - failures.length} passed, ${failures.length} failed`);

dom.window.close();
server.close();

process.exit(failures.length === 0 ? 0 : 1);
