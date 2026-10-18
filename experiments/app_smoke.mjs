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

const datasetLoaded = await waitFor(() => $("statExamples").textContent !== "0");
check("training.json is loaded", datasetLoaded, `${$("statExamples").textContent} examples`);
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
check("an example can be added",
    parseInt($("statExamples").textContent, 10) === before + 1,
    `${before} → ${$("statExamples").textContent}`);

$("btnSaveLocal").click();
await wait(200);
$("btnLoadLocal").click();
await wait(200);
check("the model can be saved and reloaded in the browser",
    $("modelMessage").textContent.includes("loaded from this browser"),
    $("modelMessage").textContent.trim());

console.log("");
console.log(`${total - failures.length} passed, ${failures.length} failed`);

dom.window.close();
server.close();

process.exit(failures.length === 0 ? 0 : 1);
