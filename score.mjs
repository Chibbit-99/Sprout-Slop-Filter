// Local text scoring for the selected label policy, no API calls.
// Examples:
//   node score.mjs --domain commits "Fix retry loop after timeout"
//   git log -1 --pretty=%s | node score.mjs --domain commits
import fs from "node:fs";
import { sprout } from "./experiments/lib/sprout.mjs";

const args = process.argv.slice(2);
let domain = "tasks";
let modelFile = null;
let threshold = 0.5;
const words = [];
for (let i = 0; i < args.length; i++) {
    if (args[i] === "--domain") domain = args[++i];
    else if (args[i] === "--model") modelFile = args[++i];
    else if (args[i] === "--threshold") threshold = Number(args[++i]);
    else if (args[i] === "--help") {
        console.log("node score.mjs [--domain tasks|commits|helpdesk|agendas] [--model file] [--threshold 0..1] [text] (or pipe text on stdin)");
        process.exit(0);
    } else words.push(args[i]);
}
const domains = { tasks: "model.json", commits: "commit_model.json",
    helpdesk: "helpdesk_model.json", agendas: "agenda_model.json" };
if (!Object.hasOwn(domains, domain) || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error("Use --domain tasks|commits|helpdesk|agendas and --threshold between 0 and 1");
}
modelFile ||= domains[domain];
const result = sprout.load_model(modelFile, { storage: () => fs.readFileSync(modelFile, "utf8") });
if (!result.success) throw new Error(result.error);
const text = words.length ? words.join(" ") : fs.readFileSync(0, "utf8").trim();
if (!text) throw new Error("Pass a message as an argument or on stdin");
const score = await sprout.classify_text(text, result.model);
const verdicts = { tasks: ["actionable", "generic/filler"],
    commits: ["informative message", "generic message"],
    helpdesk: ["specific report", "report needs detail"],
    agendas: ["specific agenda", "vague agenda"] };
console.log(JSON.stringify({ text, domain, score: Number(score.toFixed(4)),
    verdict: verdicts[domain][Number(score >= threshold)], threshold }));
