// Loads the real Sprout subsystem files the same way test.html does — as
// classic scripts sharing one global scope — so experiments measure the
// actual implementation, not a reimplementation of it.

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const subsystemDir = path.join(here, "..", "..", "subsystems");

const source = fs
    .readdirSync(subsystemDir)
    .filter(name => name.endsWith(".js"))
    .sort()
    .map(name => fs.readFileSync(path.join(subsystemDir, name), "utf8"))
    .join("\n");

const context = vm.createContext({ console, fetch, URL, Set, Map, Math, JSON, Promise, Number, Array, Object });

vm.runInContext(
    `${source}
     globalThis.__sprout = {
         normalise_text, tokenise, encode_text, predict_slop,
         calculate_loss, update_parameters, make_prediction_for_training,
         evaluate_model, train_model, initialise_model, get_word_vector
     };`,
    context
);

export const sprout = context.__sprout;
