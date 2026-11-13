// Ship independent, text-only detectors for two other label policies.
// Only the 60 explicitly training-only examples in each file are used.
// The ordinary/hard probes in other_contexts.json are NEVER imported here.
import fs from "node:fs";
import { sprout, buildVocabulary } from "./lib/sprout.mjs";

for (const [domain, input, output] of [
    ["helpdesk", "helpdesk_training.json", "helpdesk_model.json"],
    ["agendas", "agenda_training.json", "agenda_model.json"]
]) {
    const rows = sprout.parse_training_data(JSON.parse(fs.readFileSync(input, "utf8")), input);
    const model = sprout.initialise_model({ encoder: "richer-subword-v4",
        vocabulary: buildVocabulary(rows), seed: 7, epochs: 120, learningRate: 0.8 });
    sprout.train_model(rows, [], model, { epochs: 120, learningRate: 0.8 });
    model.weights = model.weights.map(weight => weight / 4);
    model.bias /= 4;
    model.options = { ...model.options, domain: input, temperature: 4, trainedFrom: {
        data: input, examples: rows.length, trainSet: rows.length, validationSet: 0,
        testSet: 0, encoder: "richer-subword-v4", seed: 7, epochs: 120, learningRate: 0.8,
        note: "All 60 rows train; separate diagnostics are not used for fitting."
    } };
    const saved = sprout.save_model(model, output, { storage: payload => fs.writeFileSync(output, payload) });
    if (!saved.success) throw new Error(saved.error);
    console.log(`${domain}: ${rows.length} labelled examples -> ${output}; ` +
        `${sprout.model_parameter_stats(model).trainable} trainable parameters, ` +
        `${fs.statSync(output).size} bytes`);
}
