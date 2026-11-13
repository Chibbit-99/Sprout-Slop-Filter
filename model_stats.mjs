// Inspect a trained model without training or scoring any text.
// node model_stats.mjs [model.json | commit_model.json | custom.json]
import fs from "node:fs";
import { sprout } from "./experiments/lib/sprout.mjs";

const file = process.argv[2] || "model.json";
const raw = fs.readFileSync(file, "utf8");
const result = sprout.load_model(file, { storage: () => raw });
if (!result.success) throw new Error(result.error);
console.log(JSON.stringify({ file, diskBytes: Buffer.byteLength(raw),
    ...sprout.model_parameter_stats(result.model) }, null, 2));
