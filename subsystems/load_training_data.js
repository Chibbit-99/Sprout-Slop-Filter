// Load labelled examples from a JSON file.
//
// The expected file shape is:
//   [
//       { "text": "Buy milk",       "label": "genuine" },
//       { "text": "asdfghjkl",      "label": "slop" },
//       ...
//   ]
//
// `load_training_data(file)` fetches the file and validates every entry.
// It throws on missing fields or unknown labels so a bad training file is
// caught loudly, not silently.
//
// Returns: a Promise resolving to an array of { text, label } records.
// Labels are normalised to "genuine" or "slop" (lowercase, trimmed).

const VALID_LABELS = new Set(["genuine", "slop"]);

async function load_training_data(file) {
    if (typeof fetch !== "function") {
        throw new Error("load_training_data: fetch is not available in this environment");
    }

    const response = await fetch(file);

    if (!response.ok) {
        throw new Error(`load_training_data: failed to load ${file} (status ${response.status})`);
    }

    const data = await response.json();

    if (!Array.isArray(data)) {
        throw new TypeError("load_training_data: expected an array of examples");
    }

    const examples = [];

    for (let i = 0; i < data.length; i++) {
        const entry = data[i];

        if (!entry || typeof entry.text !== "string" || entry.text.length === 0) {
            throw new Error(`load_training_data: example ${i} is missing valid text`);
        }

        const label = typeof entry.label === "string"
            ? entry.label.trim().toLowerCase()
            : "";

        if (!VALID_LABELS.has(label)) {
            throw new Error(
                `load_training_data: example ${i} has invalid label "${entry.label}" (expected "genuine" or "slop")`
            );
        }

        examples.push({ text: entry.text, label });
    }

    return examples;
}