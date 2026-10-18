// Build a model's vocabulary from a set of labelled examples.
//
// The vocabulary is the list of words the model knows, and the index of a
// word in that list is the index of its vector in `wordVectors`. Building it
// the same way at training time and at prediction time is what keeps the
// two in step, so this is shared rather than written out again by each
// caller.
//
// Expected input:
//   examples — array of { text, label } records
//
// Returns: an array of tokens, in the order they are first seen. The `<UNK>`
// entry is left out here; `initialise_model` adds it at index 0.

function build_vocabulary(examples) {
    if (!Array.isArray(examples)) {
        throw new TypeError("build_vocabulary: examples must be an array");
    }

    const seen = new Set();

    for (const example of examples) {
        if (!example || typeof example.text !== "string") continue;

        for (const token of tokenise(normalise_text(example.text))) {
            seen.add(token);
        }
    }

    return [...seen];
}
