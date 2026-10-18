// Predict a slop score for one training example using the current model
// parameters. This is the "forward pass" used during training; it must
// follow exactly the same tokenisation and vocabulary rules as the
// production `classify_text` so the model learns from the same
// representation it will later receive at inference time.
//
// Note: this implementation is synchronous because training does not use
// synonym lookups (which would be slow and non-deterministic across calls).
// The classifier still benefits from whatever vectors the model has
// learned for its vocabulary.
//
// Returns the predicted score (0..1), the encoded representation, and the
// `tokenIndices` — the vocabulary index of every token that contributed to
// the representation, mirroring the lookup rules of `get_word_vector`
// (a known token maps to its own entry, an unknown token to the shared
// `<UNK>` entry). `update_parameters` needs both the representation and
// the token indices to backpropagate the error into the classifier weights
// AND the word vectors.

function make_prediction_for_training(text, modelParameters) {
    if (typeof text !== "string") {
        throw new TypeError("make_prediction_for_training: text must be a string");
    }

    if (!modelParameters || !Array.isArray(modelParameters.vocabulary)) {
        throw new TypeError("make_prediction_for_training: modelParameters missing");
    }

    // Keep this private constant local because subsystem files are loaded as
    // classic scripts and share a global lexical scope.
    const UNKNOWN_TOKEN = "<UNK>";

    const clean = normalise_text(text);
    const tokens = tokenise(clean);

    const unknownIndex = modelParameters.vocabulary.indexOf(UNKNOWN_TOKEN);

    const vectors = [];
    const tokenIndices = [];

    for (const token of tokens) {
        vectors.push(get_word_vector(token, modelParameters.vocabulary, modelParameters.wordVectors, []));

        // Record which learnable vector this token used. If a token is
        // unknown and the vocabulary has no `<UNK>` entry, `get_word_vector`
        // falls back to an unlearnable zero vector, so there is no index to
        // record for it.
        const index = modelParameters.vocabulary.indexOf(token);
        if (index !== -1) {
            tokenIndices.push(index);
        } else if (unknownIndex !== -1) {
            tokenIndices.push(unknownIndex);
        }
    }

    const representation = encode_text(vectors);
    const score = predict_slop(representation, {
        weights: modelParameters.weights,
        bias: modelParameters.bias
    });

    return { score, representation, tokenIndices };
}
