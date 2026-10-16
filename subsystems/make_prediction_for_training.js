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
// Returns the predicted score (0..1) and the encoded representation, which
// `update_parameters` needs to backpropagate the error.

function make_prediction_for_training(text, modelParameters) {
    if (typeof text !== "string") {
        throw new TypeError("make_prediction_for_training: text must be a string");
    }

    if (!modelParameters || !Array.isArray(modelParameters.vocabulary)) {
        throw new TypeError("make_prediction_for_training: modelParameters missing");
    }

    const clean = normalise_text(text);
    const tokens = tokenise(clean);

    const vectors = tokens.map(token =>
        get_word_vector(token, modelParameters.vocabulary, modelParameters.wordVectors, [])
    );

    const representation = encode_text(vectors);
    const score = predict_slop(representation, {
        weights: modelParameters.weights,
        bias: modelParameters.bias
    });

    return { score, representation };
}