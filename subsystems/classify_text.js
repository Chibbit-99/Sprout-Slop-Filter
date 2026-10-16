// Top-level coordinator that turns a raw string into a slop score.
//
// Steps: normalise → tokenise → optionally expand via synonyms → look up
// each token's vector → average them → predict.
//
// The synonym step is optional: if `useSynonyms` is false (the default in
// production paths) or `map_synonyms` fails, prediction still succeeds.
//
// Expected inputs:
//   text  — raw string to classify
//   model — loaded model object from `load_model`, with shape:
//           {
//               vocabulary: string[],
//               wordVectors: number[][],
//               weights: number[],
//               bias: number,
//               options: { useSynonyms?: boolean }
//           }
//   where  — async, in case synonym expansion is enabled
//
// Returns: a Promise<number> between 0 and 1.

async function classify_text(text, model) {
    if (typeof text !== "string") {
        throw new TypeError("classify_text: text must be a string");
    }

    if (!model || !Array.isArray(model.vocabulary) || !Array.isArray(model.wordVectors)) {
        throw new TypeError("classify_text: model is missing vocabulary or wordVectors");
    }

    const clean = normalise_text(text);
    const tokens = tokenise(clean);

    let wordLinks = [];
    if (model.options && model.options.useSynonyms && tokens.length > 0) {
        try {
            const linkArrays = await Promise.all(
                tokens.map(token => map_synonyms(token, model.vocabulary))
            );
            wordLinks = linkArrays.flat();
        } catch (error) {
            // Network or API failure — continue without synonym clues.
            wordLinks = [];
        }
    }

    const vectors = tokens.map(token =>
        get_word_vector(token, model.vocabulary, model.wordVectors, wordLinks)
    );

    const representation = encode_text(vectors);

    return predict_slop(representation, {
        weights: model.weights,
        bias: model.bias
    });
}