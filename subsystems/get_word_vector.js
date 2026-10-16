// Resolve a token to its numerical vector for the model.
//
// For a token that is in the vocabulary, this returns its learned vector.
// For an unknown token, it can borrow a known synonym's vector (treated as
// a clue, not a guarantee). If no useful link is found, it falls back to a
// consistent unknown-word representation so the classifier can still
// produce a score.
//
// Expected inputs:
//   token        — string, the word to look up
//   vocabulary   — array of known words (order matters: index = word id)
//   wordVectors  — array of vectors, one entry per vocabulary word
//   wordLinks    — optional array of { token, knownWord } records from
//                  `map_synonyms`. Multiple links per token are allowed;
//                  the returned vector is the average of every known
//                  synonym's vector, which dilutes any single bad match
//                  rather than committing fully to it.

function averageVectors(vectors) {
    if (vectors.length === 0) {
        return null;
    }

    const dimension = vectors[0].length;
    const sum = new Array(dimension).fill(0);

    for (const vector of vectors) {
        for (let i = 0; i < dimension; i++) {
            sum[i] += vector[i];
        }
    }

    return sum.map(value => value / vectors.length);
}

function get_word_vector(token, vocabulary, wordVectors, wordLinks) {
    // Keep this private constant local because subsystem files are loaded as
    // classic scripts and share a global lexical scope.
    const UNKNOWN_TOKEN = "<UNK>";

    if (!Array.isArray(vocabulary) || !Array.isArray(wordVectors)) {
        throw new TypeError("get_word_vector: vocabulary and wordVectors must be arrays");
    }

    const tokenIndex = vocabulary.indexOf(token);
    if (tokenIndex !== -1) {
        return wordVectors[tokenIndex].slice();
    }

    // Token is unknown: try synonym links before giving up.
    if (Array.isArray(wordLinks) && wordLinks.length > 0) {
        const linkedVectors = [];

        for (const link of wordLinks) {
            if (!link || link.token !== token) continue;

            const knownIndex = vocabulary.indexOf(link.knownWord);
            if (knownIndex !== -1) {
                linkedVectors.push(wordVectors[knownIndex]);
            }
        }

        const averaged = averageVectors(linkedVectors);
        if (averaged !== null) {
            return averaged;
        }
    }

    // No link or no useful link: use the consistent unknown representation.
    const unknownIndex = vocabulary.indexOf(UNKNOWN_TOKEN);
    if (unknownIndex !== -1) {
        return wordVectors[unknownIndex].slice();
    }

    // Vocabulary doesn't even define an <UNK> entry — return a zero vector
    // sized to the first known word so downstream code can still operate.
    if (wordVectors.length > 0) {
        return new Array(wordVectors[0].length).fill(0);
    }

    return [];
}