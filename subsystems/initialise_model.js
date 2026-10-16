// Initialise a fresh model's parameters and vocabulary.
//
// Vocabulary order matters: the index of a word in the vocabulary is the
// index of its vector in `wordVectors`. An `<UNK>` token is always added
// (or moved to index 0 if it isn't already present) so unknown words have
// a consistent representation during training and prediction.
//
// Word vectors are drawn from a small uniform distribution to give
// training something to work with without exploding gradients.
//
// Expected input:
//   settings — {
//       vectorSize: number,   // dimensions per word vector (default 8)
//       seed?:     number,   // PRNG seed for reproducibility
//       random?:   function  // optional custom random source
//   }
//
// Returns: { vocabulary, wordVectors, weights, bias, options }

const UNKNOWN_TOKEN = "<UNK>";

function makeRandom(seed) {
    if (typeof seed === "number") {
        let a = seed >>> 0;
        return function () {
            a = (a + 0x6D2B79F5) >>> 0;
            let t = a;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    return Math.random;
}

function initialise_model(settings) {
    const config = settings || {};
    const vectorSize = typeof config.vectorSize === "number" && config.vectorSize > 0
        ? Math.floor(config.vectorSize)
        : 8;
    const random = typeof config.random === "function"
        ? config.random
        : makeRandom(typeof config.seed === "number" ? config.seed : undefined);

    const baseVocabulary = Array.isArray(config.vocabulary)
        ? config.vocabulary.slice()
        : [];

    const vocabulary = [UNKNOWN_TOKEN, ...baseVocabulary.filter(word => word !== UNKNOWN_TOKEN)];

    const wordVectors = vocabulary.map(() => {
        const vector = new Array(vectorSize);
        for (let i = 0; i < vectorSize; i++) {
            // Small symmetric range so neither positive nor negative bias
            // dominates at start.
            vector[i] = (random() * 2 - 1) * 0.1;
        }
        return vector;
    });

    const weights = new Array(vectorSize);
    for (let i = 0; i < vectorSize; i++) {
        weights[i] = (random() * 2 - 1) * 0.1;
    }

    return {
        vocabulary,
        wordVectors,
        weights,
        bias: 0,
        options: {
            useSynonyms: false,
            learningRate: typeof config.learningRate === "number" ? config.learningRate : 0.1,
            epochs: typeof config.epochs === "number" ? config.epochs : 50
        }
    };
}