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
    // Keep implementation constants local: subsystem files are loaded as
    // classic scripts, so top-level `const` names share one global scope.
    const UNKNOWN_TOKEN = "<UNK>";
    const config = settings || {};
    if (config.encoder && !["richer-subword-v4", "structured-subword-v3", "hashed-subword-v2", "legacy"].includes(config.encoder)) {
        throw new RangeError("initialise_model: unsupported encoder " + config.encoder);
    }
    const vectorSize = typeof config.vectorSize === "number" && config.vectorSize > 0
        ? Math.floor(config.vectorSize)
        : 8;
    const random = typeof config.random === "function"
        ? config.random
        : makeRandom(typeof config.seed === "number" ? config.seed : undefined);

    if (["richer-subword-v4", "structured-subword-v3", "hashed-subword-v2"].includes(config.encoder)) {
        const v4 = config.encoder === "richer-subword-v4";
        const v3 = config.encoder === "structured-subword-v3";
        if (v4 && config.featureSize !== undefined && config.featureSize !== 524288) {
            throw new RangeError("initialise_model: v4 requires 524288 features (eight namespaces)");
        }
        if (v3 && config.featureSize !== undefined && config.featureSize !== 131072) {
            throw new RangeError("initialise_model: v3 requires 131072 features (four namespaces)");
        }
        const featureSize = v4 ? 524288 : v3 ? 131072 :
            Number.isInteger(config.featureSize) && config.featureSize >= 1024
                && config.featureSize <= 262144 ? config.featureSize : 32768;
        return {
            version: v4 ? 4 : v3 ? 3 : 2,
            featureSize,
            vocabulary: ["<UNK>", ...(config.vocabulary || []).filter(word => word !== "<UNK>")],
            weights: new Array(featureSize).fill(0),
            accumulators: new Array(featureSize).fill(0),
            bias: 0,
            options: { encoder: config.encoder, useSynonyms: false,
                learningRate: config.learningRate ?? 0.8, epochs: config.epochs ?? 120 }
        };
    }

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