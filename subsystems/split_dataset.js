// Split a labelled example list into training, validation, and test sets.
//
// Splits are deterministic when `seed` is provided: a small mulberry32 PRNG
// is used to shuffle the examples before partitioning. The defaults are
// 70% / 15% / 15%.
//
// The test set must never be used to adjust the model during training; the
// caller is responsible for keeping that boundary.

function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function shuffle(array, random) {
    const copy = array.slice();

    for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const tmp = copy[i];
        copy[i] = copy[j];
        copy[j] = tmp;
    }

    return copy;
}

function split_dataset(examples, options) {
    if (!Array.isArray(examples)) {
        throw new TypeError("split_dataset: examples must be an array");
    }

    const settings = options || {};
    const trainRatio = typeof settings.trainRatio === "number" ? settings.trainRatio : 0.7;
    const validationRatio = typeof settings.validationRatio === "number" ? settings.validationRatio : 0.15;
    const seed = typeof settings.seed === "number" ? settings.seed : 1;

    if (trainRatio <= 0 || validationRatio < 0 || trainRatio + validationRatio >= 1) {
        throw new RangeError("split_dataset: invalid ratio combination");
    }

    const random = mulberry32(seed);
    const shuffled = shuffle(examples, random);

    const trainEnd = Math.floor(shuffled.length * trainRatio);
    const validationEnd = trainEnd + Math.floor(shuffled.length * validationRatio);

    return {
        trainSet: shuffled.slice(0, trainEnd),
        validationSet: shuffled.slice(trainEnd, validationEnd),
        testSet: shuffled.slice(validationEnd)
    };
}