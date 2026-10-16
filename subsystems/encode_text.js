// Combine a list of word vectors into a single representation of the whole
// string.
//
// The first version averages every word vector. Averaging is order
// insensitive, so "buy milk" and "milk buy" encode to the same
// representation. That keeps the model small and easy to reason about;
// richer combination methods can be explored later.

function encode_text(vectors) {
    if (!Array.isArray(vectors)) {
        throw new TypeError("encode_text: expected an array of vectors");
    }

    if (vectors.length === 0) {
        return [];
    }

    const dimension = vectors[0].length;
    const sum = new Array(dimension).fill(0);

    for (const vector of vectors) {
        if (!Array.isArray(vector)) continue;

        for (let i = 0; i < dimension; i++) {
            sum[i] += vector[i];
        }
    }

    return sum.map(value => value / vectors.length);
}