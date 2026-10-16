// Split text into word and punctuation tokens.
// Punctuation marks are kept as individual tokens, so "!!!" becomes ["!", "!", "!"].
// Apostrophes inside words are preserved, so "don't" stays as one token.
//
// Expected input: text, preferably already processed by normalise_text().

function tokenise(text) {
    if (typeof text !== "string") {
        throw new TypeError("tokenise() expects a string");
    }

    // Match words (including numbers and internal apostrophes), or any
    // individual non-whitespace character as its own token.
    const tokens = text.match(/[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}\p{N}]+)*|[^\s]/gu);

    // String.match() returns null when there are no matches.
    return tokens ?? [];
}
