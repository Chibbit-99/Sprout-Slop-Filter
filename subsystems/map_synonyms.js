// Look up synonyms for a token via the Datamuse API, with a simple in-memory
// cache so the same word is not repeatedly requested.
//
// Expected usage: this module is loaded once per page. `getSynonyms` is the
// low-level call used by `get_synonyms_cached`; `map_synonyms` is the higher
// level helper that connects an unknown token to a known vocabulary word.
//
// API failures, rate limits, or unavailable network access should not crash
// the rest of the classifier: every helper falls back to an empty result so
// `classify_text` can still produce a prediction.

const DATAMUSE_ENDPOINT = "https://api.datamuse.com/words";
const DATAMUSE_MAX = 20;

const synonymCache = new Map();

async function getSynonyms(word) {
    const url = new URL(DATAMUSE_ENDPOINT);
    url.searchParams.set("rel_syn", word);
    url.searchParams.set("max", String(DATAMUSE_MAX));

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Datamuse API error: ${response.status}`);
    }

    const results = await response.json();
    return results.map(result => result.word);
}

// `get_synonyms_cached` is the cache-aware wrapper described in the
// subsystems doc. It returns a list of synonyms, or an empty list if the
// lookup fails or produces nothing useful.
async function get_synonyms_cached(word) {
    if (typeof word !== "string" || word.length === 0) {
        return [];
    }

    if (synonymCache.has(word)) {
        return synonymCache.get(word);
    }

    try {
        const synonyms = await getSynonyms(word);
        synonymCache.set(word, synonyms);
        return synonyms;
    } catch (error) {
        // Cache the failure so we don't keep retrying a failing request.
        synonymCache.set(word, []);
        return [];
    }
}

// `map_synonyms(token, vocabulary)` records the relationship between an
// unknown token and any synonym that is already known. The result is an
// array of "word links" — one entry per known synonym match. Returning an
// array (rather than a single word) keeps the link information flexible
// for downstream consumers like `get_word_vector`.
//
// Synonyms are clues, not replacements: a single token can map to several
// known words, and a known word may carry a different sense of the
// original token.
async function map_synonyms(token, vocabulary) {
    if (typeof token !== "string" || token.length === 0) {
        return [];
    }

    if (!Array.isArray(vocabulary)) {
        return [];
    }

    const knownSet = new Set(vocabulary);
    if (knownSet.has(token)) {
        // The token is already known — there is nothing to map.
        return [];
    }

    const synonyms = await get_synonyms_cached(token);
    const links = [];

    for (const synonym of synonyms) {
        if (knownSet.has(synonym)) {
            links.push({ token, knownWord: synonym });
        }
    }

    return links;
}

// Exposed mainly for tests that need a clean slate. Production code does
// not need to call this.
function clear_synonym_cache() {
    synonymCache.clear();
}