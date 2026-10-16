async function getSynonyms(word) {
    const url = new URL("https://api.datamuse.com/words");
    url.searchParams.set("rel_syn", word);
    url.searchParams.set("max", "20");

    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Datamuse API error: ${response.status}`);
    }

    const results = await response.json();
    return results.map(result => result.word);
}

async function map_synonyms(token, vocabulary) {
    const synonyms = await getSynonyms(token);

    for (const word of synonyms) {
        if (vocabulary.includes(word)) {
            return word;
        }
    }

    return null;
}
