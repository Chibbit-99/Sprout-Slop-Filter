function normalise_text(text){
    var output = text;
    output = output.toLowerCase(); //Convert to lowercase
    output = output.trim(); //Remove surrounding whitespace
    output = output.replace(/\s+/g, ' '); //Convert long whitespace to single space
    return output;
};
