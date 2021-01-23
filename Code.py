import math
import os
import nltk
import sys
from heapq import nlargest
import string
from collections import Counter

# nltk.download('stopwords')
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return's a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), encoding="utf8") as f:
            text = f.read()
            files[filename] = text
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return's a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removes any
    punctuation or English stopwords.
    """

    def is_number(n):
        try:
            float(n)  # Type-casting the string to `float`.
            # If string is not a valid `float`,
            # it'll raise `ValueError` exception
        except ValueError:
            return False
        return True

    contents = []
    for word in nltk.word_tokenize(document):
        if word.isalpha():
            if not word.lower() in nltk.corpus.stopwords.words("english"):
                contents.append(word.translate(str.maketrans('', '', string.punctuation)).lower())
        elif is_number(word):
            contents.append(word)
    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return's a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents will be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents:
        words.update(documents[filename])

    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return's a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_freq = dict()
    for a in files:
        frequencies = dict()
        for word in files[a]:
            if word not in frequencies:
                frequencies[word] = 1
            else:
                frequencies[word] += 1
        file_freq[a] = frequencies
    tfidfs = dict()
    for filename in files:
        tfidfs[filename] = []
        for word in files[filename]:
            tf = file_freq[filename][word]
            tfidfs[filename].append((word, tf * idfs[word]))
    op = dict()
    for filename in tfidfs:
        op[filename] = []
        for j in tfidfs[filename]:
            if j[0] in query and j not in op[filename]:
                op[filename].append(j)

    sum_tfidf = dict()

    for f in op:
        sum_tfidf[f] = sum([i[1] for i in op[f]])
    # temp = Counter(sum_tfidf)
    # print('most_common', temp.most_common(n))
    res = nlargest(n, sum_tfidf, key=sum_tfidf.get)
    return res


def qtd(similar, query):
    if len(similar) == 1:
        return similar
    else:
        qtd_val = dict()
        for i in similar:
            temp = tokenize(i)
            match = sum(word in query for word in temp)
            qtd_val[i] = match / len(temp)
    # print(qtd_val)
    return nlargest(len(similar), qtd_val, key=qtd_val.get)


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return's a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference will
    be given to sentences that have a higher query term density.
    """
    value = dict()
    # print(query)
    for sent in sentences:
        temp = 0
        for word in query:
            if word in tokenize(sent) and word in idfs:  # and word not in done:
                temp = temp + idfs[word]
        value[sent] = temp
    res = nlargest(len(sentences), value, key=value.get)

    result = []
    val = next(iter(res))
    similar = []
    for i in res:
        if value[i] == val:
            similar.append(i)
            val = value[i]
            continue
        for j in qtd(similar, query):
            result.append(j)
        similar = [i]
        val = value[i]
    return result[0:n]


if __name__ == "__main__":
    main()
