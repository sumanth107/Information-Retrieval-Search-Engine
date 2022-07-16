# Add your import statements here
from spellchecker import SpellChecker
from nltk.corpus import wordnet


class sc:

    def querys(self, query):

        words = SpellChecker()
        query = query.split()
        corrected_query = ""
        misspelled_words = words.unknown(query)

        for i, word in enumerate(query):
            if word not in misspelled_words:
                corrected_query += word
            else:
                corrected_query += words.correction(word)
            if i != len(query) - 1:
                corrected_query += " "

        return corrected_query

    def docss(self, doc):
        words = SpellChecker()
        doc = doc.split()
        corrected_doc = ""
        misspelled_words = words.unknown(doc)

        for i, word in enumerate(doc):
            if word not in misspelled_words:
                corrected_doc += word
            else:
                corrected_doc += words.correction(word)

        return corrected_doc


class q_exp:

    def expand(self, txt, x):
        expanded_query = []
        for s in txt:
            tmp = []
            for w in s:
                tmp.append(w)
                c = 0
                for syn in wordnet.synsets(w):
                    for l in syn.lemmas():
                        if c < x:
                            if l.name() not in tmp:
                                tmp.append(l.name())
                                c += 1
            expanded_query.append(tmp)
        return expanded_query

# Add any utility functions here
