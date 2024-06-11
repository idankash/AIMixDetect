import re


class Sentence(object):
    def __init__(self, text):
        self.text = text
        self.tokens = text.split()

    def __len__(self):
        return len(self.tokens)

class Sentences(object):
    def __init__(self, text):
        def iterate(text):
            for s in re.split(r"\n", text):
                yield s
        self.sents = iterate(text)

    def __len__(self):
        return len(self.sents)

class SentenceParser(object):
    """
    Iterate over the text column of a dataframe
    """

    def __init__(self):
        self.sents = None

    def __call__(self, text):
        return Sentences(text)