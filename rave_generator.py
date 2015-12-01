import numpy as np
from collections import Counter


def peek_random(cumsum, keys=None):
    import random
    peek_value = random.randint(1, cumsum[-1])
    idx = np.searchsorted(cumsum, peek_value, side='left')
    if keys is None:
        return idx
    else:
        return keys[idx]


class TrigramMarkovChain(object):

    END_OF_SENTENCE = ['.', '!', '?', '...']

    def __init__(self):
        self.unary = {}
        self.binary = []
        # two structures bellow are used to decrease memory usage
        self.idx2word = []
        self.word2idx = {}
        # used for generation
        self.start_cumsum = None
        self.unary_cumsum = None
        self.binary_cumsum = None
        self.generate_ready = False

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
            self.binary.append({})
            return idx
        return self.word2idx[word]

    def push_word_word(self, word, follows):
        word_idx = self._add_word(word)
        follows_idx = self._add_word(follows)

        self.unary.setdefault(word_idx, Counter())[follows_idx] += 1
        self.generate_ready = False

    def push_pair_word(self, pair, follows):
        first_word, second_word = pair
        first_word_idx = self._add_word(first_word)
        second_word_idx = self._add_word(second_word)
        follows_idx = self._add_word(follows)

        self.binary[first_word_idx].setdefault(
            second_word_idx, Counter())[follows_idx] += 1
        self.generate_ready = False

    def push(self, trigram):
        if len(trigram) == 2:
            self.push_word_word(trigram[0], trigram[-1])
        elif len(trigram) == 3:
            self.push_pair_word(trigram[:-1], trigram[-1])

    def generate_prepare(self):
        self.unary_cumsum = {key: np.cumsum(wordstat.values()) for
                             (key, wordstat) in self.unary.iteritems()}
        self.unary_keys = {key: wordstat.keys() for
                           (key, wordstat) in self.unary.iteritems()}

        self.binary_cumsum = [{second: np.cumsum(pairstat.values()) for
                               (second, pairstat) in firststat.iteritems()} for
                              firststat in self.binary]
        self.binary_keys = [{second: pairstat.keys() for
                             (second, pairstat) in firststat.iteritems()} for
                            firststat in self.binary]

        self.start_cumsum = np.cumsum([cumsum[-1] for cumsum
                                       in self.unary_cumsum.values()])
        self.start_keys = self.unary_cumsum.keys()
        self.generate_ready = True

    def sentence(self, startword_idx=None):
        if not self.generate_ready:
            self.generate_prepare()

        def upper_first(word):
            return word[0].upper() + word[1:]

        if startword_idx is None:
            startword_idx = peek_random(self.start_cumsum, self.start_keys)

        yield upper_first(self.idx2word[startword_idx])

        follows_idx = peek_random(self.unary_cumsum[startword_idx],
                                  self.unary_keys[startword_idx])
        first_word_idx, second_word_idx = startword_idx, follows_idx

        while self.idx2word[follows_idx] not in self.END_OF_SENTENCE:
            yield self.idx2word[follows_idx]
            follows_idx = peek_random(
                self.binary_cumsum[first_word_idx][second_word_idx],
                self.binary_keys[first_word_idx][second_word_idx])
            first_word_idx, second_word_idx = second_word_idx, follows_idx


def trigram_split(filename):
    import re
    import itertools
    import codecs

    single_quotes = re.compile(u"[\u2019\u2018']")
    double_quotes = re.compile(u'[\u201C\u201E\u201d"]')
    end_of_sentence = re.compile(u'[!?.\u2026]')
    trash = re.compile(r'["#$%&(),*+/:;\-<=>@^_`{|}~\[\]]' +  # punctuation
                       r"|('+\B)" +  # quotes at the end of the word
                       u'|[\u2014\u2022\u2013\xa9\xb0]')  # strange symbols
    active_voice_quotes = re.compile(r"(^')|(\s')|('$)")
    number = re.compile('[0-9]+')
    multispace = re.compile(r'[\s]+')

    def lower_tail(match):
        pattern = match.group()
        return pattern[0] + pattern[1:].lower()
    strange_caps = re.compile(r'\w+[A-Z]\w*\b')

    preprocess_conveyor = [
        lambda s: single_quotes.sub(u"'", s),
        lambda s: double_quotes.sub(u'"', s),
        lambda s: number.sub(u'42', s),
        lambda s: end_of_sentence.sub(r'.', s),
        lambda s: active_voice_quotes.sub(' ', s),
        lambda s: trash.sub(u' ', s),
        lambda s: multispace.sub(u' ', s),
        lambda s: strange_caps.sub(lower_tail, s)
    ]

    with codecs.open(filename, encoding='utf-8', mode='r') as fi:
        for line in fi:
            processed_line = reduce(
                lambda x, f: f(x), preprocess_conveyor, line)
            for sentence in processed_line.split('.'):
                words = sentence.split() + ['.']
                if len(words) > 1:
                    yield words[0], words[1]
                for trigram in itertools.izip(words, words[1:], words[2:]):
                    yield trigram


def ls_recursive(path):
    import os
    if os.path.isdir(path):
        return sum([ls_recursive(path + '/' + file)
                    for file in os.listdir(path)], [])
    else:
        return [path]


def dprint(*args):
    import sys
    sys.stdout.write('\r{}'.format(' '.join(map(str, args))))
    sys.stdout.flush()


def gather_model(corpus_path='./corpus'):
    corpus_files = ls_recursive(corpus_path)
    model = TrigramMarkovChain()
    for idx, filename in enumerate(corpus_files):
        dprint('Reading ({}/{}):'.format(idx + 1, len(corpus_files)), filename)
        for trigram in trigram_split(filename):
            model.push(trigram)
    return model


def dump_model(filename, model):
    import cPickle as pickle
    if not model.generate_ready:
        model.generate_prepare()
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    import cPickle as pickle
    return pickle.load(open(filename, 'rb'))


def sentence(model):
    return u'{}.'.format(u' '.join(model.sentence()))


def split_paragraph(paragraph, maxwidth=85):
    lines = [[]]
    line_width = 5  # due to first tab of the paragraph
    for word in paragraph.split():
        if line_width + len(word) < maxwidth:
            lines[-1] += [word]
            line_width += len(word) + 1
        else:
            lines.append([word])
            line_width = len(word)
    return u'\n'.join([u' '.join(line) for line in lines])


def generate_text(filename, model, words_to_generate=10000):
    import random
    import codecs

    word_count = 0
    with codecs.open(filename, mode='w', encoding='utf-8') as file:
        while word_count < words_to_generate:
            paragrarh_len = random.randint(5, 15)
            paragrarh = u' '.\
                join((sentence(model) for x in xrange(paragrarh_len)))
            word_count += len(paragrarh.split())
            file.write(u'\t{}\n'.format(split_paragraph(paragrarh)))
            dprint('word count:', word_count)


if __name__ == '__main__':
    pass
