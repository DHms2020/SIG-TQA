import abc
import functools
import os
import time

import bpemb
import corenlp
import torch
import torchtext
import stanza

from seq2struct.resources import corenlp
from seq2struct.utils import registry

_stanza_pipeline=None

#为了替换解决corenlp依赖java的JDK版本不对问题
def get_stanza_pipeline() -> stanza.Pipeline:
    global _stanza_pipeline
    if _stanza_pipeline is None:
        _stanza_pipeline = stanza.Pipeline(processors="tokenize,mwt,lemma")
    return _stanza_pipeline

pipeline = get_stanza_pipeline()

class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, sentence):
        '''Given a string, return a list of tokens suitable for lookup.'''
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        '''Undo tokenize.'''
        pass

    @abc.abstractmethod
    def lookup(self, token):
        '''Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None.'''
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Transfer the pretrained embeddings to the given device.'''
        pass


@registry.register('word_emb', 'glove')
class GloVe(Embedder):

    def __init__(self, kind, lemmatize=False):
        cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors
        self.lemmatize = lemmatize
        self.corenlp_annotators = ['tokenize', 'ssplit']
        if lemmatize:
            self.corenlp_annotators.append('lemma')

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        # ann = corenlp.annotate(text, self.corenlp_annotators)
        doc=pipeline(text)
        if self.lemmatize:
            # return [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            return [word.lemma.lower() for sentence in doc.sentences for word in sentence.words]
        else:
            # return [tok.word.lower() for sent in ann.sentence for tok in sent.token]
            return [word.text.lower() for sentence in doc.sentences for word in sentence.words]

    
    @functools.lru_cache(maxsize=1024)
    def tokenize_for_copying(self, text):
        # ann = corenlp.annotate(text, self.corenlp_annotators)
        doc=pipeline(text)
        # text_for_copying = [tok.originalText.lower() for sent in ann.sentence for tok in sent.token]
        text_for_copying = [tok.text.lower() for sentence in doc.sentences for tok in sentence.words]
        if self.lemmatize:
            # text = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            text = [tok.lemma.lower() for sentence in doc.sentences for tok in sentence.words]
        else:
            # text = [tok.word.lower() for sent in ann.sentence for tok in sent.token]
            text = [tok.text.lower() for sentence in doc.sentence for tok in sentence.words]
        return text, text_for_copying

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        i = self.glove.stoi.get(token)
        if i is None:
            return None
        return self.vectors[i]

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)


@registry.register('word_emb', 'bpemb')
class BPEmb(Embedder):
    def __init__(self, dim, vocab_size, lang='en'):
        self.bpemb = bpemb.BPEmb(lang=lang, dim=dim, vs=vocab_size)
        self.dim = dim
        self.vectors = torch.from_numpy(self.bpemb.vectors)

    def tokenize(self, text):
        return self.bpemb.encode(text)

    def untokenize(self, tokens):
        return self.bpemb.decode(tokens)

    def lookup(self, token):
        i = self.bpemb.spm.PieceToId(token)
        if i == self.bpemb.spm.unk_id():
            return None
        return self.vectors[i]

    def contains(self, token):
        return self.lookup(token) is not None

    def to(self, device):
        self.vectors = self.vectors.to(device)
