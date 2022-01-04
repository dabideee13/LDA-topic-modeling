from typing import Optional, Any

import spacy
from nltk.corpus import stopwords

import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import Phrases, TfidfModel
from gensim.models.phrases import Phraser

from tools import pipe

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stop_words = stopwords.words("english")


def lemmatize(texts: list[str]) -> list[str]:
    def _lemmatize(text: str, allowed_postags: Optional[list[str]] = None) -> str:
        if allowed_postags is None:
            allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]

        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return [_lemmatize(text) for text in texts]


def gensim_preprocess(lemmatized_texts: list[str]) -> list[list[str]]:
    def _gensim_preprocess(text: str) -> list[str]:
        return simple_preprocess(text, deacc=True)

    return [_gensim_preprocess(text) for text in lemmatized_texts]


def remove_stopwords(texts: list[list[str]]) -> list[list[str]]:
    def _remove_stopwords(words: list[str]) -> list[str]:
        return [word for word in words if word not in stop_words]

    return [_remove_stopwords(words) for words in texts]


def generate_ngrams(words: list[list[str]]):
    bigram_phrases = Phrases(words, min_count=5, threshold=50)
    trigram_phrases = Phrases(bigram_phrases[words], threshold=50)

    bigram = Phraser(bigram_phrases)
    trigram = Phraser(trigram_phrases)

    def _make_bigrams(words: list[list[str]]):
        return (bigram[doc] for doc in words)

    def _make_trigrams(words: list[list[str]]):
        return (trigram[bigram[doc]] for doc in words)

    data_bigrams = _make_bigrams(words)
    data_bigrams_trigrams = _make_trigrams(data_bigrams)

    return list(data_bigrams_trigrams)


def make_corpus(words: list[list[str]]) -> tuple[list[tuple[int, int]], corpora.dictionary.Dictionary]:
    id2word = corpora.Dictionary(words)
    return [id2word.doc2bow(text) for text in words], id2word


def apply_tfidf(corpus: list[tuple[int, int]], id2word: corpora.dictionary.Dictionary) -> list[tuple[int, int]]:
    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value = 0.03
    words = []
    words_missing_in_tfidf = []

    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []

        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfidf

        for item in drops:
            words.append(id2word[item])

        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    return corpus, id2word


def preprocess_text(texts: list[str]) -> Any:
    return pipe(
        texts,
        lemmatize,
        gensim_preprocess,
        remove_stopwords,
        generate_ngrams,
        make_corpus,
    )