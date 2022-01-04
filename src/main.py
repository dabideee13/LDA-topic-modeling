import pandas as pd

from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from tools import get_data_path
from preprocessing import apply_tfidf, preprocess_text


def main():

    # Import data
    file = "keychron_K2_reviews.csv"
    data = pd.read_csv(get_data_path(file))["content"].values.tolist()[:-1]

    # Preprocessing
    corpus, id2word = preprocess_text(data)

    # TF-IDF
    corpus, id2word = apply_tfidf(corpus, id2word)

    # Model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=10,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto"
    )

    # Visualization
    vis = gensimvis.prepare(
        lda_model,
        corpus,
        id2word,
        mds="mmds",
        R=30
    )

    # Export figure as html
    pyLDAvis.save_html(vis, str(get_data_path("lda_keychron_reviews.html")))


if __name__ == "__main__":
    main()
