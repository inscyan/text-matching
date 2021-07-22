import jieba

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities.docsim import SparseMatrixSimilarity


def text_matching_tfidf(text, candidate_texts, top_n=1):
    """
    文本匹配：基于TF-IDF
    :param text:
    :param candidate_texts:
    :param top_n:
    :return:
    """
    text_cut = jieba.lcut(text)
    candidate_texts_cut = [jieba.lcut(item) for item in candidate_texts]
    # todo: 可以选择去一下停用词

    dct = Dictionary(candidate_texts_cut)
    dct_size = len(dct.token2id.keys())
    corpus_bow = [dct.doc2bow(item) for item in candidate_texts_cut]

    tfidf_model = TfidfModel(corpus_bow, dictionary=dct)
    corpus_tfidf = tfidf_model[corpus_bow]

    similarity = SparseMatrixSimilarity(corpus_tfidf, num_features=dct_size)

    text_bow = dct.doc2bow(text_cut)
    text_tfidf = tfidf_model[text_bow]
    cosine_similarities = similarity[text_tfidf]
    sims_argsort = (-cosine_similarities).argsort()[:top_n]

    return [(candidate_texts[idx], cosine_similarities[idx]) for idx in sims_argsort]


if __name__ == '__main__':
    print(text_matching_tfidf('我爱北京天安门', ['天安门', '我爱天安门', '西湖好地方']))
