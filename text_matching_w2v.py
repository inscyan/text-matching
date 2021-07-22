import jieba

import numpy as np


def text_matching_w2v(text, candidate_texts, top_n=1):
    """
    文本匹配：基于词向量
    :param text:
    :param candidate_texts:
    :param top_n:
    :return:
    """
    text_cut = jieba.lcut(text)
    candidate_texts_cut = [jieba.lcut(item) for item in candidate_texts]
    # todo: 可以选择去一下停用词

    text_cut_vec = []
    for word in text_cut:
        try:
            text_cut_vec.append(w2v_dict[word])
        except:
            pass
    text_cut_vec_ave = np.sum(text_cut_vec, axis=0) / len(text_cut_vec)

    candidate_texts_cut_vec_ave = []
    for sen in candidate_texts_cut:
        sen_vec_temp = []
        for word in sen:
            try:
                sen_vec_temp.append(w2v_dict[word])
            except:
                pass
        candidate_texts_cut_vec_ave.append(np.sum(sen_vec_temp, axis=0) / len(sen_vec_temp))

    cosine_similarities = []
    for vec_ave in candidate_texts_cut_vec_ave:
        temp_score = np.dot(text_cut_vec_ave, vec_ave) / (np.linalg.norm(text_cut_vec_ave) * np.linalg.norm(vec_ave))
        cosine_similarities.append(temp_score)

    sims_argsort = (-np.array(cosine_similarities)).argsort()[:top_n]

    return [(candidate_texts[idx], cosine_similarities[idx]) for idx in sims_argsort]


if __name__ == '__main__':
    # 加载词向量（词向量预训练文件网上很多，可优先寻找与所做任务领域对应的）
    with open('resource/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', encoding='utf8') as f:
        lines = f.readlines()[1:]
    w2v_dict = {}
    for line in lines:
        temp = line.strip().split()
        w2v_dict[temp[0]] = list(map(float, temp[1:]))

    print(text_matching_w2v('我爱北京天安门', ['天安门', '我爱天安门', '西湖好地方']))
    print(text_matching_w2v('我爱北京天安门', ['天安门', '我爱天安门', '西湖好地方'], top_n=2))
    print(text_matching_w2v('我爱北京天安门', ['天安门', '我爱天安门', '西湖好地方'], top_n=3))
    print(text_matching_w2v('我爱北京天安门', ['天安门', '我爱天安门', '西湖好地方'], top_n=4))
