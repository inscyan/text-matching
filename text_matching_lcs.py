import numpy as np


def find_lcsubstr(s1, s2):
    """
    求两个字符串最长公共子串 https://blog.csdn.net/wateryouyo/article/details/50917812
    :param s1: abcdfg
    :param s2: abdfg
    :return:
    """
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1

    return s1[p - mmax:p], mmax  # 返回最长子串及其长度


def text_matching_lcs(text, candidate_texts, top_n=1):
    """
    文本匹配：基于最长公共子串
    :param text:
    :param candidate_texts:
    :param top_n:
    :return:
    """
    text_len = len(text)

    lcs_score = []
    for item in candidate_texts:
        common_str, common_length = find_lcsubstr(text, item)
        if common_length == 0:
            lcs_score.append(0)
        else:
            lcs_score.append((common_length / text_len) * (common_length / len(item)))

    sims_argsort = (-np.array(lcs_score)).argsort()[:top_n]

    return [(candidate_texts[idx], lcs_score[idx]) for idx in sims_argsort]


if __name__ == '__main__':
    print(text_matching_lcs('我爱北京天安门', ['天安门', '我爱天安门', '西湖好地方']))
