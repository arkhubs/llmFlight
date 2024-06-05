import numpy as np
import jieba
from sklearn.decomposition import PCA
from collections import Counter


def calculate_yules_k(freqs, N):
    # 计算Yule's K特征
    sum_i_squared_v_i = sum(frequency * (freq ** 2) for freq, frequency in freqs.items())
    yules_k = 10**4 * (sum_i_squared_v_i - N) / (N ** 2)
    return yules_k

def calculate_diversity_d(word_counts, N):
    # 计算频次的平方和
    sum_fi_squared = sum(freq ** 2 for freq in word_counts.values())
    
    # 计算D指数
    diversity_d = N / sum_fi_squared if sum_fi_squared > 0 else 0
    return diversity_d

def main(text):
    # 将文本分割成单词列表，这里简单地使用空格进行分割
    words = jieba.lcut(text)
    # 计算f_i，即每个单词的出现次数
    word_counts = Counter(words)
    # 计算V(i)，即出现次数为i的单词数量
    freqs = Counter(word_counts.values())

    # 计算标记（Token）的数量
    N = len(words)
    # 计算类型（Type）的数量
    V = len(set(words))

    ttr = V / N if N > 0 else 0
    yules_k = calculate_yules_k(freqs, N)
    d_index = calculate_diversity_d(word_counts, N)
    return ttr, yules_k, d_index
