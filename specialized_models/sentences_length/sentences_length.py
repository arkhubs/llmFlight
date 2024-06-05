import numpy as np
import re
from typing import List
from collections import Counter

# 将文本分割成句子列表
def split_into_sentences(text: str) -> List[str]:
    # 这里使用简单的方法，根据句号、问号和感叹号分割句子
    text1 = text.replace('?', '[*END*]').replace('!', '[*END*]').replace('？', '[*END*]').replace('！', '[*END*]').replace('。', '[*END*]').replace('\n', '[*END*]').replace('.', '[*END*]')
    text2 = text1.replace(',', '[*END*]').replace('，', '[*END*]')
    text1 = re.sub(r'\[\*END\*\]{2,}', '[*END*]', text1)
    text2 = re.sub(r'\[\*END\*\]{2,}', '[*END*]', text2)
    texts1 = text1.split('[*END*]')
    texts2 = text2.split('[*END*]')
    return texts1, texts2


# 计算句子的平均长度和变异系数
def calculate_sentence_metrics(sentences):
    lengths = [len(sentence) for sentence in sentences if sentence]  # 计算每个句子的长度
    mean_length = np.mean(lengths) if lengths else 0  # 计算平均长度
    std_length = np.std(lengths) if lengths else 0  # 计算标准差
    cvl = std_length / mean_length if mean_length else 0  # 计算变异系数

    return mean_length, cvl


# 计算句子结构综合指标 (CSV)
def calculate_csv(mean_length: float, cvl: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    return alpha * mean_length + beta * cvl

def main(text):

    # 分割文本为句子
    sentences1, sentences2 = split_into_sentences(text)

    # 计算句子的平均长度和变异系数
    mean_length1, cvl = calculate_sentence_metrics(sentences1)
    mean_length2, cv2 = calculate_sentence_metrics(sentences2)

    return mean_length1, cvl, mean_length2, cv2
