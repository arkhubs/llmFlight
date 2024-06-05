import re
from snownlp import SnowNLP

def process_text(text):
    # 定义匹配句子的正则表达式，包括中英文句号、问号、感叹号、省略号、换行符和制表符
    sentence_endings = re.compile(r'[。！？…\.\?\!\n\t]+')
    
    # 按句子分隔符拆分段落成句子
    sentences = sentence_endings.split(text)
    
    # 去除空白句子并合并连续的分隔符
    processed_sentences = []
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            processed_sentences.append(stripped_sentence)

    return processed_sentences


def get_polar(sentences):
    """
    :param sentences: 分好句的句子列表
    :return: 每一句所对应的情感极性值的列表，情感极性取值0-1，
    越接近0代表越负面，越接近1代表越正面
    """
    text_polar = []
    for sentence in sentences:
        score = SnowNLP(sentence)
        text_polar.append(score.sentiments)

    return text_polar


def get_eps(sen):
    """
    :param text: 一整段输入的中文文本段落
    :return: 这段文本段落的EPS值
    """
    # 计算文本中每一句的情感极性值
    polar = get_polar(sen)
    # 计算EPS值
    eps = 0
    for i in range(1, len(polar)):
        temp = abs(polar[i] - polar[i - 1])
        eps += temp
    eps = eps / len(polar) - 1 if len(polar) > 1 else 0
    return eps


def get_intensity(sentences):
    """
    :param sentences: 分好句的句子列表
    :return: 每一句所对应的情感强度的列表，其中情感强度 = |情感极值 - 0.5|
    """
    intensity = []
    for sentence in sentences:
        score = SnowNLP(sentence)
        intens = abs(score.sentiments - 0.5)
        intensity.append(intens)
    return intensity


def get_intensity_mean(intensity):
    """
    :param intensity: 每一句所对应的情感强度的列表
    :return: 这一段话情感强度的均值
    """
    sum = 0
    for intens in intensity:
        sum += intens
    sum = sum / len(intensity) if len(intensity) > 0 else 0
    return sum


def get_eiv(sen):
    """
    :param text: 一整段输入的中文文本段落
    :return: 这段文本段落的EIV值
    """
    intensity = get_intensity(sen)
    intensity_mean = get_intensity_mean(intensity)
    eiv = 0
    for intens in intensity:
        temp = (intens - intensity_mean) ** 2
        eiv += temp
    eiv = eiv / len(intensity) if len(intensity) > 0 else 0
    return eiv ** (1/2)


def main(text):
    sentences = process_text(text)
    return get_eps(sentences), get_eiv(sentences)