import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from snownlp import SnowNLP
import re
import re

def process_text(text):
    # 定义匹配句子的正则表达式，包括中英文句号、问号、感叹号、省略号、换行符和制表符
    sentence_endings = re.compile(r'[。！？…\.\?\!\n\t]+')
    
    # 按句子分隔符拆分段落成句子
    sentences = sentence_endings.split(text)
    
    # 去除空白句子并合并长度小于10的句子与上一句
    processed_sentences = []
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            if processed_sentences and len(stripped_sentence) < 10:
                # 如果上一个句子存在且当前句子长度小于10，则与上一个句子合并
                processed_sentences[-1] += stripped_sentence
            else:
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

def fit_arma(time_series, order=(2, 0, 2)):
    """
    拟合ARMA模型并评估模型拟合效果

    参数：
    - time_series: 输入的时间序列数据（numpy数组或pandas Series）
    - order: ARIMA模型的参数，默认为(2, 0, 2)，可以用来模拟ARMA(p, q)
    注：(2, 0, 2)中0表示用的是ARMA模型，第一和第三个数字分别表示p和q

    返回：
    - model_fit: 拟合的模型
    - mse: 均方误差
    - r_squared: 决定系数
    """
    # 拟合ARMA模型
    model = ARIMA(time_series, order=order)
    model_fit = model.fit()
    print("1")
    # 打印模型总结
    print(model_fit.summary())

    # 进行预测
    predictions = model_fit.predict(start=0, end=len(time_series) - 1)

    # 评估模型拟合效果
    mse = np.mean((time_series - predictions) ** 2)
    r_squared = 1 - np.sum((time_series - predictions) ** 2) / np.sum((time_series - np.mean(time_series)) ** 2)

    # 绘制结果
    plt.plot(time_series, label='Original Data')
    plt.plot(predictions, label='Fitted Values', color='red')
    plt.legend()
    plt.title('ARMA Model Fit')
    plt.show()

    return model_fit, mse, r_squared


# 调用函数
def main(text):
    sentences = process_text(text)
    polars = get_polar(sentences)
    cum_polars = np.cumsum(polars)
    model_fit, mse, r_squared = fit_arma(cum_polars, order=(2, 0, 2))
    print(model_fit, mse, r_squared)
