import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle, os, sys

sys.path.append(os.path.dirname(__file__))
from words_richness import main

class Model():
    def __init__(self, pkl='../specialized_models\words_richness\words_richness_202406051735.pkl'):

        # 加载模型参数
        with open(pkl, 'rb') as f:
            params = pickle.load(f)
        self.params = params
        
    def infer(self, text):
        ttr, yules_k, d_index = main(text)
        df_predict = pd.DataFrame(dict(ttr=[ttr], log_yules_k=np.log([yules_k]), d_index=[d_index]))
        df_predict['intercept'] = 1.0

        # 定义自变量
        X_new = df_predict[['intercept', 'ttr', 'log_yules_k', 'd_index']]

        # 进行预测
        model = sm.Logit([0] * len(X_new), X_new)  # 这里的因变量无关紧要，只是为了创建模型实例
        model.initialize()  # 初始化模型
        prediction = model.predict(self.params)
        return prediction