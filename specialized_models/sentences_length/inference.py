import statsmodels.api as sm
import numpy as np
import pandas as pd
import pickle, os, sys

sys.path.append(os.path.dirname(__file__))
from sentences_length import main

class Model():
    def __init__(self, pkl='../specialized_models/sentences_length/sentences_length_202406051750.pkl'):

        # 加载模型参数
        with open(pkl, 'rb') as f:
            params = pickle.load(f)
        self.params = params
        
    def infer(self, text):
        mean1, cv1, mean2, cv2 = main(text)
        df_predict = pd.DataFrame(dict(logmean1=np.log([mean1]), cv1=[cv1], logmean2=np.log([mean2]), cv2=[cv2]))
        df_predict['intercept'] = 1.0

        # 定义自变量
        X_new = df_predict[['intercept', 'logmean1', 'cv1', 'logmean2', 'cv2']]

        # 进行预测
        model = sm.Logit([0] * len(X_new), X_new)  # 这里的因变量无关紧要，只是为了创建模型实例
        model.initialize()  # 初始化模型
        prediction = model.predict(self.params)
        return prediction