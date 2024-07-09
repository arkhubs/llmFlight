import statsmodels.api as sm
import pandas as pd
import pickle, os, sys

sys.path.append(os.path.dirname(__file__))
from emotion_Dou import main

class Model():
    def __init__(self, pkl='../specialized_models/emotion_Dou/emotion_Dou_202406051604.pkl'):

        # 加载模型参数
        with open(pkl, 'rb') as f:
            params = pickle.load(f)
        self.params = params
        
    def infer(self, text):
        eps, eiv = main(text)
        df_predict = pd.DataFrame(dict(eps=[eps], eiv=[eiv]))
        df_predict['intercept'] = 1.0

        # 定义自变量
        X_new = df_predict[['intercept', 'eps', 'eiv']]

        # 进行预测
        model = sm.Logit([0] * len(X_new), X_new)  # 这里的因变量无关紧要，只是为了创建模型实例
        model.initialize()  # 初始化模型
        prediction = model.predict(self.params)
        return prediction