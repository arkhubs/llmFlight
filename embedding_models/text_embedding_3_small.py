from openai import OpenAI, Embedding
import logging, sys, os
sys.path.append(os.path.join(os.getcwd(), '../../'))
import API_KEYS

# 设置基本日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = 'text-embedding-3-small'
client = OpenAI(**API_KEYS.get_api_key(model))

def get_embeddings(texts: list):
    """
    使用 OpenAI 的文本嵌入模型将文本转换为嵌入向量。
    输入：字符串列表；输出：字符串列表
    """
    try:
        # 尝试获取文本的嵌入向量
        resp = client.embeddings.create(input = texts, model=model)
        ans = [info['embedding'] for info in resp.to_dict()['data']]
        return ans
    except Exception as e:
        # 记录错误日志
        logging.error(f"获取嵌入向量时出错：{e}")
        return None


