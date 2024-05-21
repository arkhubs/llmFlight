from openai import OpenAI, Embedding
import logging, sys, os
sys.path.append(os.path.join(os.getcwd(), '../../'))
import API_KEYS

# 设置基本日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model="gpt-3.5-turbo-0125"
client = OpenAI(**API_KEYS.get_api_key(model))

def get_chat_response(question, temperature=1.0, n=1):
    """向 API 发送问题并获取 AI 的响应。"""
    try:
        # 尝试从 API 获取聊天响应
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            n = n,
        )
        if response:
            # 如果成功获取 AI 的响应,则记录并获取响应内容
            logging.info(f"AI 响应：{response}")
            choices = response.choices
            ans = [choice.message.content for choice in choices]
            return ans
        else:
            # 如果未能获取 AI 的响应,则记录错误日志
            logging.error("未从 API 收到响应。")
            return None
    except Exception as e:
        # 记录错误日志
        logging.error(f"从 API 获取响应时出错：{e}")
        return None


