import os, sys

with open('../Scripts/multi_language/choice.settings', 'r', encoding='utf-8') as file:
    # 将字符串变量写入文件
    choice = file.read()



content = {
    # 'new_question': {'zh-CN': '请根据关键词${kw}和以${start_sentences}为开头的进行写作。',
    #     'EN': 'Please write based on the keyword ${kw} and starting with ${start_sentences}.'},
            
    'new_question': {
        'zh-CN': '请根据关键词${kw}进行写作。',
        'EN': 'Please write based on the keyword ${kw}.'
    },

    '#delete_session': {
        'zh-CN': "#### 删除会话",
        'EN': '#### Delete this Session'
    },

    'confirm_delete_session': {
        'zh-CN': '确认删除该用例？',
        'EN': 'Confirm deletion of this case?'
    },

    'confirm': {
        'zh-CN': '确认',
        'EN': 'Confirm'
    },

    'cancel':{
        'zh-CN': '取消',
        'EN': 'Cancel'
    },

    '#confirm_rename': {
        'zh-CN': '#### 确认重命名',
        'EN': '#### Confirm Renaming'
    },

    'confirm_rename_session': {
        'zh-CN': '确认重命名  \n`${oldname}`  \n为：  \n`${newname}`？',
        'EN': 'Confirm renaming  \n`${oldname}`  \nto:  \n`${newname}`?'
    },

    '#rename_session': {
        'zh-CN': '#### 重命名会话',
        'EN': '#### Rename Session'
    },

    '#create_session' :{
        'zh-CN': '#### 新建会话',
        'EN': '#### New Session'
    },

    'my_session': {
        'zh-CN': '会话',
        'EN': 'My Session'
    }, 

    'default_local_templates': {
        "zh-CN": """
                    <div style="background-color: ${color}; padding: 10px; margin: 10px 0;">
                        <strong>${num}. 段落预测概率: ${prob}%</strong>
                        <p>${text}</p>
                    </div>
                """,
        "EN": """
                    <div style="background-color: ${color}; padding: 10px; margin: 10px 0;">
                        <strong>${num}. Probability predicted from this paragraph: ${prob}%</strong>
                        <p>${text}</p>
                    </div>
                """
    }, 

    'default_prompt_templates': {
        "zh-CN": """
                    ${num}. 段落预测概率: ${prob}%
                """,
        "EN": """
                    ${num}. Probability predicted from this paragraph: ${prob}%
                """
    }, 

    'default_global_templates': {
        "zh-CN": """
                    ### 主模型综合预测概率  
                        ${main}；
                    ### 特化模型预测概率：  
                    - 词语丰富度模型：
                        ${words_richness}；
                    - 句子长度模型：
                        ${sentences_length}；
                    - 情感强度模型：
                        ${emotion_Dou}；                
                """,
        "EN": """
                    ### Probability predicted by Main Model (Conbined)  
                        ${main};
                    ### Probability predicted by Specialized Model:  
                    - Word Richness Model：
                        ${words_richness}；
                    - Sentence Length Model：
                        ${sentences_length}；
                    - Emotion Intensity Model：
                        ${emotion_Dou}；                
                """
    }, 

    'session_history': {
        "zh-CN": "历史会话",
        "EN": "Sessions History"
    },

    '+new_session': {
        "zh-CN": "＋ 新建会话",
        "EN": "＋ New Session"
    },

    'generate_question': {
        "zh-CN": "不能提供问题文本？点击为你生成一个问题！",
        "EN": "Difficult to provide question text? Click to generate a question for you!"
    },

    'input': {
        "zh-CN": "输入",
        "EN": "Input"
    },

    'detection': {
        "zh-CN": "检测！",
        "EN": "Detection!"
    },

    '#agent_summary': {
        "zh-CN": "### Agent总结：",
        "EN": "### Agent Summary:"
    }, 

    'prompt_agent_summary': {
        "zh-CN": """
                        以下是AI文本检测报告，但不够直观，请分别帮我汇总一个简洁的结论和一个具体分析。以综合概率为主，如果综合概率低，要点出不太可能由AI生成；如果综合概率高，请进行归因。以这样的格式：
                        【简洁结论】\n
                        ……\n
                        【具体分析】\n
                        词语丰富度模型预测文本为AI生成的概率……，表明……
                        句子长度模型和情感强度模型的预测概率分别为…………
                        分段预测中，……的预测概率……，分别为……，提示……\n
                        综合来看，……
                    """,
        "EN": """
                        Below is the AI text detection report, but it's not intuitive enough. Please summarize a concise conclusion and a detailed analysis. Focus mainly on the overall probability. If the overall probability is low, point out that it is unlikely to be AI-generated. If the overall probability is high, provide attribution. Use the following format:
                        【Concise Conclusion】\n
                        ……\n
                        【Detailed Analysis】\n
                        The word richness model predicts the probability that the text is AI-generated as …… , indicating …… 
                        The sentence length model and emotion intensity model predict probabilities of …… respectively.
                        In paragraph predictions, the probability of …… is …… , with predictions of …… , suggesting ……\n
                        Overall, …… 
                    """
    },

    '#predict each': {
        "zh-CN": "### 分段预测概率：",
        "EN": "### Probability Predicted from each paragraph: "
    },

    "text_examples": {
        "zh-CN": "我们提供的文本示例",
        "EN": "Text Examples We Provide"
    }, 

    "#app_share": {
        "zh-CN": "# APP分享",
        "EN": "# Share APP to Other Devices"
    }, 

    "intranet": {
        "zh-CN": "内网（推荐）",
        "EN": "Intranet (recommended)"
    },

    "public_network": {
        "zh-CN": "公网（不推荐）",
        "EN": "Public Network (not recommended)"
    }, 

    "internal_ip": {
        "zh-CN": "内网地址：",
        "EN": "Intranet ip: "
    },

    "external_ip": {
        "zh-CN": "外网地址：",
        "EN": "Public Network ip: "
    },

    "#project_site": {
        "zh-CN": "### 项目地址",
        "EN": "### Project Site"
    },
}

# 遍历原始字典的键和值
used_content = {key: value[choice] for key, value in content.items()}