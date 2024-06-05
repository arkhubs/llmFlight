'''
Copyright 2024 Zhixuan Hu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import streamlit as st
from streamlit_modal import Modal

import sys, os, re, math, importlib
from string import Template
import jieba.analyse

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.getcwd(), '../'))
from settings import *
from llm_models.gpt_3_5_turbo_0125 import get_chat_response
supported_models = [model for model in os.listdir('../main_models') if os.path.isdir(os.path.join('../main_models', model))]
specialized_models = [(model_name, importlib.import_module(f"specialized_models.{model_name}.inference").Model()) for model_name in [
    'emotion_Dou',
    'sentences_length',
    'words_richness'
]]

## 加载模型，导入模型对应的inference模块
@st.cache_resource
def init_main_model(model_name):
    pths = os.listdir(os.path.join('../main_models', model_name, 'pths'))
    inference = importlib.import_module(f"main_models.{model_name}.inference")
    return pths, inference

## 规范字符串到边栏会话按钮要求
@st.cache_data
def adj_str(input_str, length=18, padding_char='\u3000'):
    actual_length = sum(2 if ord(char) > 127 else 1 for char in input_str)
    if actual_length > length:
        return input_str[:length]
    else:
        return input_str.ljust(length - (actual_length - len(input_str)), padding_char)

## 分割文本为语段
@st.cache_data
def split_text_into_segments(text, tol=200, reg=r'[\n]'):
    # 使用正则表达式分割文本，分隔符为句号、问号或感叹号
    segments = re.split(reg, text)
    # 过滤掉空字符串，并在每个段落后加上原来的结束符号
    segments = [seg.strip() for seg in segments if seg.strip() != '']
     # 合并长度小于 tol 的语段
    merged_segments = []
    current_segment = ''
    
    for seg in segments:
        if len(current_segment) + len(seg) + 1 <= tol:
            current_segment += (' ' + seg if current_segment else seg)
        else:
            if current_segment:
                merged_segments.append(current_segment)
            current_segment = seg

    if current_segment:
        merged_segments.append(current_segment)
    
    return merged_segments

## 补全问题文本
def generate_question(text, num_topics):

    stopwords = {
        '的', '了', '和', '是', '在', '也', '有', '就', '不', '都', '而', '及', '与', '或', '一个', '中', '这', '以及', '一个',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '其', '将', '要', '已经', '还', '还要', '再',
        '没有', '不是', '非常', '特别', '很', '特别', '太', '然后', '但是', '所以', '如果', '因为', '因为而', '所以才', '由于', '因此', 
        '就', '而且', '然而', '并且', '就是', '即', '又', '这', '那', '这些', '那些', '时候', '当时', '已', '我', '你', '他',
        '她', '它', '的', '一个', '和', '而且', '并', '与', '来', '在', '到', '也', '有', '自己', '我们', '你们', '他们',
        '她们', '它们', '该', '以及', '但', '并', '又', '同', '接着', '等', '可是', '而', '于是', '而是', '并且', '以及', 
        '还', '还是', '仍然', '还是', '甚至', '的', '了', '就', '都', '而', '和', '不', '这', '也', '一', '来', '就', '不',
        '中', '为', '在', '可', '却', '与', '于', '他', '她', '它', '能', '所', '个', '人', '这', '那', '也', '而', '但', 
        '把', '却', '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '这', '那', '这些', '那些'
    }

    kw = jieba.analyse.extract_tags(text, topK=num_topics)

    # # 获取文章开头的2-4句话
    # sentences = split_sentences(text)
    # start_sentences = ' '.join(sentences[:random.randint(2, 4)])

    # 构建新的问题文本
    # new_question = f"请根据关键词{kw}和以{start_sentences}为开头的进行写作。"
    new_question = f"请根据关键词{kw}进行写作。"

    return new_question

## 删除会话
def delete_session(id):
    def delete():
        del st.session_state.history[id]
    # 弹出确认对话框
    session = st.session_state.history[id]
    msg = Modal(title="#### 删除会话", key="delete", max_width=500).container()
    with msg:
        st.markdown(f"确认删除该用例？ \n{session['name']}")
        b1, b2 = st.columns(spec=[1, 6])
        b1.button("确认", on_click=delete) # 使用onlick才会调用成功并成功刷新，不能用if
        b2.button("取消")

## 重命名会话
def rename_session(id):
    def tru_rename():
        st.session_state.history[id]['name'] = st.session_state['text']
    # 弹出确认对话框
    def rename():
        st.session_state.text = st.session_state['rename-text'] # 读取自动保存的输入文本，目前只能采用此策略读到文本
        msg = Modal(title="#### 确认重命名", key="confirm-rename", max_width=500).container()
        with msg:
            st.markdown(f"确认重命名  \n`{oldname}`  \n为：  \n`{st.session_state['text']}`？")
            b1, b2 = st.columns(spec=[1, 6])
            b1.button("确认", on_click=tru_rename)
            b2.button("取消")
    # 弹出输入对话框
    session = st.session_state.history[id]
    oldname = session['name']
    msg = Modal(title="#### 重命名会话", key="rename", max_width=500).container()
    with msg:
        st.text_input(label=" ", key="rename-text", value=session['name'], on_change=rename) # 当按下回车会触发on_change

## 新建会话
def new_session():
    def create():
        st.session_state.history.append({
            "id": len(st.session_state.history),   # 唯一标识
            'name': st.session_state['new-text'],  # 会话名称
            "tab_count": 2,                        # tab数量
            'data': [{                             # 各个tab的保存信息
                    "model_index": DEFAULT_MODEL_INDEX,        # 选过的模型id
                    "device_index": DEFAULT_DEVICE_INDEX,       # 选过的设备id
                    "pth_index": DEFAULT_PTH_INDEX,          # 选过的参数id
                }, {
                    "model_index": DEFAULT_MODEL_INDEX,
                    "device_index": DEFAULT_DEVICE_INDEX,
                    "pth_index": DEFAULT_PTH_INDEX,
                },
            ]
        })
    # 弹出输入对话框
    msg = Modal(title="#### 新建会话", key="new", max_width=500).container()
    with msg:
        st.text_input(label=" ", key="new-text", value="会话 ", on_change=create)

## 通过onchange触发，缓存问题文本
def save_question_text(data, key):
    data['question_text'] = st.session_state[key]

## 通过onchange触发，缓存回答文本
def save_answer_text(data, key):
    data['answer_text'] = st.session_state[key]

class Reporter():
    def __init__(self):
        self.local_templates = {
            'default': Template("""
                <div style="background-color: ${color}; padding: 10px; margin: 10px 0;">
                    <strong>${num}. 段落预测概率: ${prob}%</strong>
                    <p>${text}</p>
                </div>
            """)}
        
        self.prompt_templates = {
            'default': Template("""
                ${num}. 段落预测概率: ${prob}%
            """)}
        
        self.global_templates = {
            'default': Template("""
                ### 主模型综合预测概率  
                    ${main}；
                ### 特化模型预测概率：  
                - 词语丰富度模型：
                    ${words_richness}；
                - 句子长度模型：
                    ${sentences_length}；
                - 情感强度模型：
                    ${emotion_Dou}；                
            """)}

    # 根据概率计算颜色的深度
    def color_gradient(self, prob):
        intensity = 255 - int(math.floor(155 * prob))  # 保证至少是100，所以颜色不会太深
        return f"rgb(255, {intensity}, {intensity})"  # 红色渐变
    
    # 生成分段预测报告
    def local_render(self, template, segment, probs):
        local_template = self.local_templates[template]
        ans_seg = [
            local_template.substitute(text=text, prob=f"{prob*100:.1f}", color=self.color_gradient(prob), num=num)
            for num, (text, prob) in enumerate(zip(segment, probs), start=1)
        ]
        return "\n".join(ans_seg)
    
    def local_prompt(self, template, probs):
        template = self.prompt_templates[template]
        ans = [
            template.substitute(prob=f"{prob*100:.1f}", num=num)
            for num, prob in enumerate(probs, start=1)
        ]
        return "\n".join(ans)

    # 生成总体预测报告
    def global_render(self, template, probs):
        global_template = self.global_templates[template]
        return global_template.substitute(**probs)
    


























## 初始化，创建历史会话记录
if not hasattr(st.session_state, 'history'):
    st.session_state.history = []
if not hasattr(st.session_state, 'current'):
    st.session_state.current = {}

if 'reporter' not in globals():
    reporter = Reporter()

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')

## 侧边栏
st.sidebar.header("历史会话")

for session in st.session_state.history:
    c1, c2, c3 = st.sidebar.columns(spec=[9, 2, 2])
    if c1.button(adj_str(session['name']), key=f"{session['id']}-on"):
        # 点击后加载session到current
        st.session_state.current = session
    c2.button("🗑️", key=f"{session['id']}-delete", on_click=delete_session, args=(session['id'],))
    c3.button("🖉", key=f"{session['id']}-rename", on_click=rename_session, args=(session['id'],))

st.sidebar.button("＋ 新建会话", key="new_session", type="primary", on_click=new_session)

## 根据current加载主界面
if st.session_state.current:
    session = st.session_state.current
    id = session['id']
    st.markdown(f"#### {session['name']}")
    # 加载各个tab
    for tabid, tab in enumerate(st.tabs([f"Tab{i+1}" for i in range(session['tab_count'])])):
        with tab:
            # 选择模型、设备、参数
            data = session['data'][tabid]
            num_options = len(supported_models)
            model_name = st.selectbox(label="model", key=f'select-model-{id}-{tabid}', options=supported_models, index=(num_options+data['model_index']) % num_options)
            data['model_index'] = supported_models.index(model_name)
            pths, inference = init_main_model(model_name)
            supported_devices = st.cache_resource(inference.supported_devices)()
            num_options = len(supported_devices)
            device = st.selectbox(label="device", key=f'select-device-{id}-{tabid}', options=list(supported_devices.keys()), index=(num_options+data['device_index']) % num_options)
            data['device_index'] = list(supported_devices.keys()).index(device)
            num_options = len(pths)
            pth = st.selectbox(label="model params", key=f'select-params-{id}-{tabid}', options=pths, index=(num_options+data['pth_index']) % num_options)
            data['pth_index'] = pths.index(pth)
            model = inference.Model(os.path.join("../main_models", model_name, "pths", pth), supported_devices[device])

            # 输入文本
            question_text = st.text_area(label="question", value=data.get('question_text', ""), height=100, key=f"question-input-{id}-{tabid}", 
                                         on_change=save_question_text, args=(data, f"question-input-{id}-{tabid}"))
            st.write(f"The text has about `{sum(2 if ord(char) > 127 else 1 for char in question_text)}` tokens.")
            answer_text = st.text_area(label="answer", value=data.get('answer_text', ""), height=500, key=f"answer-input-{id}-{tabid}",
                                       on_change=save_answer_text, args=(data, f"answer-input-{id}-{tabid}"))
            st.write(f"The text has about `{sum(2 if ord(char) > 127 else 1 for char in answer_text)}` tokens.")

            cc1, cc2 = st.columns(spec=[1, 9])
            # 补全问题文本
            if cc1.button("生成问题", key=f"generate-question-{id}-{tabid}"):
                question_text = generate_question(answer_text, 10)
                st.markdown(f"`{question_text}`")
                data['question_text'] = question_text

            # 预测
            if cc2.button("检测", key=f"infer-{id}-{tabid}"):
                question_embedding = inference.get_embeddings([question_text])[0]
                answer_seg = [answer_text] + split_text_into_segments(answer_text)
                answer_embeddings = inference.get_embeddings(answer_seg)
                probs = [model.infer(question_embedding, answer_embedding) for answer_embedding in answer_embeddings]
                specialized_probs = {**{'main': f"{probs[0]*100:.1f}%"}, **{model_name: f"{model.infer(answer_text)[0]*100:.1f}%" for model_name, model in specialized_models}}

                global_info = reporter.global_render('default', specialized_probs)
                st.markdown(global_info)
                st.markdown("### Agent总结：")
                prompt = """
                    以下是AI文本检测报告，但不够直观，请帮我汇总一个简洁的结论出来。如果综合概率高，请进行归因。以这样的格式：
                    【简洁结论】\n
                    ……\n
                    【具体分析】\n
                    词语丰富度模型预测文本为AI生成的概率……，表明……
                    句子长度模型和情感强度模型的预测概率分别为…………
                    分段预测中，……的预测概率较高，分别为……，提示……\n
                    综合来看，……
                """ + global_info + reporter.local_prompt('default', probs[1:])
                st.write(get_chat_response(prompt)[0])
                st.markdown("### 分段预测概率：")
                st.html(reporter.local_render('default', answer_seg[1:], probs[1:]))
                
































































