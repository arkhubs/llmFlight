import streamlit as st
from streamlit_modal import Modal

import sys, os, importlib
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.getcwd(), '../'))
supported_models = [model for model in os.listdir('../models') if os.path.isdir(os.path.join('../models', model))]

## 加载模型，导入模型对应的inference模块
@st.cache_resource
def init_model(model_name):
    pths = os.listdir(os.path.join('../models', model_name, 'pths'))
    inference = importlib.import_module(f"models.{model_name}.inference")
    return pths, inference

## 规范字符串到边栏会话按钮要求
@st.cache_data
def adj_str(input_str, length=18, padding_char='\u3000'):
    actual_length = sum(2 if ord(char) > 127 else 1 for char in input_str)
    if actual_length > length:
        return input_str[:length]
    else:
        return input_str.ljust(length - (actual_length - len(input_str)), padding_char)

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
                    "model_index": 0,        # 选过的模型id
                    "device_index": 0,       # 选过的设备id
                    "pth_index": 0,          # 选过的参数id
                }, {
                    "model_index": 0,
                    "device_index": 0,
                    "pth_index": 0,
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






























## 初始化，创建历史会话记录
if not hasattr(st.session_state, 'history'):
    st.session_state.history = []
if not hasattr(st.session_state, 'current'):
    st.session_state.current = {}

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
            model_name = st.selectbox(label="model", key=f'select-model-{id}-{tabid}', options=supported_models, index=data['model_index'])
            data['model_index'] = supported_models.index(model_name)
            pths, inference = init_model(model_name)
            supported_devices = st.cache_resource(inference.supported_devices)()
            device = st.selectbox(label="device", key=f'select-device-{id}-{tabid}', options=list(supported_devices.keys()), index=data['device_index'])
            data['device_index'] = list(supported_devices.keys()).index(device)
            pth = st.selectbox(label="model params", key=f'select-params-{id}-{tabid}', options=pths, index=data['pth_index'])
            data['pth_index'] = pths.index(pth)
            model = st.cache_resource(inference.Model)(os.path.join("../models", model_name, "pths", pth), supported_devices[device])
            # 输入文本
            question_text = st.text_area(label="question", value=data.get('question_text', ""), height=100, key=f"question-input-{id}-{tabid}", 
                                         on_change=save_question_text, args=(data, f"question-input-{id}-{tabid}"))
            st.write(f"The text has about `{sum(2 if ord(char) > 127 else 1 for char in question_text)}` tokens.")
            answer_text = st.text_area(label="answer", value=data.get('answer_text', ""), height=500, key=f"answer-input-{id}-{tabid}",
                                       on_change=save_answer_text, args=(data, f"answer-input-{id}-{tabid}"))
            st.write(f"The text has about `{sum(2 if ord(char) > 127 else 1 for char in answer_text)}` tokens.")
            # 预测
            if st.button("检测", key=f"infer-{id}-{tabid}"):
                question_embedding = st.cache_data(inference.get_embeddings)([question_text])[0]
                answer_embedding = st.cache_data(inference.get_embeddings)([answer_text])[0]
                res = st.cache_data(model.infer)(question_embedding, answer_embedding)
                st.write(res, "inferenced by", device)































































