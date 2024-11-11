import streamlit as st
import os, sys

sys.path.append(os.path.join(os.getcwd(), '.../'))

if not hasattr(st.session_state, 'setting_point'):
    st.session_state.setting_point = 0

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')
st.title('Settings')


def set_lang():
    st.session_state.setting_point = 'language'

with st.sidebar:
    st.sidebar.button("Language", key="settings_language", type="primary", on_click=set_lang)

if st.session_state.setting_point:
    if st.session_state.setting_point == 'language':
        ## 默认渲染到主界面
        st.title('Language Settings')
        options = ['zh-CN', 'EN']
        with open('../Scripts/multi_language/choice.settings', 'r', encoding='utf-8') as file:
            # 将字符串变量写入文件
            choice = file.read()
        index = options.index(choice)
        lang = st.radio(
            label = 'Choose Your Language.',
            options = tuple(options),
            index = index,
            format_func = str
        )
        with open('../Scripts/multi_language/choice.settings', 'w+', encoding='utf-8') as file:
            # 将字符串变量写入文件
            file.write(lang)
        st.write("Your settings will be saved for long.")
        