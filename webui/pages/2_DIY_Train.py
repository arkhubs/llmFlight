import streamlit as st

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')

st.markdown('> Streamlit 支持通过 st.markdown 直接渲染 markdown')

with st.sidebar:
    st.title('欢迎来到我的应用')
    st.markdown('---')
    st.markdown('这是它的特性：\n- feature 1\n- feature 2\n- feature 3')
    c1, c2 = st.columns(spec=2)
    c1.title('This is Column Ⅰ')
    c2.title('This is Column Ⅱ')
## 默认渲染到主界面
st.title('这是主界面')
st.info('这是主界面内容')