import streamlit as st

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')

st.title('我们提供的文本示例')

with open('examples/chatgpt.txt', 'rb') as f:
   st.download_button('Example of ChatGPT', f, 'chatgpt.txt')

with open('examples/human.txt', 'rb') as f:
   st.download_button('Example of Human', f, 'human.txt')