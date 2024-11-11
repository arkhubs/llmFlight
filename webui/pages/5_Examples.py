import streamlit as st

from Scripts.multi_language.languages import used_content as language

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')

st.title(language["text_examples"])

with open('examples/chatgpt.txt', 'rb') as f:
   st.download_button('Example of ChatGPT', f, 'chatgpt.txt')

with open('examples/human.txt', 'rb') as f:
   st.download_button('Example of Human', f, 'human.txt')