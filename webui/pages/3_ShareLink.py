import streamlit as st
import qrcode, socket
import numpy as np

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')

ip1 = 'http://' + socket.gethostbyname(socket.gethostname()) + ':8502'
ip2 = 'https://68d340106n.goho.co'
qrcode.make(ip1, error_correction=2).save('ip1.png')
qrcode.make(ip2, error_correction=2).save('ip2.png')

st.markdown('### APP分享')
c1, c2 = st.columns(spec=2)

with c1:
    st.image('ip1.png', caption='内网（推荐）')
    st.markdown(f"内网地址：{ip1}")
with c2:
    st.image('ip2.png', caption='公网（不推荐）')
    st.markdown(f"公网地址：{ip2}")

ip3 = 'http://' + socket.gethostbyname(socket.gethostname()) + ':8000'
st.markdown("### 项目地址")
st.markdown('https://github.com/arkhubs/llmFlight')
st.markdown('https://huggingface.co/datasets/xuanfl/datasets-for-llmFlight')
st.markdown(f"{ip3}")