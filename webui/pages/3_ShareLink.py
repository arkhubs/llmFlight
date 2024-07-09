import streamlit as st
import qrcode, socket
import numpy as np

def extract_ip():
    ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        #ss.connect(('10.255.255.255', 1))
        ss.connect(('8.8.8.8', 80))
        IP = ss.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        ss.close()
    return IP

st.set_page_config(page_title="llmFlight", page_icon=' ', layout='wide')

ip1 = 'http://' + extract_ip() + ':8502'
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

st.markdown("### 项目地址")
st.markdown('https://github.com/arkhubs/llmFlight')
st.markdown('https://huggingface.co/datasets/xuanfl/datasets-for-llmFlight')