import streamlit as st
import qrcode, socket
import numpy as np

from Scripts.multi_language.languages import used_content as language

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

st.markdown(language["#app_share"])
c1, c2 = st.columns(spec=2)

with c1:
    st.image('ip1.png', caption=language["intranet"])
    st.markdown(f"{language['internal_ip']}{ip1}")
with c2:
    st.image('ip2.png', caption=language["public_network"])
    st.markdown(f"{language['external_ip']}{ip2}")

st.markdown(language["#project_site"])
st.markdown('https://github.com/arkhubs/llmFlight')
st.markdown('https://huggingface.co/datasets/xuanfl/datasets-for-llmFlight')