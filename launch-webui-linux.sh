conda_path="/miniconda"
env_name="llmFlight"


#!/bin/bash
source ${conda_path}/bin/activate ${env_name}
cd %~dp0
streamlit run webui/AIContentDetector.py