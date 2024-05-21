set conda_path="E:\miniconda"
set env_name="pytorchDML"



call %conda_path%\Scripts\activate.bat %env_name%
cd %~dp0
streamlit run webui/AIContentDetector.py --server.port 8502