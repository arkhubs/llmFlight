import json

API_KEYS = {
    "text-embedding-3-small": {
        "api_key": "",   # Please fill in your own api_key 
        "base_url": "https://api.crowtalk.live/v1"
    }, 
    "gpt-3.5-turbo-0125": {
        "api_key": "",   # Please fill in your own api_key  
        "base_url": "https://api.crowtalk.live/v1"
    }
}

def get_api_key(model_name):
    return API_KEYS[model_name]