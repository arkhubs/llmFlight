import json

API_KEYS = {
    "text-embedding-3-small": {
        "api_key": "11111111111", 
        "base_url": "https://api.crowtalk.live/v1"
    }, 
    "gpt-3.5-turbo-0125": {
        "api_key": "11111111111", 
        "base_url": "https://api.crowtalk.live/v1"
    }
}

def get_api_key(model_name):
    return API_KEYS[model_name]