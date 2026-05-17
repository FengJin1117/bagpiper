import requests


url = "http://127.0.0.1:8000/infer"

data = {
    "audio_path": "examples/test.wav",
    "prompt": "Describe this audio in detail.",
}

resp = requests.post(url, json=data)

print(resp.json())