# opjective

I havbe a hugging-face (multilinaul e5 small) that i need to serve.

the serve should be using fastapi

put all the code in an app folder

add a download_model.py for managing the download of the model from HF. save it in a data folder


use this code as client

import requests
import numpy as np
from transformers import AutoTokenizer

TOKENIZER_NAME = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


MODEL_NAME = "multilingual-e5-small"
URL = ""  # TODO 

def get_embedding(text: str):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    payload = {}  # TODO convert input to request payload

    response = requests.post(URL, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Server inference failed: {response.text}")

    result = response.json()
    # TODO: retrieve the embedding from the result 
    return result


if __name__ == "__main__":
    text = "hi"
    embedding = get_embedding(text)
    print(f"Embedding shape: {embedding.shape}")
    print(embedding[:10])
