import json
from uuid import uuid4

from locust import HttpUser, constant_pacing, task
from transformers import AutoTokenizer

TOKENIZER_NAME = "intfloat/multilingual-e5-small"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# MODEL_NAME = "multilingual-e5-small"
# INFER_PATH = ""  # TOpythinDO add url (without host)

MODEL_NAME = "e5"  # Use the correct model name in your server
INFER_PATH = (
    "http://XXXXX:8000/v2/models/e5/infer"  # Update to the correct inference path
)

TEXT = "let's make model serving easy!"


# to run the stress test run
#  locust -f <file-path> --autostart --host <host> -r 1000 -u <requests per seconds>
class TestUser(HttpUser):
    wait_time = constant_pacing(1.0)

    @task
    def send_embedding_request(self):
        text = TEXT + " " + str(uuid4())

        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        payload = {}  # TODO convert input to request payload

        with self.client.post(
            INFER_PATH,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"{resp.status_code} | {resp.text}")
                return

            resp.success()
