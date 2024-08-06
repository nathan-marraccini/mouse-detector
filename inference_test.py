# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="fZkwfr3c0A2hZjtLSdM8"
)

# infer on a local image
result = CLIENT.infer("https://imgix.bustle.com/mic/cwba7skl4g0zwiom2yvlrghrqtsw11dkkto7lxucifk92qqjmkitebtie2zwqwiy.jpg", model_id="rps-nvidia-demo/1")
print(result)
