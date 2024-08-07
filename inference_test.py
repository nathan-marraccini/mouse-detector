# import the inference-sdk
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="V2lTkhacO8LY3TPugNVO"
)

result = client.run_workflow(
    workspace_name="hunter-diminick",
    workflow_id="custom-workflow",
    images={"image": "YOUR_IMAGE.jpg"}
)
