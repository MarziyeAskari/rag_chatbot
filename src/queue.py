import json
import boto3

class SQSClient:
    def __init__(self, queue_url: str, region: str):
        if not queue_url:
            raise ValueError("SQS_QUEUE_URL is required")
        self.queue_url = queue_url
        self.client = boto3.client("sqs", region_name=region)

    def send(self, payload: dict) -> None:
        self.client.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(payload),
        )