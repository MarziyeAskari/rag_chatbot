import json
import os
import tempfile
import logging
import boto3
from src.config_loader import get_setting
from src.documents_processor import DocumentProcessor
from src.vector_store import VectorStore

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("rag-worker")

def main():
    settings = get_setting()
    queue_url = os.getenv("SQS_QUEUE_URL", "")
    if not queue_url:
        raise RuntimeError("SQS_QUEUE_URL missing")

    sqs = boto3.client("sqs", region_name=settings.aws_region)
    s3 = boto3.client("s3", region_name=settings.aws_region)

    doc = DocumentProcessor()
    vs = VectorStore(
        persist_directory=settings.vector_store_path,
        collection_name=settings.collection_name,
        vector_store_type=settings.vector_store_type,
        db_url=settings.vector_store_db_url,
    )

    logger.info("Worker polling %s", queue_url)

    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=600,
        )
        msgs = resp.get("Messages", [])
        if not msgs:
            continue

        msg = msgs[0]
        receipt = msg["ReceiptHandle"]
        tmp_path = None

        try:
            payload = json.loads(msg["Body"])
            bucket = payload["bucket"]
            key = payload["key"]
            job_id = payload.get("job_id")

            logger.info("Processing job=%s s3://%s/%s", job_id, bucket, key)

            with tempfile.NamedTemporaryFile(delete=False, dir="/tmp") as tmp:
                tmp_path = tmp.name
            s3.download_file(bucket, key, tmp_path)

            chunks = doc.process_files(tmp_path)
            vs.add_documents(chunks)

            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
            logger.info("Done job=%s", job_id)

        except Exception:
            logger.exception("Worker failed; message will retry")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

if __name__ == "__main__":
    main()