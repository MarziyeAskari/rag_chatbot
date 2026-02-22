import os
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.exceptions import ClientError


@dataclass
class UploadResult:
    storage: str           # "local" | "s3"
    uri: str               # local path or s3://...
    bucket: Optional[str]  # for s3
    key: Optional[str]     # for s3


class UploadStorage:
    def save_bytes(self, filename: str, content: bytes) -> UploadResult:
        raise NotImplementedError


class LocalUploadStorage(UploadStorage):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_bytes(self, filename: str, content: bytes) -> UploadResult:
        safe_name = os.path.basename(filename)
        path = os.path.join(self.base_dir, safe_name)
        with open(path, "wb") as f:
            f.write(content)
        return UploadResult(storage="local", uri=path, bucket=None, key=None)


class S3UploadStorage(UploadStorage):
    def __init__(self, bucket: str, prefix: str = "uploads/", region: str = "us-east-1"):
        self.bucket = bucket
        self.prefix = (prefix.strip("/") + "/") if prefix else ""
        self.s3 = boto3.client("s3", region_name=region)

    def save_bytes(self, filename: str, content: bytes) -> UploadResult:
        safe_name = os.path.basename(filename)
        key = f"{self.prefix}{safe_name}"

        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=content)
        except ClientError as e:
            raise RuntimeError(f"S3 put_object failed: {e.response['Error']}") from e

        return UploadResult(
            storage="s3",
            uri=f"s3://{self.bucket}/{key}",
            bucket=self.bucket,
            key=key,
        )