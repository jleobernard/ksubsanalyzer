import logging
import os
from typing import List

from google.cloud import storage
import google.auth

logger = logging.getLogger(__name__)


class GCPStorage:

    def __init__(self, credentials_path: str, bucket_id: str):
        creds, project_id = google.auth.load_credentials_from_file(credentials_path)
        self.storage_client = storage.Client(credentials=creds)
        self.bucket = self.storage_client.bucket(bucket_id)

    def download_incoming_video(self, video_name: str, target_directory: str) -> str:
        destination_file_name = f"{target_directory}/{video_name}"
        if os.path.exists(destination_file_name):
            logger.info(f"Video file {video_name} already downloaded in {destination_file_name}")
        else:
            logger.info(f"Downloading file {video_name} from GCP...")
            base_name = self._get_video_path(video_name)
            blob = self.bucket.blob(base_name)
            blob.download_to_filename(destination_file_name)
            logger.info(f"File downloaded from GCP to {destination_file_name}")
        return destination_file_name

    def upload_analysis_products(self, video_name: str, products_path: List[str]):
        base_name: str = os.path.splitext(video_name)[0]
        gcp_prefix: str = f"videos/{base_name}"
        for product_path in products_path:
            file_name: str = os.path.basename(product_path)
            if file_name != video_name:
                logger.info(f"Uploading file {product_path} to GCP...")
                blob = self.bucket.blob(f"{gcp_prefix}/{file_name}")
                blob.upload_from_filename(product_path)
                logger.info(f"File uploaded to GCP to {gcp_prefix}/{file_name}")

    @staticmethod
    def _get_video_path(video_name: str) -> str:
        base_name: str = os.path.splitext(video_name)[0]
        return f"videos/{base_name}/{video_name}"
