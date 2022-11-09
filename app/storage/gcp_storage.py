import logging
import os
from typing import List

from google.cloud import storage

logger = logging.getLogger(__name__)


class GCPStorage:

    def __init__(self, bucket_id: str):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_id)

    def download_incoming_video(self, video_name: str, target_directory: str) -> str:
        destination_file_name = f"{target_directory}/{video_name}"
        base_name = self._get_video_path(video_name)
        self.download(from_path=base_name, to_path=destination_file_name)
        return destination_file_name

    def download(self, from_path: str, to_path: str, force: bool = False) -> bool:
        downloaded: bool
        if os.path.exists(to_path) and not force:
            logger.info(f"File {to_path} already exists")
            downloaded = False
        else:
            logger.info(f"Downloading file {from_path} from GCP...")
            blob = self.bucket.blob(from_path)
            blob.download_to_filename(to_path)
            logger.info(f"File downloaded from GCP to {to_path}")
            downloaded = True
        return downloaded

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
