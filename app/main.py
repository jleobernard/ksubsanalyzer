import json
import logging
import os
from typing import List

import google.auth
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1

logging.basicConfig(format='%(asctime)s[%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

from analyzer.v2.analyzer import VideoAnalyzer
from storage.gcp_storage import GCPStorage

subscription_id = os.getenv("GCP_PUBSUB_SUBSCRIPTION_ID")
pub_sub_creds, project_id = google.auth.load_credentials_from_file(os.getenv("GCP_PUBSUB_CREDENTIALS"))
subscriber = pubsub_v1.SubscriberClient(credentials=pub_sub_creds)
subscription_path = subscriber.subscription_path(project_id, subscription_id)

work_directory = os.getenv("WORK_DIRECTORY")
target_directory = os.getenv("TARGET_DIRECTORY")

gcp_storage = GCPStorage(credentials_path=os.getenv("GCP_STORAGE_CREDENTIALS"),
                         bucket_id=os.getenv("GCP_STORAGE_BUCKET_ID"))
analyzer = VideoAnalyzer(work_directory=work_directory,
                         model_name=os.getenv("MODEL_NAME"),
                         target_directory=target_directory,
                         skip_frames=int(os.getenv("SKIP_FRAMES")),
                         credentials_path=os.getenv("GCP_VISION_CREDENTIALS"),
                         storage=gcp_storage)


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    message.ack()
    try:
        data = json.loads(message.data)
        video_name = data['videoName']
        logger.info(f"Received message of type {type} for video {video_name}.")
        video_path: str = gcp_storage.download_incoming_video(video_name=video_name, target_directory=work_directory)
        products_path: List[str] = analyzer.treat_incoming_file(video_path)
        gcp_storage.upload_analysis_products(video_name=video_name, products_path=products_path)

    except BaseException as err:
        logger.error(f"Error while treating message {message}")
        logger.error(err)


streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
logger.info(f"Listening for messages on {subscription_path}..\n")

# Wrap subscriber in a 'with' block to automatically call close() when done.
with subscriber:
    try:
        streaming_pull_future.result()
    except TimeoutError:
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.
