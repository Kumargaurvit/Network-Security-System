'''
CODE TO SYNC FILES TO AND FROM AWS S3 BUCKET
'''
import os
import sys
from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging

class S3Sync:
    def __init__(self):
        pass

    def sync_folder_to_s3(self,folder,aws_bucket_url):
        try:
            logging.info('Syncing files to AWS S3')
            command = f"aws s3 sync {folder} {aws_bucket_url}"
            os.system(command)
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
    
    def sync_folder_from_s3(self,folder,aws_bucket_url):
        try:
            logging.info('Syncing files from AWS S3')
            command = f"aws s3 sync {aws_bucket_url} {folder}"
            os.system(command)
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)