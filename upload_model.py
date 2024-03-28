import os
from datetime import datetime

import boto3

if __name__ == '__main__':
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('BOTO_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('BOTO_SECRET_KEY'),
        region_name='ap-northeast-2'
    )
    s3.upload_file('./mnist.model', 'kreimben-general-bucket',
                   f'trained_models/mnist_classifier/mnist.model')
