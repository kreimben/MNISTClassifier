import os

import boto3
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.module import ResNet
from src.transform import inference_transform


@st.cache_resource
def download_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('BOTO_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('BOTO_SECRET_KEY'),
        region_name='ap-northeast-2'
    )
    s3.download_file('kreimben-general-bucket', 'trained_models/mnist_classifier/mnist.model', 'mnist.model')


@st.cache_resource
def load_model():
    model = ResNet.load_from_checkpoint('mnist.model')
    model.eval()
    return model


if 'mnist.model' not in os.listdir('./'):
    download_from_s3()
model = load_model()

st.title("손글씨 **숫자** 예측 딥러닝 모델")

st.write('MNIST 데이터 기반의 손글씨 숫자')
st.write('숫자 1개를 그려보세요.')

# Create a canvas component
canvas_result = st_canvas(
    background_color='#FFFFF0',
    height=250,
    width=250,
    key="full_app",
)


def predict(image_data):
    image = Image.fromarray(np.uint8(image_data)).convert('RGB')

    # Preprocess the image
    image_tensor = inference_transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)

    prediction_result = prediction.argmax().item()

    st.write(f'예측 결과: {prediction_result}입니다.')


# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.button('예측하기', on_click=lambda: predict(canvas_result.image_data))
