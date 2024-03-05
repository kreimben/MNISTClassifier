import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.module import ResNet
from src.transform import transform


# load model

@st.cache_resource
def load_model():
    model = ResNet.load_from_checkpoint('mnist.ckpt')
    model.eval()
    return model


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

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    image = Image.fromarray(np.uint8(canvas_result.image_data)).convert('RGB')

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        prediction = model(image_tensor)

    prediction_result = prediction.argmax().item()

    st.write(f'예측 결과: {prediction_result}입니다.')

    # if st.checkbox('틀렸나요?'):
    #     st.write()
    #     right = st.radio('원래 정답은 뭐였나요?', [str(i) for i in range(10)])
    #     st.write(right)
