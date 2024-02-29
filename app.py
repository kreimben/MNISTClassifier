import base64
import json
import logging
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

from src.module import ResNet

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set to DEBUG, INFO, WARNING, ERROR as needed


# Custom transform for inverting image colors
class InvertColours(object):
    def __call__(self, img):
        return Image.eval(img, lambda x: 255 - x)


def response(code, message):
    if code == 200:
        key = "prediction"
    else:
        key = "error_message"

    return {
        "statusCode": code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps({key: message})
    }


def lambda_handler(event, context):
    logger.info(f'Received event: {json.dumps(event)}')

    model = ResNet.load_from_checkpoint('mnist.ckpt')  # , block=ResidualBlock, layers=[3, 4, 6, 3], grayscale=True)
    model.eval()
    logger.info('after loading model.')

    # Define your transform pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        InvertColours(),  # change white to black, black to white.
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    try:
        # Decode the image
        body = event['body']

        if body is None:
            logger.error(f'No body received: {body=}')
            return response(400, 'There are no body. please input a valid image throgh your request.')

        # logger.info(f'{type(body)=}')
        if type(body) == str: body = json.loads(body)

        if body['image'] is None:
            logger.error(f'No image: {body=}')
            return response(400, 'There are no image. please input a valid image throgh your request.')

        image_data = body['image']
        image_decoded = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_decoded))
        # logger.info(f'Received image: {image}')

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            prediction = model(image_tensor)

        # Process the prediction as needed
        # For example, convert tensor to JSON serializable output
        prediction_result = prediction.argmax().item()
        logger.info(f'prediction result: {prediction_result}')

        return response(200, prediction_result)
    except Exception as e:
        return response(500, f'{e}, {event}, {context}')
