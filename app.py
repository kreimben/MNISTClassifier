import base64
import json
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

from src.module import ResNet, ResidualBlock

model = ResNet.load_from_checkpoint('mnist.ckpt',
                                    block=ResidualBlock, layers=[2, 2, 2, 2], grayscale=True, batch_size=128)
model.eval()

# Define your transform pipeline
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
])


def lambda_handler(event, context):
    print(f'Received event: {json.dumps(event)}')
    global model
    print(f'after loading model: {json.dumps(model)}')

    try:
        # Decode the image
        body = json.loads(event['body'])
        if body is None:
            return json.dumps({'error': 'There are no body. please input a valid image throgh your request.'})

        if body['image'] is None:
            return json.dumps({'error': 'There are no image. please input a valid image throgh your request.'})

        image_data = body['image']
        image_decoded = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_decoded))

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            prediction = model(image_tensor)

        # Process the prediction as needed
        # For example, convert tensor to JSON serializable output
        prediction_result = prediction.argmax().item()

        return json.dumps({
            'statusCode': 200,
            'body': {'prediction': prediction_result}
        })
    except Exception as e:
        return json.dumps({
            'statusCode': 400,
            'body': f'{e}'
        })
