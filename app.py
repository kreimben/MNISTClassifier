import base64
import json
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

# Load model globally
model = torch.jit.load('scripted_model.pt')
model.eval()

# Define your transform pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def lambda_handler(event, context):
    try:
        # Decode the image
        image_data = event['body']
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

        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': prediction_result})
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': e
        }
