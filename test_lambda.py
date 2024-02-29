import base64
import json
import random

from icecream import ic
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from app import lambda_handler


def test1():
    dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    image, label = dataset[random.randint(0, len(dataset) - 1)]
    image = to_pil_image(image)

    image.save('test_image.png')

    def encode_image_to_base64(image_path='test_image.png'):
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

    image_base64 = encode_image_to_base64('test_image.png')

    body = {
        'image': image_base64
    }

    event = {
        "httpMethod": "POST",
        'body': body
    }
    print(json.dumps(body))

    res = lambda_handler(event, None)

    ic(res)
    print(f'Actual label was: {label}')

    status_code = res['statusCode']
    prediction = json.loads(res['body'])['prediction']

    assert status_code == 200
    assert prediction == label


def test2():
    with open("test_data.json", 'r') as f:
        body = json.loads(f.read())

    event = {
        'httpMethod': 'POST',
        'body': body
    }
    print(json.dumps(body))

    res = lambda_handler(event, None)

    ic(res)
    print(f'Actual label was: {9}')

    status_code = res['statusCode']
    prediction = json.loads(res['body'])['prediction']

    assert status_code == 200
    assert prediction == 9


if __name__ == '__main__':
    test1()
    test2()
