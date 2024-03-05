# Custom transform for inverting image colors
from PIL import Image
from torchvision.transforms import transforms


class InvertColours(object):
    def __call__(self, img):
        return Image.eval(img, lambda x: 255 - x)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    InvertColours(),  # change white to black, black to white.
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
