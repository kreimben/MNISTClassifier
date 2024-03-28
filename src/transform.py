from PIL import Image
from torchvision.transforms import transforms, AutoAugmentPolicy, AutoAugment


class InvertColours(object):
    def __call__(self, img):
        return Image.eval(img, lambda x: 255 - x)


inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    InvertColours(),  # change white to black, black to white.
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

training_transforms = transforms.Compose([
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # Example values for MNIST
])

