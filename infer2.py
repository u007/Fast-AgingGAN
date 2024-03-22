import os
import random
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator

parser = ArgumentParser()
parser.add_argument('--image_dir', default='img/', help='The image directory')

@torch.no_grad()
def main():
    args = parser.parse_args()
    if not os.path.exists(args.image_dir):
        print("The specified image directory does not exist.")
        return
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if x.endswith(('.png', '.jpg'))]
    if len(image_paths) == 0:
        print("No suitable images found in the directory.")
        return
    print(f'Found {len(image_paths)} images')

    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    random.shuffle(image_paths)
    for i, path in enumerate(image_paths):
        print(f'Processing image {i + 1}/{len(image_paths)}')
        img = Image.open(path).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

        # Prepare the save path with '_aged' suffix before the extension
        base, ext = os.path.splitext(path)
        aged_path = f"{base}_aged{ext}"

        # Convert aged_face numpy array back to an image and save
        aged_image = Image.fromarray((aged_face * 255).astype('uint8'))
        aged_image.save(aged_path)

if __name__ == '__main__':
    main()
