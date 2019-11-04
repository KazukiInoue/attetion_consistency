import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


def save_numpy_image(img_np, save_path):
    img_np = img_np.astype(np.uint8)

    img_pil = Image.fromarray(img_np)
    img_pil.save(save_path, 'JPEG', quality=100)


if __name__ == '__main__':

    path = './cat.jpeg'
    pimg = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(pimg)

    mask = torch.linspace(-1, 1, 16).view(1, 1, 4, 4)

    upsample = nn.Upsample(size=(img.size()[1], img.size()[2]))

    mask = upsample(mask).squeeze(0).squeeze(0)

    img_np = img.cpu().float().numpy()
    img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0

    mask_np = mask.cpu().float().numpy()
    mask_np = (mask_np + 1) / 2.0 * 255.0
    mask_color = cv2.applyColorMap(mask_np.astype(np.uint8), cv2.COLORMAP_JET)
    mask_color = mask_color[:, :, ::-1]

    cam_np = (img_np + mask_color) / 2.0

    save_numpy_image(img_np, './cat_recon.jpg')
    save_numpy_image(mask_np, './mask_gray.jpg')
    save_numpy_image(mask_color, './mask_color.jpg')
    save_numpy_image(cam_np, './cat_cam.jpg')
