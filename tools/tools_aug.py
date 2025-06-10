import os
import albumentations as A

from PIL import Image
import numpy as np

def make_abs_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def RGBShift(image:Image):
    transform = A.RGBShift(p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

def HorizontalFlip(image:Image):
    transform = A.HorizontalFlip(p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

def VerticalFlip(image:Image):
    transform = A.VerticalFlip(p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

def Rotate(image:Image):
    transform = A.Rotate(p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

def BrightnessContrast(image:Image):
    transform = A.RandomBrightnessContrast(p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

def ResizedCrop(image:Image):
    transform = A.RandomResizedCrop(size=(640, 640), p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

def ColorJit(image:Image):
    transform = A.ColorJitter(p=1.0)
    img_trans = transform(image=np.array(image))
    return img_trans["image"]

if __name__ == "__main__":
    for image_name in ["algal_leaf_spot_0_original.jpg", "leaf_spot_0_original.jpg", "leaf_blight_0_original.jpg", "no_disease_0_original.jpg"]:
        image_path = make_abs_path(fr"aug_result\\{image_name}")
        image = Image.open(image_path).convert("RGB")
        # RGBShift
        img_rgbshift = Image.fromarray(RGBShift(image))
        img_rgbshift.save(image_path.replace("0_original", "1_rgbshift"))
        # Flip
        img_hflip = Image.fromarray(HorizontalFlip(image))
        img_hflip.save(image_path.replace("0_original", "2_hflip"))
        img_vflip = Image.fromarray(VerticalFlip(image))
        img_vflip.save(image_path.replace("0_original", "3_vflip"))
        # Rotate
        img_rotate = Image.fromarray(Rotate(image))
        img_rotate.save(image_path.replace("0_original", "4_rotate"))
        # Brightness
        img_brightness = Image.fromarray(BrightnessContrast(image))
        img_brightness.save(image_path.replace("0_original", "5_brightness"))
        # ResizedCrop
        img_rcrop = Image.fromarray(ResizedCrop(image))
        img_rcrop.save(image_path.replace("0_original", "6_rcrop"))
        # ColorJit
        img_colorjit = Image.fromarray(ColorJit(image))
        img_colorjit.save(image_path.replace("0_original", "7_colorjit"))
