# Demo pipeline for Sea Ice Detection (U-Net based)

import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image

def dummy_unet_model(image):
    # Placeholder for actual model prediction
    mask = np.zeros((256, 256))
    return mask

def main():
    image = preprocess_image("input.png")
    mask = dummy_unet_model(image)
    print("Segmentation completed")

if __name__ == "__main__":
    main()