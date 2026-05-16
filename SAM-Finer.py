import os
from tkinter.tix import MAIN
import torch
import numpy as np
import time
import random
from tqdm import tqdm
from torch.backends import cudnn
import torch.nn.functional as F
import time
import cv2
from scipy.ndimage import zoom
import argparse
import tifffile as tiff
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

_CLASS_NAMES_ = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_topn_bounding_boxes(binary_image, top_n=3):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        top_indices = np.argsort(areas)[-top_n:] + 1
    else:
        return []
    bounding_boxes = []
    for idx in top_indices:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        category_id = 1
        is_crowd = 0
        bounding_boxes.append([x, y, x + w, y + h])
    return bounding_boxes

def fill_holes(image):
    num_labels, labels = cv2.connectedComponents(image)
    mask = np.zeros_like(image)
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    filled_image = cv2.bitwise_or(image, mask)
    return filled_image

def samfiner(args, test_type):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint="backbones/sam_vit_h_4b8939.pth") #backbones/weights/sam_vit_b_01ec64.pth
    predictor = SamPredictor(sam.to(device))
    dataset_dir = args.data_path
    save_dir = args.bin_savedir
    binary_dir = os.path.join(args.bin_savedir, 'anomaly_images_thresholded')
    save_dir = os.path.join(save_dir, 'anomaly_images_thresholded')
    for class_name in _CLASS_NAMES_:
        print(class_name)
        tag = True
        setup_seed(1)
        dataset_class_dir = os.path.join(dataset_dir, class_name, test_type)
        image_path_list = []
        if test_type == 'test_public':
            image_name_list = os.listdir(os.path.join(dataset_class_dir, 'bad'))
            image_path_list.extend([os.path.join(dataset_class_dir, 'bad', x) for x in image_name_list])
            image_name_list = os.listdir(os.path.join(dataset_class_dir, 'good'))
            image_path_list.extend([os.path.join(dataset_class_dir, 'good', x) for x in image_name_list])
        else:
            image_name_list = os.listdir(dataset_class_dir)
            image_path_list.extend([os.path.join(dataset_class_dir, x) for x in image_name_list])
        image_path_list = sorted(image_path_list)
        for image_path in tqdm(image_path_list):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            binary_mask_path = image_path.replace(dataset_dir, binary_dir)
            binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
            zoom_factors = (binary_mask.shape[0] / image.shape[0], binary_mask.shape[1] / image.shape[1], 1)
            image = zoom(image, zoom_factors, order=1)
            top_n = 3
            bbox_list = get_topn_bounding_boxes(binary_mask, top_n=top_n)
            if len(bbox_list) == 0:
                masks = np.zeros_like(binary_mask)
            else:
                predictor.set_image(image)
                mask_list = []
                for bbox in bbox_list:
                    box_prompt = np.array(bbox)
                    masks, _, _ = predictor.predict(box=box_prompt, return_logits=False, multimask_output=tag)
                    masks = masks.sum(0)
                    masks[masks > 0] = 255
                    masks = masks.astype(np.uint8)
                    mask_list.append(masks)
                masks = np.array(mask_list).max(0)
                masks = fill_holes(masks)
            save_path = binary_mask_path.replace(binary_dir, save_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, masks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'./mvtec_ad_2/')
    parser.add_argument('--bin_savedir', type=str, default=r'./results')
    parser.add_argument('--test_type', type=str, default=r'challenge')
    args = parser.parse_args()
    if args.test_type != 'challenge':
        samfiner(args, test_type='test_public')
    else:
        samfiner(args, test_type='test_private')
        samfiner(args, test_type='test_private_mixed')

