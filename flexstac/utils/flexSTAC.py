from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, detection_collate
from data import VOC_CLASSES as labelmap

from utils.augmentations import SSDAugmentation, SSDAugmentation_soft, SSDAugmentation_hard
from layers.modules import MultiBoxLoss
from ssd512 import build_ssd
import numpy as np
import time, pickle, random, logging
from utils import augmentations
from collections import Counter

def prep_unsupervised_data(
    voc_root,
    batch_iterator,
    model_path,
    ssd_dim,
    means,
    num_classes,
    detection_thresh=0.9,
    min_detection_thresh=0.65,
    use_dynamic_thresh=True,
):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = build_ssd('test', ssd_dim, num_classes=num_classes)
    net.load_state_dict(torch.load(model_path))
    
    images_found = 0
    aug_image_count = 0
    i = 0
    ssod_labels = []
    ssod_images = []
    detections_array = []
    classes_found = []

    for images, labels in batch_iterator:
        if i >= 10000:
            break
        for im, lab in zip(images, labels):
            i += 1
            print("================================================================================================", flush=True)
            print("Image {}".format(i), flush=True)
            x = Variable(im.unsqueeze(0))
            x = x.cuda()
            detections = net(x).data
            detections_array.append((im, detections.clone().detach()))
            
            ssod_label_tmp_id = 0
            for k in range(detections.size(1)):
                # k here is for class. Note that k=0 is background class and score is always 0
                l = 0
                while detections[0,k,l,0] >= detection_thresh:
                    score = detections[0,k,l,0]
                    ssod_label_tmp = torch.cat((detections[0,k,l,1:],torch.tensor([float(k-1)]))).unsqueeze(0)
                    classes_found.append(float(k-1))
                    if ssod_label_tmp_id == 0:
                        ssod_label = ssod_label_tmp
                    else:
                        ssod_label = torch.cat([ssod_label, ssod_label_tmp], dim=0)
                    l+=1
                    ssod_label_tmp_id += 1

            if not use_dynamic_thresh:
                if ssod_label_tmp_id > 0:
                    ssod_images.append(im.unsqueeze(0))
                    ssod_labels.append(ssod_label)
                    images_found += 1
                    print("{} images labeled by teacher model".format(images_found), flush=True)

                    ## data augmentation
                    im_aug = im.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
                    bbox_aug = ssod_label[:, :4].detach().cpu().numpy().astype(np.float32)
                    label_aug = np.reshape(ssod_label[:, 4].detach().cpu().numpy().astype(np.float32), (-1, 1))

                    augmenter = SSDAugmentation_hard(ssd_dim, means)
                    im_aug, bbox_aug, label_aug = augmenter(im_aug, bbox_aug, label_aug)

                    if len(label_aug) > 0:
                        print("{} augmented images generated".format(aug_image_count), flush=True)
                        aug_image_count += 1
                        im_aug = torch.from_numpy(im_aug.astype(np.float32)).permute(2, 0, 1)
                        bbox_label_aug = np.append(bbox_aug, label_aug, 1)
                        bbox_label_aug = torch.from_numpy(bbox_label_aug.astype(np.float32))

                        ssod_images.append(im_aug.unsqueeze(0))
                        ssod_labels.append(bbox_label_aug)
                        
    print("Initial class distribution of pseudo-labels:", [(labelmap[int(k)], v) for k, v in Counter(classes_found).items()], flush=True)
    
    dynamic_classes_found = []
    if use_dynamic_thresh:
        classes_found_counter = Counter(classes_found)
        max_class_count = sorted(list(classes_found_counter.values()))[-2] # person class is too dominant, so excluding
        dynamic_thresholds = np.array([1.0] + [classes_found_counter[class_]/max_class_count if class_ in classes_found_counter 
         else min(classes_found_counter.values())/max_class_count
         for class_ in range(20)]) * detection_thresh
        dynamic_thresholds = [min(max(dynamic_threshold, min_detection_thresh), 0.95) for dynamic_threshold in dynamic_thresholds]
        print("Dynamic thresholds:", [(d_t, d_l) for d_t, d_l in zip(dynamic_thresholds[1:], labelmap)], flush=True)

        images_found = 0
        aug_image_count = 0
        ssod_labels = []
        ssod_images = []
        i = 0
        for abc in [0]:
            for im, detections in detections_array:
                i += 1
                print("{} images reviewed".format(i), flush=True)
                ssod_label_tmp_id = 0
                for k in range(detections.size(1)):
                    # k here is for class. Note that k=0 is background class and score is always 0
                    l = 0
                    while detections[0,k,l,0] >= dynamic_thresholds[k]:
                        dynamic_classes_found.append(float(k-1))
                        score = detections[0,k,l,0]
                        ssod_label_tmp = torch.cat((detections[0,k,l,1:],torch.tensor([float(k-1)]))).unsqueeze(0)
                        if ssod_label_tmp_id == 0:
                            ssod_label = ssod_label_tmp
                        else:
                            ssod_label = torch.cat([ssod_label, ssod_label_tmp], dim=0)
                        l+=1
                        ssod_label_tmp_id += 1

                if ssod_label_tmp_id > 0:
                    ssod_images.append(im.unsqueeze(0))
                    ssod_labels.append(ssod_label)
                    images_found += 1
                    print("{} images labeled by teacher model".format(images_found), flush=True)

                    ## data augmentation
                    im_aug = im.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
                    bbox_aug = ssod_label[:, :4].detach().cpu().numpy().astype(np.float32)
                    label_aug = np.reshape(ssod_label[:, 4].detach().cpu().numpy().astype(np.float32), (-1, 1))

                    augmenter = SSDAugmentation_hard(ssd_dim, means)
                    im_aug, bbox_aug, label_aug = augmenter(im_aug, bbox_aug, label_aug)

                    if len(label_aug) > 0:
                        aug_image_count += 1
                        print("{} augmented images generated".format(aug_image_count), flush=True)
                        im_aug = torch.from_numpy(im_aug.astype(np.float32)).permute(2, 0, 1)
                        bbox_label_aug = np.append(bbox_aug, label_aug, 1)
                        bbox_label_aug = torch.from_numpy(bbox_label_aug.astype(np.float32))

                        ssod_images.append(im_aug.unsqueeze(0))
                        ssod_labels.append(bbox_label_aug)
                    
        print("Final class distribution of pseudo-labels:", [(labelmap[int(k)], v) for k, v in Counter(dynamic_classes_found).items()], flush=True)
                
    return ssod_images, ssod_labels


def get_smaller_set(dataset, max_cat_thresh=20, min_cat_thresh=15, categories_count=20):
    keep_indices = []
    labels = list(range(0,categories_count))
    labels_dict = {l: 0 for l in labels}

    for ix, (i, j) in enumerate(dataset):
        keep = True
        for val in j[:, -1]:
            if labels_dict[(int(val))] >= max_cat_thresh:
                keep = False
                break
        if keep:
            for val in j[:, -1]:
                labels_dict[(int(val))] += 1
            keep_indices.append(ix)

        if min(labels_dict.values()) >= min_cat_thresh:
            break
            
    return keep_indices, labels_dict

def labels_check(dataset, categories_count=20):
    labels = list(range(0,categories_count))
    labels_dict = {l: 0 for l in labels}

    for ix, (i, j) in enumerate(dataset):
        for val in j[:, -1]:
            labels_dict[(int(val))] += 1
        
    return labels_dict
