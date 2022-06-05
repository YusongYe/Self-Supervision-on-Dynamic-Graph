import numpy as np
import random


def mask_node(features,mask_ratio):
    total_num = features.shape[0]
    mask_num = int(total_num*mask_ratio)
    randomlist = random.sample(range(0, total_num), mask_num)
    for node in randomlist:
        features[node] = [10.0]*features.shape[1]
    return features
    