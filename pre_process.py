import h5py
import cv2
import numpy as np
from connectomics.utils.process import polarity2instance


def readh5(path):
    """Reads the h5 file in the specified path
    Args:
        path (str): path to .h5 file
    Returns:
        h5py.File object
    """
    data = h5py.File(path, 'r')
    vol = list(data)[0]
    return data[vol]


def label2instance (label):
    """
    Converts ground truth labels into instances

    Args:
        label (np.ndarray): ground truth label array

    Returns:
        np.ndarray: pre-processed ground truth array
    """
    tmp = [None]*3
    tmp[0] = np.array(np.logical_and((label % 2) == 1, label > 0), int) * 255
    tmp[1] = np.array(np.logical_and((label % 2) == 0, label > 0), int) * 255
    tmp[2] = np.array((label > 0), int) * 255
    return np.array(tmp)


def most_common(lst):
    """
    Utility function to find mode in a list

    Args:
        lst (list): list of elements 

    Returns:
        int: mode of the list
    """
    return max(set(lst), key=lst.count)


def dilate(seg_mask, idx, equal_idx, unequal_idx, shape):
    """
    Dilation function for each slice: 
     - Performs dilation using a 5x5 kernel
     - Takes intersection with the mask of the cell it resides in
     - Takes interseciton with the negative mask of the corresponding pre/post synaptic cell

    Args:
        seg_mask (np.ndarray): segmentation mask of the volume
        idx (np.ndarray): boolean array specifying where the synaptic region is present
        equal_idx (int): label of cell to which the synaptic region belongs to
        unequal_idx (int): label of corresponding pre/post synaptic cell
        shape (tuple): shape of slice

    Returns:
        np.ndarray: array containing [0,1]
    """
    tmp = np.zeros(shape)
    tmp[idx] = 1

    kernel = np.ones((5,5), np.uint8)
    tmp = cv2.dilate(tmp, kernel, iterations=1)
    tmp = tmp * (seg_mask == equal_idx) * (seg_mask != unequal_idx)

    return tmp


def process_slice(gt, seg_mask):
    """
    Processes each slice to perform dilation

    Args:
        gt (np.ndarray): ground truth label volume
        seg_mask (np.ndarray): segmentation mask volume

    Returns:
        np.ndarray: processed ground truth volume
    """
    gt_ids = np.unique(gt)
    final = np.zeros(gt.shape)

    i = 1
    while i < len(gt_ids):


        if (i+1 < len(gt_ids)) and ( gt_ids[i] % 2 == 1) and ( gt_ids[i] + 1 == gt_ids[i+1] ) :

            idx_pre = (gt == gt_ids[i])
            idx_post = (gt == gt_ids[i+1])

            cor_pre_seg_mask = most_common(list(seg_mask[idx_pre]))
            cor_post_seg_mask = most_common(list(seg_mask[idx_post]))

            tmp_pre = dilate(seg_mask, idx_pre, cor_pre_seg_mask, cor_post_seg_mask, gt.shape)
            tmp_post = dilate(seg_mask, idx_post, cor_post_seg_mask, cor_pre_seg_mask, gt.shape)

            final[tmp_pre > 0] = 1
            final[tmp_post > 0] = 2

            i += 2

        else: # case where singly pre/post synapse is present

            idx_pre = (gt == gt_ids[i])
            cor_pre_seg_mask = most_common(list(seg_mask[idx_pre]))
            tmp_pre = dilate(seg_mask, idx_pre, cor_pre_seg_mask, 0, gt.shape)

            if gt_ids[i] % 2 == 1:
                final[tmp_pre > 0] = 1
            else:
                final[tmp_pre > 0] = 2

            i += 1

    return final


gt = np.array(readh5('demo_data/vol3_syn_gt_v2.h5'))
gt = polarity2instance(label2instance(gt))

seg_mask = np.array(readh5('demo_data/vol3_seg_gt_v2.h5'))

final = np.zeros(gt.shape)

for z in range(gt.shape[0]):
    final[z] = process_slice(gt[z], seg_mask[z])

hf = h5py.File('processed_vol3.h5', 'w')
hf.create_dataset('dataset_1', data=final)
hf.close()