import h5py
import numpy as np
from tqdm import tqdm
from random import randint
import time
from connectomics.utils.process import polarity2instance
from skimage.measure import label


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


def process_gt(gt):
    """Pre-processing ground truth labels

    Args:
        gt (np.ndarray): Ground Truth label volumne

    Returns:
        np.ndarray: pre-processed ground truth volume
    """

    gt = gt[:100,:,:]

    c3 = np.zeros(gt.shape, int)
    c3[gt > 0] = 1
    c3 = label(c3)

    c2 = np.zeros(gt.shape, int)
    c2[gt == 2] = 1

    c1 = np.zeros(gt.shape, int)
    c1[gt == 1] = 1

    idx = c3 > 0
    c2[idx] = 2*c3[idx]*c2[idx]

    c1[idx] = (2*c3[idx] - 1)*c1[idx]

    fin = np.maximum(c1,c2)

    return fin


def seg_bbox3d(seg, ids):
    """Generates bounding box for each instance in the given volumne

    Args:
        seg (np.ndarray): volumne
        ids (np.array): array containing the labels present in seg

    Returns:
        np.ndarray: For each instance, the bounding box coordinates are present
                    (z_min, z_max, y_min, y_max, x_min, x_max)
    """

    seg_size = seg.shape
    
    # Finding the number of instances
    um = int(ids.max())

    # Initialising array for storing bounding box for each instance
    out = np.zeros((1+um,7),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1], out[:,3], out[:,5] = seg_size[0], seg_size[1], seg_size[2]

    # Identifying slices which contain atleast 1 instance
    zids = np.where((seg>0).sum(axis=1).sum(axis=1)>0)[0]
    # For each such slice
    for zid in zids:
        # Unique instances in that slice
        sid = np.unique(seg[zid])
        # Finding 'valid' instances
        sid = sid[(sid>0)*(sid<=um)]
        # Finding minimum and maximum z where that instance occurs
        out[sid,1] = np.minimum(out[sid,1], zid)
        out[sid,2] = np.maximum(out[sid,2], zid)

    # Similarly with rows
    rids = np.where((seg>0).sum(axis=0).sum(axis=1)>0)[0]
    for rid in rids:
        sid = np.unique(seg[:,rid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,3] = np.minimum(out[sid,3],rid)
        out[sid,4] = np.maximum(out[sid,4],rid)
    
    # Similarly with columns
    cids = np.where((seg>0).sum(axis=0).sum(axis=0)>0)[0]
    for cid in cids:
        sid = np.unique(seg[:,:,cid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,5] = np.minimum(out[sid,5],cid)
        out[sid,6] = np.maximum(out[sid,6],cid)

    return out[ids]


def seg_iou3d(pred, gt ):
    """
    Computes bounding boxes for each prediction, 
    Calculates IoU with the ground truth in that bounding box, 
    and finds the best match among them

    Args:
        pred (np.ndarray): predicted volumne. Size = (Z,Y,X) 
        gt (np.ndarray): ground truth labels. Size = (Z,Y,X)

    Returns:
        (Matching Pairs of ground truth IDs and Prediction IDs, IoU of each pair, Prediction IDs which didn't match)
    """

    tp = fn = fp = 0.0

    # Finding the instances present in the prediction as well as their count
    pred_id, pred_id_count = np.unique(pred, return_counts=True)
    pred_id_count = pred_id_count[pred_id>0]
    pred_id = pred_id[pred_id>0]
    
    # Finding the instances present in the ground truth as well as their count
    gt_id, gt_id_count = np.unique(gt,return_counts=True)
    gt_id_count = gt_id_count[gt_id>0]
    gt_id = gt_id[gt_id>0]

    start = time.time()
    # Gives the bounding box for each instance
    bbx = seg_bbox3d(pred, pred_id)[:,1:]   
    print('Computed bounding boxes',time.time()-start,'s')

    not_matched_id = []
    gt_matched_id = np.zeros(1+gt_id.max(), int)
    gt_matched_iou = np.zeros(1+gt_id.max(), float)

    start = time.time()

    for j,i in enumerate(pred_id):
    
        # Bounding box of that instance
        bb = bbx[j]
    
        # Finding intersection of predicted and ground truth instance inside the bounding box, along with the count
        match_id, match_id_count = np.unique( gt[ bb[0]:bb[1]+1, bb[2]:bb[3]+1 ] * ( pred[ bb[0]:bb[1]+1, bb[2]:bb[3]+1 ] == i ), return_counts = True )

        match_id_count = match_id_count[match_id>0] # Intersection counts
        match_id = match_id[match_id>0]             # Intersection ids
        
        if len(match_id) > 0: # if there is an intersecting label

            # Get all possible pairs inside bounding box 
            gt_id_count_match = gt_id_count[np.isin(gt_id, match_id)]
            # All possible IoUs are calculated
            ious = match_id_count.astype(float)/(pred_id_count[j] + gt_id_count_match - match_id_count)
            
            # Update the IoUs and matching pairs
            idx = gt_matched_iou[match_id] < ious            
            gt_matched_iou[match_id[idx]] = ious[idx]
            gt_matched_id[match_id[idx]] = i
        
        else: # if it matches no ground truth instance then it is a false positive
            not_matched_id.append(i)

    print('Computed matching pairs',time.time()-start,'s')

    return gt_matched_id, gt_matched_iou, not_matched_id


def eval_metrics(gt_matched_id, gt_matched_iou, not_matched_id, thres = 0.3):
    """Calculates the classification metrics (F1 score, Precision, Recall)

    Args:
        gt_matched_id (np.ndarray): Matching Pairs of ground truth IDs and Prediction IDs
        gt_matched_iou (np.ndarray): IoU of each pair
        not_matched_id (list): Prediction IDs which didn't match
        thres (float, optional): Threshold value for IoU. Defaults to 0.3.

    Returns:
        dict: classification metrics and list of true positives, false positives and false negatives
    """

    false_positives, false_negatives, true_positives = [], [], [] 

    start = time.time()

    i = 0
    while i < len(not_matched_id):
        # Identifying pairs of synaptic clefts which were false positives
        if not_matched_id[i] + 1 == not_matched_id[i+1]:
            false_positives.append((not_matched_id[i], not_matched_id[i+1]))
            i += 2
        # Singly predicted pre/post synaptic cleft
        else:
            false_positives.append((not_matched_id[i], -1))
            i += 1

    for i in range(1,len(gt_matched_id), 2):

        k = (i+1) // 2

        # Ground Truth IDs which didn't match with any Prediction ID
        if gt_matched_id[ 2*k-1 ] == 0 or gt_matched_id[ 2*k ] == 0:
            false_negatives.append((2*k-1, 2*k, 0, 0))
            continue

        pre_iou = gt_matched_iou[2*k-1]
        post_iou = gt_matched_iou[2*k]

        if pre_iou > thres and post_iou > thres:
            true_positives.append((2*k-1, 2*k, pre_iou, post_iou))      
        else:
            false_negatives.append((2*k-1, 2*k, pre_iou, post_iou))
   
    len_tp = len(true_positives)
    len_fp = len(false_positives)
    len_fn = len(false_negatives)

    precision = len_tp / (len_tp + len_fp)
    recall = len_tp / (len_tp + len_fn)

    f1 = 2*precision*recall / ( precision + recall )

    print('Computed evaluation metrics',time.time()-start,'s')

    return {
        'threshold' : thres,
        'f1'        : f1,
        'precision' : precision,
        'recall'    : recall,
        'tp'        : true_positives,
        'fp'        : false_positives,
        'fn'        : false_negatives
    }


def main():

    start = time.time()

    gt = readh5('demo_data/vol3_syn_gt_v2.h5')
    gt = process_gt(np.array(gt))

    pred = readh5('demo_data/result_vol3.h5')
    pred = np.array(pred)
    
    pred = polarity2instance(pred)

    print('Loading and pre-processing data', time.time()-start,'s')

    match_id, match_iou, not_match = seg_iou3d(pred, gt)
    eval_data = eval_metrics(match_id, match_iou, not_match, thres=0.3)

    print(eval_data)

if __name__=='__main__':
    main()
