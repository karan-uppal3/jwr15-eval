# JWR15-eval

## Detailed steps:
- The bounding box of each instance in the predicted volume is found ([seg_bbox3d](https://github.com/karan-uppal3/jwr15-eval/blob/6e06001f94270e9ef248286a1c9a7a7413c8c446/demo.py#L54))
- For each instance of the predicted volume, the intersection of all possible ground truth labels is found within that bounding box. IoU is calculated with each possible pair and the list of matching pairs of ground truth IDs and prediction IDs is updated based on the calculated IoU values. A separate list of prediction IDs that didn’t match is also maintained ([seg_iou3d](https://github.com/karan-uppal3/jwr15-eval/blob/6e06001f94270e9ef248286a1c9a7a7413c8c446/demo.py#L107))
- Calculating classification metrics: ([eval_metrics](https://github.com/karan-uppal3/jwr15-eval/blob/6e06001f94270e9ef248286a1c9a7a7413c8c446/demo.py#L173))
  - If an instance pair (corresponding pre and post synaptic cleft) doesn’t match with any ground truth instance pair, then it is counted as a false positive
  - If an instance pair matches with a ground truth instance pair and has IoU value above the specified threshold for both of them, then it is counted as a true positive
  - If a ground truth instance pair doesn’t match with an instance pair from the predicted volume or doesn’t meet the threshold, then it is counted as a false negative
  - Using the above values, precision, recall and F1-score are calculated 

## Time taken

|               Task              |   Time (in s.)  |
|:-------------------------------:|:---------------:|
| Loading and pre-processing data |     7.415350461 |
| Computed bounding boxes         |     1.877528429 |
| Computed matching pairs         |    0.9498236179 |
| Computed evaluation metrics     | 0.0003816127777 |

Note: The values are averaged over 10 runs
