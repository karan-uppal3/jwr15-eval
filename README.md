# JWR15-pre-process

## Detailed steps:
- Process the ground truth volume
- For each slice:
  - For each label, find the neuron mask it resides in
  - For each pre-synaptic region, conduct 2d dilation, then calculate the intersection with both the mask it resides in and the negative mask of the post-synaptic compartment
  - Perform the same for post-synaptic region
  - In case of singly pre/post synaptic region, perform usual dilation with intersection with the mask it resides in
