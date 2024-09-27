# LEGACY: A LIGHTWEIGHT ADAPTIVE GRADIENT COMPRESSION STRATEGY FOR DISTRIBUTED LEARNING

## Overview

This repository contains the implementation of LEGACY adaptive gradient compression strategies described in the accompanying paper.

## Resources

The gradient compression implementations provided here are based on the following papers:

- AAAI 2020: "On the Discrepancy between Theoretical Analysis and Practical Implementations of Compressed Communication for Distributed Deep Learning"
- NeurIPS 2021: "Rethinking Gradient Sparsification as Total Error Minimization"

Refer to the `compressors` folder to review the modifications applied to each implementation. To use these compressors in distributed learning, you simply need to invoke the compressor function before the optimization step. The compressor will handle the compression and aggregation of gradients.

## Requirements and Data Preparation

Please refer to the original implementations for detailed setup and data preparation guidelines:

+ NCF & Transformer: [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch)

+ Resnet50: [PyTorch Examples](https://github.com/pytorch/examples/tree/main/imagenet)

+ Reset9 & Alexnet: [On the Discrepancy between the Theoretical Analysis and Practical Implementations of Compressed Communication for Distributed Deep Learning](https://github.com/sands-lab/layer-wise-aaai20)

+ Resnet18: [Rethinking Gradient Sparsification as Total error minimization](https://github.com/sands-lab/rethinking-sparsification)
## Training

### Arguments
The execution follows the pattern from the original implementation with two additional arguments:

- **group_compressions**: A comma-separated list of compression ratios to use for each group. For uniform compression, only the first value is used. For the epoch-based strategy, the total training time is divided according to the number of specified ratios.

- **group_splits**:  A comma-separated list of the thresholds used for creating groups.

