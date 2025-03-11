# CIFAR-10 Image Classification with Knowledge Distillation

This project implements a knowledge distillation approach for CIFAR-10 image classification, transferring knowledge from a complex teacher model (Shake-Shake network) to a more compact student model (ResNet variant).

## Project Overview

Knowledge distillation is a technique where a smaller model (student) learns not only from ground truth labels but also from the soft outputs of a larger pre-trained model (teacher). This approach often allows the student model to achieve performance close to the teacher model while being more efficient.

In this project:
1. We first train a powerful Shake-Shake teacher model on CIFAR-10
2. We then distill this knowledge to a smaller ResNet student model
3. The student model can be used to generate predictions for Kaggle submissions

## Model Architecture

### Teacher Model: Shake-Shake Network

The teacher model is a Shake-Shake network, which is a variant of ResNet with specialized regularization techniques:

- **Architecture**: Deep residual network with 26 layers
- **Feature Maps**: Starting with 64 base channels
- **Regularization**: Uses shake-shake regularization which stochastically combines the outputs of multiple branches
- **Shake-Forward**: Applies random weights during forward pass
- **Shake-Backward**: Applies random weights during backward pass
- **Shake-Image**: Applies shake regularization at the image level

### Student Model: Modified ResNet

The student model is a modified ResNet architecture that can be configured as ResNet-18, ResNet-34, or ResNet-50:

- **ResNet-18**: 2-2-2-2 block configuration with BasicBlock structure
- **ResNet-34**: 3-4-6-3 block configuration with BasicBlock structure
- **ResNet-50**: 3-4-6-3 block configuration with BottleneckBlock structure (more efficient parameterization)
- **Initial Channels**: The model starts with 32 channels for ResNet-18/34 and 24 for ResNet-50 to control parameter count
- **Parameter Count**: The student model is designed to have under 5 million parameters

## Training Techniques

### Teacher Model Training

- **Optimizer**: SGD with momentum and weight decay
- **Learning Rate Schedule**: Cosine annealing learning rate schedule
- **Data Augmentation**: Standard CIFAR-10 augmentations (random crop, horizontal flip)
- **Regularization**: Utilizes Cutout technique, which randomly masks out square regions of the input during training to improve generalization

### Knowledge Distillation Process

- **Temperature Scaling**: Uses temperature parameter (T=3.0) to soften probability distributions
- **Compound Loss**: Combines cross-entropy loss (hard targets) and KL divergence loss (soft targets)
- **Enhanced Student Augmentation**: The student model uses stronger data augmentation (color jitter, rotation)
- **Unlabeled Data Utilization**: Leverages unlabeled Kaggle test data through teacher's soft predictions
- **Mixed Batch Training**: Simultaneously trains on labeled CIFAR-10 data and unlabeled Kaggle test data

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- pandas
- matplotlib

### Installation

```bash
pip install torch torchvision numpy pandas matplotlib
```

### Running the Models

#### Step 1: Train the Teacher Model

```bash
python teacher_train.py --depth 26 --base_channels 64 --shake_forward True --shake_backward True --shake_image True --outdir teacher/pytorch_shake_shake/results
```

This will train the Shake-Shake teacher model and save the best model checkpoint to the specified output directory.

#### Step 2: Distill Knowledge to Student Model

```bash
python student_distillation.py
```

By default, this will:
- Load the trained teacher model
- Create a ResNet-50 student model
- Train the student using knowledge distillation
- Evaluate the model on the test set
- Generate a submission file for Kaggle

#### Step 3 (Optional): Generate Submission Only

If you already have a trained student model and only want to generate Kaggle submissions:

```bash
python student_distillation.py --predict_only
```

Additional options:
```bash
python student_distillation.py --predict_only --model_path custom_model.pth --test_data_path custom_test_data.pkl --output_file custom_submission.csv
```

## Implementation Details

### Data Pipeline

- Custom dataset classes handle both CIFAR-10 and Kaggle test data
- Separate transformations for teacher and student models enable different augmentation strategies
- Efficient data loading with multiple workers and pinned memory

### Distillation Loss

The distillation loss combines hard and soft targets:
```
total_loss = CE_loss + lambda_distill * KL_div_loss
```
Where:
- `CE_loss` is the standard cross-entropy with hard labels
- `KL_div_loss` is the KL divergence between soft predictions of teacher and student
- `lambda_distill` controls the balance between these losses

### Monitoring and Evaluation

- Tracks training and validation metrics throughout training
- Saves the best model based on validation accuracy
- Generates visualizations of training progress
- Provides detailed class distribution analysis for predictions

## Results

The student model achieves competitive accuracy compared to the teacher model while having significantly fewer parameters. The knowledge distillation approach helps the student model learn more effectively than training from scratch.