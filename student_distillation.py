import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from cifar10_resnet import ModifiedResNet, BasicBlock, BottleneckBlock, create_resnet, count_parameters
from PIL import Image
from shake_shake import Network
import pandas as pd
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
TEMPERATURE = 3.0           # Temperature parameter T, used to soften logits
LAMBDA_DISTILL = 1        # Distillation loss weight
MODEL_TYPE = 'resnet50'     # Student model type: 'resnet18', 'resnet34', 'resnet50'

class my_CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, download=True, transform_teacher=None, transform_student=None):
        super(my_CIFAR10Dataset, self).__init__(root=root, train=train, download=download, transform=None)
        self.transform_teacher = transform_teacher
        self.transform_student = transform_student
    
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        if self.transform_teacher:
            image_teacher = self.transform_teacher(image)
        else:
            image_teacher = image
        if self.transform_student:
            image_student = self.transform_student(image)
        else:
            image_student = image
        return image_teacher, image_student, target
    
# Kaggle test dataset class
class KaggleDataset(Dataset):
    def __init__(self, data, ids, transform_teacher=None, transform_student=None):
        self.data = data
        self.ids = ids
        self.transform_teacher = transform_teacher
        self.transform_student = transform_student
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # CIFAR-10 format processing
        image = self.data[idx]
        id_value = self.ids[idx]
        
        # Determine processing method based on the shape of test data
        if len(image.shape) == 1:  # If it's a flattened one-dimensional array
            # Reshape one-dimensional array (3072,) to (32, 32, 3)
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        
        # Ensure image is uint8 type
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        image_pil = Image.fromarray(image)
        
        # Always apply transformations to ensure we return tensors, not PIL images
        if self.transform_teacher:
            image_teacher = self.transform_teacher(image_pil)
        else:
            # If no transform is provided, use a basic ToTensor transform
            image_teacher = transforms.ToTensor()(image_pil)
            
        if self.transform_student:
            image_student = self.transform_student(image_pil)
        else:
            # If no transform is provided, use a basic ToTensor transform
            image_student = transforms.ToTensor()(image_pil)
            
        return image_teacher, image_student, id_value

# Load teacher model
def load_teacher_model(model_path):
    # Configure shake_shake model
    config = {
        'input_shape': (1, 3, 32, 32),
        'n_classes': 10,
        'base_channels': 64,
        'depth': 26,
        'shake_forward': True,
        'shake_backward': True,
        'shake_image': True
    }
    
    # Create model instance
    model = Network(config)
    
    # Load pre-trained weights
    checkpoint = torch.load(model_path, weights_only=False)
    
    # checkpoint contains multiple keys, we need to extract the state_dict part
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Freeze teacher model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model

# Get data loaders
def get_data_loaders(batch_size=128, include_kaggle_test=True):

    transform_train_teacher = transforms.Compose([   
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_train_student = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load CIFAR-10 training and test sets
    trainset = my_CIFAR10Dataset(
        root='./data', train=True, download=True, transform_teacher=transform_train_teacher, transform_student=transform_train_student
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # If needed, load Kaggle test set
    kaggle_loader = None
    if include_kaggle_test:
        try:
            # Load Kaggle test data
            with open('./data/cifar_test_nolabel.pkl', 'rb') as f:
                test_data_dict = pickle.load(f)
            
            test_images = test_data_dict[b'data']
            test_ids = test_data_dict[b'ids']
            
            print(f"Kaggle test data shape: {test_images.shape}")
            print(f"Number of IDs: {len(test_ids)}")
            
            # Create Kaggle test dataset and loader
            kaggle_dataset = KaggleDataset(test_images, test_ids, transform_teacher=transform_train_teacher, transform_student=transform_train_student)
            kaggle_loader = DataLoader(
                kaggle_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
            )
        except Exception as e:
            print(f"Failed to load Kaggle test data: {e}")
            print("Will continue training without Kaggle test data")
    
    return trainloader, testloader, kaggle_loader

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature):
    """
    Calculate KL divergence loss between student and teacher models
    
    Args:
        student_logits: Logits output by the student model
        teacher_logits: Logits output by the teacher model
        temperature: Temperature parameter T, used to soften probability distributions
    
    Returns:
        KL divergence loss, multiplied by T^2 to compensate for gradient scaling
    """
    # Soften teacher model logits and calculate probability distribution
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    # Soften student model logits and calculate log probabilities
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    # Calculate KL divergence, multiplied by T^2 to compensate for gradient scaling
    loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    return loss_kl

# Train student model
def train_student(student_model, teacher_model, trainloader, testloader, kaggle_loader=None, 
                 epochs=100, lr=0.1, weight_decay=5e-4, temperature=3.0, lambda_distill=0.5):
    """
    Train student model through knowledge distillation
    
    Args:
        student_model: Student model
        teacher_model: Teacher model
        trainloader: CIFAR-10 training data loader
        testloader: CIFAR-10 test data loader
        kaggle_loader: Kaggle test data loader (no labels)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        temperature: Temperature parameter T
        lambda_distill: Distillation loss weight
    
    Returns:
        Trained student model and training history
    """
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    student_model.to(device)
    best_acc = 0.0
    train_losses = []
    train_accs = []
    
    # If there's no Kaggle test data, only train on CIFAR-10
    kaggle_training = kaggle_loader is not None
    
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss_train = 0.0
        running_kd_loss_kaggle = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Prepare Kaggle data iterator
        if kaggle_training:
            kaggle_iter = iter(kaggle_loader)
        
        for batch_idx, (inputs_teacher, inputs_student, targets) in enumerate(trainloader):
            inputs_teacher, inputs_student, targets = inputs_teacher.to(device), inputs_student.to(device), targets.to(device)
            
            # If using Kaggle data, get next batch
            if kaggle_training:
                try:
                    kaggle_inputs_teacher, kaggle_inputs_student, kaggle_ids = next(kaggle_iter)       
                    kaggle_inputs_teacher = kaggle_inputs_teacher.to(device)
                    kaggle_inputs_student = kaggle_inputs_student.to(device)
                except StopIteration:
                    kaggle_iter = iter(kaggle_loader)
                    kaggle_inputs_teacher, kaggle_inputs_student, kaggle_ids = next(kaggle_iter)       
                    kaggle_inputs_teacher = kaggle_inputs_teacher.to(device)
                    kaggle_inputs_student = kaggle_inputs_student.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - CIFAR-10 training data
            student_outputs = student_model(inputs_student)
            
            # Calculate cross entropy loss
            loss_ce = criterion_ce(student_outputs, targets)
            running_ce_loss += loss_ce.item()
            
            # Get soft labels from teacher model
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs_teacher)
            
            # Calculate distillation loss for CIFAR-10 training data
            loss_kd_train = distillation_loss(student_outputs, teacher_outputs, temperature)
            running_kd_loss_train += loss_kd_train.item()
            
            # Initialize total loss
            total_loss = loss_ce + lambda_distill * loss_kd_train
            
            # If using Kaggle data, calculate additional distillation loss
            if kaggle_training:
                # Student model output on Kaggle data
                student_outputs_kaggle = student_model(kaggle_inputs_student)
                
                # Teacher model output on Kaggle data
                with torch.no_grad():
                    teacher_outputs_kaggle = teacher_model(kaggle_inputs_teacher)
                
                # Calculate distillation loss for Kaggle data
                loss_kd_kaggle = distillation_loss(student_outputs_kaggle, teacher_outputs_kaggle, temperature)
                running_kd_loss_kaggle += loss_kd_kaggle.item()
                
                # Add Kaggle data distillation loss to total loss
                total_loss += lambda_distill * loss_kd_kaggle
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Track accuracy
            running_loss += total_loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Display progress
            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(trainloader)} | '
                      f'Loss: {running_loss/(batch_idx+1):.4f} | '
                      f'CE Loss: {running_ce_loss/(batch_idx+1):.4f} | '
                      f'KD Loss Train: {running_kd_loss_train/(batch_idx+1):.4f} | '
                      f'KD Loss Kaggle: {running_kd_loss_kaggle/(batch_idx+1) if kaggle_training else 0:.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # Calculate training statistics
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} completed | Training loss: {train_loss:.4f} | Training accuracy: {train_acc:.2f}% | Time: {epoch_time:.2f} seconds')
        
        # Evaluate on test set
        test_acc = evaluate(student_model, testloader)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student_model.state_dict(), 'best_student_model.pth')
            print(f'Model saved | Best test accuracy: {best_acc:.2f}%')
        
        # Update learning rate
        scheduler.step()
    
    return student_model, train_losses, train_accs

# Evaluate model
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f'Test accuracy: {acc:.2f}%')
    return acc

# Plot training process
def plot_training(train_losses, train_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('student_training_plot.png')
    plt.close()

# Generate Kaggle submission file
def generate_submission(model, test_data_path, output_file='student_kaggle_submission.csv'):
    # Load test data
    print(f"Loading test data {test_data_path}...")
    with open(test_data_path, 'rb') as f:
        test_data_dict = pickle.load(f)
    
    test_images = test_data_dict[b'data']
    test_ids = test_data_dict[b'ids']
    
    print(f"Test data shape: {test_images.shape}")
    print(f"Number of IDs: {len(test_ids)}")
    
    # Data preprocessing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Create dataset and data loader
    test_dataset = KaggleDataset(test_images, test_ids, transform_teacher=None, transform_student=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Start prediction
    print("Starting prediction...")
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for teacher_img, student_img, id_values in test_loader:
            # In predict only mode, we only need the student image   
            student_img = student_img.to(device)
            outputs = model(student_img)
            _, predicted = outputs.max(1)
            
            predictions.extend(predicted.cpu().numpy())
            ids.extend(id_values.numpy())
    
    # Create submission file
    print("Creating submission file...")
    submission = pd.DataFrame({
        'ID': ids,
        'Labels': predictions
    })
    
    # Save submission file
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")
    
    # Display class distribution
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    print("\nClass distribution:")
    value_counts = submission['Labels'].value_counts().sort_index()
    for label_idx, count in value_counts.items():
        class_name = classes[label_idx]
        print(f"{label_idx} ({class_name}): {count} ({count/len(submission)*100:.2f}%)")

# Main function
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Knowledge Distillation for CIFAR-10')
    parser.add_argument('--predict_only', action='store_true', 
                        help='Skip training and only generate Kaggle submission')
    parser.add_argument('--model_path', type=str, default='best_student_model.pth',
                        help='Path to the model weights for prediction (used with --predict_only)')
    parser.add_argument('--test_data_path', type=str, default='./data/cifar_test_nolabel.pkl',
                        help='Path to the test data (used with --predict_only)')
    parser.add_argument('--output_file', type=str, default='student_kaggle_submission.csv',
                        help='Output file name for Kaggle submission (used with --predict_only)')
    args = parser.parse_args()
    
    if args.predict_only:
        print("Prediction mode: Generating Kaggle submission file without training...")
        
        # Create student model
        print(f"Creating student model {MODEL_TYPE}...")
        student_model = create_resnet(MODEL_TYPE)
        
        # Load pre-trained weights
        print(f"Loading model weights from {args.model_path}...")
        student_model.load_state_dict(torch.load(args.model_path))
        student_model = student_model.to(device)
        student_model.eval()
        
        # Generate Kaggle submission file
        print(f"Generating Kaggle submission file...")
        generate_submission(student_model, args.test_data_path, args.output_file)
        
        print(f"Done! Submission file saved as {args.output_file}")
    else:
        print("Starting knowledge distillation training...")
        
        # Load data
        print("Loading data...")
        trainloader, testloader, kaggle_loader = get_data_loaders(BATCH_SIZE)
        
        # Load teacher model
        print("Loading teacher model...")
        teacher_model = load_teacher_model('teacher/pytorch_shake_shake/results/best_model_state.pth')
        print("Teacher model loaded and set to evaluation mode")
        
        # Create student model
        print(f"Creating student model {MODEL_TYPE}...")
        student_model = create_resnet(MODEL_TYPE)
        param_count = count_parameters(student_model)
        print(f'Student model parameter count: {param_count:,}')
        
        # Ensure parameter count is within 5 million
        if param_count > 5000000:
            print(f"Warning: Model parameter count exceeds limit ({param_count:,} > 5,000,000)")
        
        # Train student model (via knowledge distillation)
        print("\nStarting knowledge distillation training...")
        student_model, train_losses, train_accs = train_student(
            student_model, teacher_model, trainloader, testloader, kaggle_loader,
            epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            temperature=TEMPERATURE, lambda_distill=LAMBDA_DISTILL
        )
        
        # Evaluate best student model
        print("\nLoading best student model...")
        student_model.load_state_dict(torch.load('best_student_model.pth'))
        final_acc = evaluate(student_model, testloader)
        print(f'Final test accuracy: {final_acc:.2f}%')
        
        # Plot training process
        plot_training(train_losses, train_accs)
        
        # Generate Kaggle submission file
        print("\nGenerating Kaggle submission file...")
        generate_submission(student_model, './data/cifar_test_nolabel.pkl', 'student_kaggle_submission.csv') 