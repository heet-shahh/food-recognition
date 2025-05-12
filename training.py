import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler # Use autocast and GradScaler from torch.cuda.amp
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
config = {
    "DATA_DIR": "dataset/images", # Base directory with train/, val/, test/
    "MODEL_NAME": "efficientnet_b4", # Can be used later to select different models
    "BATCH_SIZE": 32,
    "IMG_SIZE": 384, # Input size expected by EfficientNet-B4
    "NUM_EPOCHS": 5,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 5e-2,
    "DROPOUT_P": 0.4,
    "NUM_WORKERS": 4, # Number of workers for DataLoader
    "PIN_MEMORY": True, # Set based on device later
    "USE_WEIGHTED_LOSS": True, # Attempt to use class weighting
    "FREEZE_BACKBONE": True, # Freeze pre-trained layers initially
    # Paths
    "MODEL_SAVE_PATH": "efficientnet_b4_finetuned.pth", # Final model save path (optional)
    "BEST_MODEL_SAVE_PATH": "best_efficientnet_b4.pth", # Path to save the best model during training
    "MAPPING_SAVE_PATH": "class_mapping.json",
    "PLOT_SAVE_PATH": "training_validation_metrics.png",
}

# --- Helper Functions ---

def get_device():
    """Gets the appropriate device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True # Enable cuDNN benchmark
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def get_transforms(img_size, weights):
    """Gets the train and validation/test transforms"""
    # Use normalization parameters from the pre-trained weights
    normalization_mean = weights.transforms().mean
    normalization_std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) # Optional
    ])

    # Use the standard transforms recommended by the weights for validation/testing
    val_test_transform = weights.transforms()

    print("Train Transforms:", train_transform)
    print("Val/Test Transforms:", val_test_transform)
    return train_transform, val_test_transform

def get_dataloaders(data_dir, batch_size, train_transform, val_test_transform, num_workers, pin_memory, mapping_save_path):
    """Creates datasets and dataloaders for train, val, and test sets."""
    print("Loading datasets...")
    try:
        train_path = os.path.join(data_dir, 'train')
        val_path = os.path.join(data_dir, 'val')
        test_path = os.path.join(data_dir, 'test')

        if not os.path.isdir(train_path):
            raise FileNotFoundError(f"Training directory not found: {train_path}")
        if not os.path.isdir(val_path):
            raise FileNotFoundError(f"Validation directory not found: {val_path}")
        if not os.path.isdir(test_path):
            raise FileNotFoundError(f"Test directory not found: {test_path}")

        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)

        # Determine number of classes and save mapping
        num_classes = len(train_dataset.classes)
        class_mapping = train_dataset.class_to_idx
        print(f"Found {num_classes} classes: {train_dataset.classes}")

        # Save class mapping
        try:
            with open(mapping_save_path, 'w') as f:
                json.dump(class_mapping, f, indent=4)
            print(f"Class mapping saved to: {mapping_save_path}")
        except IOError as e:
            print(f"Warning: Could not save class mapping to {mapping_save_path}: {e}")


        if not train_dataset or not val_dataset or not test_dataset:
            raise ValueError("One or more datasets (train, val, test) are empty.")

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print("DataLoaders created.")

        return train_loader, val_loader, test_loader, train_dataset, num_classes # Return train_dataset for weight calculation

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'train', 'val', and 'test' subdirectories exist in DATA_DIR and contain images.")
        exit()

def calculate_class_weights(train_dataset, device):
    """Calculates class weights for handling imbalance."""
    print("Calculating class weights...")
    try:
        class_labels = train_dataset.targets
        class_indices = sorted(list(set(class_labels)))

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(class_indices),
            y=np.array(class_labels)
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"Class weights calculated (first 5): {class_weights_tensor[:5]}")
        return class_weights_tensor
    except Exception as e:
        print(f"ERROR: Could not calculate class weights: {e}. Using standard loss.")
        return None

def get_model(model_name, num_classes, freeze_backbone=True):
    """Loads a pre-trained model and modifies its classifier."""
    print(f"Loading {model_name} model...")
    if model_name == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights)

        if freeze_backbone:
            print("Freezing backbone layers...")
            for param in model.parameters():
                param.requires_grad = False

        # Replace the classifier - new layers automatically have requires_grad=True
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        print("Model classifier replaced.")
        print("Trainable parameters after modification:")
        for name, param in model.named_parameters():
             if param.requires_grad:
                 print(f"- {name}")

        return model, weights # Return weights for transforms
    else:
        raise ValueError(f"Model {model_name} not implemented yet.")


def get_optimizer_scheduler(model, learning_rate, weight_decay):
    """Creates the optimizer and learning rate scheduler."""
    print("Defining Optimizer and Scheduler...")
    # Filter parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    # Reduce LR when validation accuracy plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    print(f"Optimizer: AdamW (LR={learning_rate}, WD={weight_decay})")
    print("Scheduler: ReduceLROnPlateau (Factor=0.1, Patience=3, Mode=max)")
    return optimizer, scheduler

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch_num, num_epochs):
    """Performs one training epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    is_cuda = device == torch.device("cuda")

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num}/{num_epochs} [Training]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Use autocast only if on CUDA
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if is_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # No scaling needed for CPU
            loss.backward()
            optimizer.step()


        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar
        batch_acc = (predicted == labels).sum().item() / labels.size(0) if labels.size(0) > 0 else 0
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"[Epoch {epoch_num}] Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, phase="Validation", epoch_num=None):
    """Evaluates the model on a given dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    is_cuda = device == torch.device("cuda")

    epoch_str = f"Epoch {epoch_num}" if epoch_num else "Final"
    progress_bar = tqdm(loader, desc=f"{epoch_str} [{phase}]", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast(device_type=device.type, enabled=is_cuda):
            outputs = model(inputs)
            loss = criterion(outputs, labels) # Can still calculate loss for info if needed

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar
        batch_acc = (predicted == labels).sum().item() / labels.size(0) if labels.size(0) > 0 else 0
        progress_bar.set_postfix(acc=f"{batch_acc:.4f}")


    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"[{epoch_str}] {phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """Plots training and validation metrics."""
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o', linestyle='-', color='r')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='s', linestyle='--', color='m')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', color='b')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='s', linestyle='--', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Metrics plot saved as {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    device = get_device()
    config["PIN_MEMORY"] = True if device == torch.device("cuda") else False # Adjust pin_memory based on device

    # --- Data Loading ---
    # Need temporary weights object to get transforms
    temp_weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    train_transform, val_test_transform = get_transforms(config["IMG_SIZE"], temp_weights)
    train_loader, val_loader, test_loader, train_dataset, num_classes = get_dataloaders(
        config["DATA_DIR"],
        config["BATCH_SIZE"],
        train_transform,
        val_test_transform,
        config["NUM_WORKERS"],
        config["PIN_MEMORY"],
        config["MAPPING_SAVE_PATH"]
    )

    # --- Class Weights (Optional) ---
    class_weights_tensor = None
    if config["USE_WEIGHTED_LOSS"]:
        class_weights_tensor = calculate_class_weights(train_dataset, device)
        if class_weights_tensor is None:
            print("Proceeding without weighted loss as calculation failed.")

    # --- Model ---
    model, _ = get_model( # Don't need weights object returned here anymore
        config["MODEL_NAME"],
        num_classes,
        config["FREEZE_BACKBONE"]
    )
    model.to(device)

    # --- Loss Function ---
    print("Defining Loss function...")
    if config["USE_WEIGHTED_LOSS"] and class_weights_tensor is not None:
        print("Using Weighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        print("Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    # --- Optimizer and Scheduler ---
    optimizer, scheduler = get_optimizer_scheduler(
        model,
        config["LEARNING_RATE"],
        config["WEIGHT_DECAY"]
    )

    # --- Mixed Precision Scaler ---
    # Initialize GradScaler only if on CUDA
    scaler = GradScaler(device='cuda')
    print(f"Mixed precision enabled: {scaler.is_enabled()}")


    # --- Training Loop ---
    print("\n--- Starting Training ---")
    start_time = time.time()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, config["NUM_EPOCHS"] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config["NUM_EPOCHS"]
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, phase="Validation", epoch_num=epoch
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update LR Scheduler
        if scheduler:
            # scheduler.step(val_loss) # If monitoring loss
            scheduler.step(val_acc) # If monitoring accuracy

        # Save Best Model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving best model to {config['BEST_MODEL_SAVE_PATH']}...")
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), config['BEST_MODEL_SAVE_PATH'])
            except Exception as e:
                print(f"Error saving best model: {e}")

    end_time = time.time()
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # --- Plotting ---
    print("\n--- Plotting Metrics ---")
    plot_metrics(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        config["PLOT_SAVE_PATH"]
    )

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set using Best Model ---")
    try:
        # Load the best weights saved during training
        print(f"Loading best model weights from: {config['BEST_MODEL_SAVE_PATH']}")
        model.load_state_dict(torch.load(config['BEST_MODEL_SAVE_PATH'], map_location=device))

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device, phase="Testing (Best Model)"
        )
        print(f"\nFinal Test Accuracy (Best Model): {test_accuracy:.4f}")
        print(f"Final Test Loss (Best Model): {test_loss:.4f}")

    except FileNotFoundError:
        print(f"Warning: Best model file '{config['BEST_MODEL_SAVE_PATH']}' not found.")
        print("Evaluating using the model from the final epoch instead.")
        # Evaluate the model as it is (from the last epoch)
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device, phase="Testing (Final Epoch Model)"
        )
        print(f"\nFinal Test Accuracy (Final Epoch Model): {test_accuracy:.4f}")
        print(f"Final Test Loss (Final Epoch Model): {test_loss:.4f}")

    except Exception as e:
        print(f"An error occurred during final evaluation: {e}")

    # Optional: Save the final model state if needed
    # print(f"Saving final model state to {config['MODEL_SAVE_PATH']}")
    # torch.save(model.state_dict(), config['MODEL_SAVE_PATH'])

    print("\n--- Experiment Complete ---")