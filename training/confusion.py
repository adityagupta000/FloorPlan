import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import json


# ============ MODEL COMPONENTS (Same as training) ============
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet50_UNet_Attention(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_UNet_Attention, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.att4 = AttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.att3 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(2048, 1024, dropout=True)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(1024, 512, dropout=True)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256, dropout=False)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout=False)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4)
        e3_att = self.att4(g=d4, x=e3)
        d4 = self.dec4(torch.cat([d4, e3_att], dim=1))

        d3 = self.up3(d4)
        e2_att = self.att3(g=d3, x=e2)
        d3 = self.dec3(torch.cat([d3, e2_att], dim=1))

        d2 = self.up2(d3)
        e1_att = self.att2(g=d2, x=e1)
        d2 = self.dec2(torch.cat([d2, e1_att], dim=1))

        d1 = self.up1(d2)
        e0_att = self.att1(g=d1, x=e0)
        d1 = self.dec1(torch.cat([d1, e0_att], dim=1))

        out = self.final_conv(d1)
        return out


# ============ DATASET ============
class SegmentationDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None):
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform = transform

        self.image_paths = []
        for root, _, files in os.walk(img_root):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        relative_path = os.path.relpath(img_path, self.img_root)
        mask_path = os.path.join(self.mask_root, relative_path)
        mask_path = os.path.splitext(mask_path)[0] + "_mask.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((512, 512), resample=Image.BILINEAR)
        mask = mask.resize((512, 512), resample=Image.NEAREST)

        image = transforms.ToTensor()(image)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(image)

        mask_np = np.array(mask)
        allowed_classes = {1, 2, 3, 4}
        mask_np = np.where(np.isin(mask_np, list(allowed_classes)), mask_np, 0)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image, mask_tensor


# ============ EVALUATION FUNCTIONS ============
def evaluate_model(model, dataloader, device, num_classes):
    """Collect all predictions and ground truth labels"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            outputs = model(imgs)
            outputs = F.interpolate(
                outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
            )
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Flatten all predictions and targets
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_targets = np.concatenate([t.flatten() for t in all_targets])
    
    return all_preds, all_targets


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    """
    Create beautiful confusion matrix visualization
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        normalize: Whether to normalize values (show percentages)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Percentage'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        square=True
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: {save_path}")
    plt.close()


def calculate_per_class_metrics(cm, class_names):
    """Calculate precision, recall, F1 for each class"""
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(cm[i, :].sum())
        }
    
    return metrics


def plot_per_class_metrics(metrics, save_path):
    """Create bar chart of per-class metrics"""
    class_names = list(metrics.keys())
    precision = [metrics[c]['precision'] for c in class_names]
    recall = [metrics[c]['recall'] for c in class_names]
    f1 = [metrics[c]['f1_score'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to: {save_path}")
    plt.close()


# ============ MAIN EVALUATION ============
if __name__ == "__main__":
    # === CONFIGURATION ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # UPDATE THESE PATHS
    model_path = "results/best_model.pth"
    img_dir_test = r"C:\Project_Files\v_Final\training\ImagesReal\test"
    mask_dir_test = r"C:\Project_Files\v_Final\training\masksReal\test"
    output_dir = "evaluation_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Class information
    num_classes = 5
    class_names = ['Background', 'Walls', 'Doors', 'Windows', 'Floors']
    class_colors = ['black', 'red', 'green', 'blue', 'yellow']
    
    # === LOAD MODEL ===
    print("Loading model...")
    model = ResNet50_UNet_Attention(num_classes=num_classes).to(device)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        exit(1)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully from {model_path}")
    print(f"  - Trained epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Dice: {checkpoint.get('val_dice', 'N/A'):.4f}\n")
    
    # === LOAD DATA ===
    print("Loading test dataset...")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    test_dataset = SegmentationDataset(
        img_dir_test,
        mask_dir_test,
        transform=normalize
    )
    
    print(f"Test images: {len(test_dataset)}\n")
    
    if len(test_dataset) == 0:
        print("ERROR: No test images found!")
        exit(1)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # === EVALUATE ===
    all_preds, all_targets = evaluate_model(model, test_loader, device, num_classes)
    
    # === COMPUTE CONFUSION MATRIX ===
    print("\nComputing confusion matrix...")
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    # === PLOT CONFUSION MATRICES ===
    print("Generating visualizations...")
    
    # Raw counts
    plot_confusion_matrix(
        cm, 
        class_names, 
        os.path.join(output_dir, 'confusion_matrix_counts.png'),
        normalize=False
    )
    
    # Normalized (percentages)
    plot_confusion_matrix(
        cm, 
        class_names, 
        os.path.join(output_dir, 'confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # === CALCULATE METRICS ===
    print("\nCalculating per-class metrics...")
    metrics = calculate_per_class_metrics(cm, class_names)
    
    # Plot metrics
    plot_per_class_metrics(
        metrics,
        os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    # === CLASSIFICATION REPORT ===
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        all_targets, 
        all_preds, 
        labels=range(num_classes),
        target_names=class_names,
        digits=4
    ))
    
    # === SAVE DETAILED RESULTS ===
    results = {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'per_class_metrics': metrics,
        'overall_accuracy': float((all_preds == all_targets).sum() / len(all_targets)),
        'total_pixels': int(len(all_targets))
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    
    # === PRINT SUMMARY ===
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Total Pixels Evaluated: {results['total_pixels']:,}")
    print("\nPer-Class Performance:")
    for class_name, metric in metrics.items():
        print(f"  {class_name:12s}: Precision={metric['precision']:.4f}, "
              f"Recall={metric['recall']:.4f}, F1={metric['f1_score']:.4f}")
    print("="*70)
    print(f"\nAll results saved in: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - confusion_matrix_counts.png (raw pixel counts)")
    print(f"  - confusion_matrix_normalized.png (percentages)")
    print(f"  - per_class_metrics.png (precision/recall/F1 bars)")
    print(f"  - evaluation_results.json (detailed metrics)")