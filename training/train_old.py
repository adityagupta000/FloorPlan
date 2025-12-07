import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import time


# ============ METRICS ============
def calculate_dice(pred, target, num_classes):
    """Calculates mean Dice Coefficient (excluding background)"""
    smooth = 1e-6
    dice_per_class = []

    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    return (
        torch.mean(torch.stack(dice_per_class)) if dice_per_class else torch.tensor(0.0)
    )


def calculate_iou(pred, target, num_classes):
    """Calculates mean IoU (excluding background)"""
    smooth = 1e-6
    iou_per_class = []

    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou)

    return (
        torch.mean(torch.stack(iou_per_class)) if iou_per_class else torch.tensor(0.0)
    )


# ============ FOCAL LOSS (Better for extreme imbalance) ============
class FocalLoss(nn.Module):
    """Focal Loss to handle extreme class imbalance - focuses on hard examples"""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============ BOUNDARY-AWARE LOSS ============
def boundary_loss(pred, target, num_classes):
    """Emphasizes boundaries/edges - critical for thin walls"""
    pred_soft = torch.softmax(pred, dim=1)
    target_onehot = (
        F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    )

    # Sobel filters for edge detection
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)

    sobel_x = sobel_x.to(pred.device)
    sobel_y = sobel_y.to(pred.device)

    loss = 0
    for c in range(num_classes):
        pred_c = pred_soft[:, c : c + 1, :, :]
        target_c = target_onehot[:, c : c + 1, :, :]

        # Edge detection
        pred_edge_x = F.conv2d(pred_c, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_c, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)

        target_edge_x = F.conv2d(target_c, sobel_x, padding=1)
        target_edge_y = F.conv2d(target_c, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)

        # MSE on edges
        loss += F.mse_loss(pred_edge, target_edge)

    return loss / num_classes


# ============ WINDOW-SPECIFIC LOSS ============
def small_object_loss(pred, target, class_id, num_classes):
    """Extra penalty for missing small objects like windows"""
    pred_soft = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Focus on specific class (windows = class 3)
    pred_class = pred_soft[:, class_id, :, :]
    target_class = target_onehot[:, class_id, :, :]
    
    # Binary cross-entropy for this specific class
    bce_loss = F.binary_cross_entropy(pred_class, target_class, reduction='mean')
    
    return bce_loss


# ============ DATASET ============
class SegmentationDataset(Dataset):
    def __init__(self, img_root, mask_root, transform=None, augment=True):
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform = transform  # For normalization only
        self.augment = augment

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

        # SYNCHRONIZED TRANSFORMS (apply same transform to both)
        if self.augment:
            # Resize both together
            image = image.resize((512, 512), resample=Image.BILINEAR)
            mask = mask.resize((512, 512), resample=Image.NEAREST)
            
            # Random horizontal flip (same for both)
            if np.random.rand() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random vertical flip (same for both)
            if np.random.rand() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # AGGRESSIVE rotation - ANY angle from 0-360° for rotation invariance
            angle = np.random.uniform(0, 360)
            image = transforms.functional.rotate(image, angle, fill=255)  # Fill white
            mask = transforms.functional.rotate(mask, angle, fill=0)  # Fill black (background)
            
            # Color jitter (only for image)
            image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(image)
        else:
            # Validation - just resize
            image = image.resize((512, 512), resample=Image.BILINEAR)
            mask = mask.resize((512, 512), resample=Image.NEAREST)

        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Normalize (only image)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])(image)

        # Process mask
        mask_np = np.array(mask)
        allowed_classes = {1, 2, 3, 4}
        mask_np = np.where(np.isin(mask_np, list(allowed_classes)), mask_np, 0)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image, mask_tensor


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    return images, masks


# ============ ATTENTION MODULE ============
class AttentionBlock(nn.Module):
    """Attention mechanism to focus on thin features like walls"""

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


# ============ MODEL COMPONENTS ============
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
    """U-Net with ResNet50 backbone and attention gates for thin feature detection"""

    def __init__(self, num_classes):
        super(ResNet50_UNet_Attention, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Encoder (ResNet50 backbone)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels

        # Attention gates
        self.att4 = AttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.att3 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # Decoder with attention
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
        # Encoder
        e0 = self.encoder0(x)  # [B,64,H/2,W/2]
        p0 = self.pool0(e0)  # [B,64,H/4,W/4]
        e1 = self.encoder1(p0)  # [B,256,H/4,W/4]
        e2 = self.encoder2(e1)  # [B,512,H/8,W/8]
        e3 = self.encoder3(e2)  # [B,1024,H/16,W/16]
        e4 = self.encoder4(e3)  # [B,2048,H/32,W/32]

        # Decoder with attention gates
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


# ============ LOSS FUNCTIONS ============
def dice_loss(pred, target, num_classes, smooth=1):
    """Dice loss for better boundary detection"""
    pred = torch.softmax(pred, dim=1)
    target_onehot = (
        F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    )

    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# ============ TRAINING & VALIDATION ============
def train_epoch(
    model, dataloader, optimizer, criterion, device, num_classes, use_boundary_loss=True
):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for imgs, masks in tqdm(dataloader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        outputs = F.interpolate(
            outputs, size=masks.shape[1:], mode="bilinear", align_corners=False
        )

        # Combined loss: CrossEntropy + Dice + Boundary + Window-specific
        ce_loss = criterion(outputs, masks)
        d_loss = dice_loss(outputs, masks, num_classes)

        if use_boundary_loss:
            b_loss = boundary_loss(outputs, masks, num_classes)
            w_loss = small_object_loss(outputs, masks, class_id=3, num_classes=num_classes)
            loss = ce_loss + d_loss + 0.2 * b_loss + 0.3 * w_loss
        else:
            loss = ce_loss + d_loss

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(outputs, dim=1)
        running_dice += calculate_dice(preds, masks, num_classes).item() * imgs.size(0)
        running_iou += calculate_iou(preds, masks, num_classes).item() * imgs.size(0)

    N = len(dataloader.dataset)
    return running_loss / N, running_dice / N, running_iou / N


def val_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Validation", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            outputs = F.interpolate(
                outputs, size=masks.shape[1:], mode="bilinear", align_corners=False
            )

            ce_loss = criterion(outputs, masks)
            d_loss = dice_loss(outputs, masks, num_classes)
            loss = ce_loss + d_loss

            running_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)
            running_dice += calculate_dice(
                preds, masks, num_classes
            ).item() * imgs.size(0)
            running_iou += calculate_iou(preds, masks, num_classes).item() * imgs.size(
                0
            )

    N = len(dataloader.dataset)
    return running_loss / N, running_dice / N, running_iou / N


# ============ VISUALIZATION ============
def visualize_predictions(
    model, dataloader, device, epoch, num_samples=5, save_path=None
):
    """Save side-by-side comparison: Original | Ground Truth | Prediction"""
    model.eval()

    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        outputs = F.interpolate(
            outputs, size=masks.shape[1:], mode="bilinear", align_corners=False
        )
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # Denormalize images
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    imgs = images.cpu().permute(0, 2, 3, 1).numpy()
    imgs_denorm = imgs * STD + MEAN
    masks = masks.cpu().numpy()

    num_samples = min(num_samples, len(imgs))

    # Create figure with better layout
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Color map: 0=black, 1=red(walls), 2=green(doors), 3=blue(windows), 4=yellow(floors)
    from matplotlib.colors import ListedColormap

    colors = ["black", "red", "green", "blue", "yellow"]
    cmap = ListedColormap(colors)

    for i in range(num_samples):
        # Original Image
        axes[i, 0].imshow(imgs_denorm[i].clip(0, 1))
        axes[i, 0].set_title(f"Original Image {i+1}", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        # Ground Truth
        im1 = axes[i, 1].imshow(
            masks[i], cmap=cmap, vmin=0, vmax=4, interpolation="nearest"
        )
        axes[i, 1].set_title(
            "Ground Truth\n(Red=Walls, Green=Doors, Blue=Windows, Yellow=Floors)",
            fontsize=10,
            fontweight="bold",
        )
        axes[i, 1].axis("off")

        # Prediction
        im2 = axes[i, 2].imshow(
            preds[i], cmap=cmap, vmin=0, vmax=4, interpolation="nearest"
        )
        axes[i, 2].set_title(
            f"Prediction (Epoch {epoch})\n(Red=Walls, Green=Doors, Blue=Windows, Yellow=Floors)",
            fontsize=10,
            fontweight="bold",
        )
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved predictions to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_training_history(history, save_path):
    """Generate 4-panel training history plot"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dice Score
    axes[0, 1].plot(epochs, history["train_dice"], "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_dice"], "r-", label="Validation", linewidth=2)
    axes[0, 1].set_title("Dice Coefficient", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IoU
    axes[1, 0].plot(epochs, history["train_iou"], "b-", label="Train", linewidth=2)
    axes[1, 0].plot(epochs, history["val_iou"], "r-", label="Validation", linewidth=2)
    axes[1, 0].set_title(
        "IoU (Intersection over Union)", fontsize=14, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].semilogy(epochs, history["learning_rate"], "g-", linewidth=2)
    axes[1, 1].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate (log scale)")
    axes[1, 1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history to {save_path}")


# ============ MAIN ============
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === PATHS - UPDATE THESE ===
    img_dir_train = r"C:\Users\LENOVO\Downloads\TrainModel\TrainModel\ImagesReal\train"
    mask_dir_train = r"C:\Users\LENOVO\Downloads\TrainModel\TrainModel\MasksReal\train"
    img_dir_val = r"C:\Users\LENOVO\Downloads\TrainModel\TrainModel\ImagesReal\val"
    mask_dir_val = r"C:\Users\LENOVO\Downloads\TrainModel\TrainModel\MasksReal\val"
    img_dir_test = r"C:\Users\LENOVO\Downloads\TrainModel\TrainModel\ImagesReal\test"     
    mask_dir_test = r"C:\Users\LENOVO\Downloads\TrainModel\TrainModel\MasksReal\test"     

    # === DATA TRANSFORMS ===
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # === DATASETS ===
    train_dataset = SegmentationDataset(
        img_dir_train, mask_dir_train, 
        transform=normalize,
        augment=True
    )
    val_dataset = SegmentationDataset(
        img_dir_val, mask_dir_val, 
        transform=normalize,
        augment=False
    )

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("No data found! Check your paths.")
        exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )

    # === MODEL WITH ATTENTION ===
    num_classes = 5  # 0=bg, 1=walls, 2=doors, 3=windows, 4=floors
    model = ResNet50_UNet_Attention(num_classes=num_classes).to(device)

    # === CRITICAL: CLASS WEIGHTS FOR THIN FEATURES ===
    # Class mapping: 0=bg, 1=walls, 2=doors, 3=windows, 4=floors
    # Walls are thin lines (few pixels) - need high weight
    # Doors and windows are small features - need very high weight
    # Floors are large areas - normal weight
    class_weights = torch.tensor(
        [
            0.5,  # Class 0: Background (abundant, low weight)
            3.0,  # Class 1: Walls (CRITICAL - thin lines, high weight)
            3.5,  # Class 2: Doors (small features, high weight)
            5.0,  # Class 3: Windows (very small features, VERY high weight)
            1.0,  # Class 4: Floors (large areas, normal weight)
        ]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # === TRAINING CONFIG ===
    epochs = 50
    best_val_dice = 0.0
    patience = 10
    patience_counter = 0
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "learning_rate": [],
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # === TRAINING LOOP ===
    print("\nStarting training with BOUNDARY-AWARE LOSS for thin walls...\n")

    for epoch in range(epochs):
        print(f"{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        # Train with boundary loss
        train_loss, train_dice, train_iou = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes,
            use_boundary_loss=True,
        )

        # Validate
        val_loss, val_dice, val_iou = val_epoch(
            model, val_loader, criterion, device, num_classes
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["learning_rate"].append(current_lr)

        print(
            f"Train - Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}"
        )
        print(
            f"Val   - Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}"
        )
        print(f"LR: {current_lr:.2e}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model based on Dice score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            print("Model improved and saved!")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        # === SAVE VISUALIZATIONS EVERY 5 EPOCHS ===
        if (epoch + 1) % 5 == 0:
            pred_path = os.path.join(
                save_dir, f"predictions_epoch_{epoch+1:03d}_{timestamp}.png"
            )
            visualize_predictions(
                model, val_loader, device, epoch + 1, num_samples=5, save_path=pred_path
            )

    # === FINAL OUTPUTS ===
    print("\nGenerating final plots...")

    # Save history JSON
    history_json = os.path.join(save_dir, f"history_{timestamp}.json")
    with open(history_json, "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=4)
    print(f"Saved history: {history_json}")

    # Save history plot
    history_plot = os.path.join(save_dir, f"training_history_{timestamp}.jpg")
    plot_training_history(history, history_plot)

    print(f"\nTraining complete! Best Dice: {best_val_dice:.4f}")
    print(f"Results saved in: {save_dir}/")
    print("\nClass Mapping:")
    print("   0: Background, 1: Walls, 2: Doors, 3: Windows, 4: Floors")
    print("\nClass weights used:")
    print("   Background: 0.5, Walls: 3.0, Doors: 3.5, Windows: 5.0, Floors: 1.0")
    
    
    if os.path.exists(img_dir_test) and os.path.exists(mask_dir_test):
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET...")
        print("="*70)
        
        test_dataset = SegmentationDataset(
            img_dir_test, 
            mask_dir_test,
            transform=normalize,
            augment=False
        )
        
        print(f"Test set: {len(test_dataset)} images")
        
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=custom_collate,
                num_workers=4,
                pin_memory=True
            )
            
            # Load best model
            print("Loading best model...")
            checkpoint = torch.load(os.path.join(save_dir, "best_model.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            print("Running evaluation...")
            test_loss, test_dice, test_iou = val_epoch(
                model, test_loader, criterion, device, num_classes
            )
            
            print("\n" + "="*70)
            print("FINAL TEST SET RESULTS:")
            print("="*70)
            print(f"   Loss: {test_loss:.4f}")
            print(f"   Dice: {test_dice:.4f}")
            print(f"   IoU:  {test_iou:.4f}")
            print("="*70 + "\n")
            
            # Save test results to JSON
            test_results = {
                "test_loss": float(test_loss),
                "test_dice": float(test_dice),
                "test_iou": float(test_iou),
                "best_val_dice": float(best_val_dice)
            }
            test_results_path = os.path.join(save_dir, f"test_results_{timestamp}.json")
            with open(test_results_path, "w") as f:
                json.dump(test_results, f, indent=4)
            print(f"Test results saved to: {test_results_path}\n")
        else:
            print("Test set is empty!")
    else:
        print("\nTest set not found - skipping test evaluation")
        print(f"   Expected at: {img_dir_test}\n")


# ============ INFERENCE WITH ROTATION INVARIANCE (FOR FRONTEND) ============
def predict_with_tta(model, image, device, num_classes=5):
    """
    Test-Time Augmentation: Predict on multiple rotations and average results
    Makes model work on images at ANY angle - USE THIS IN YOUR FRONTEND!
    
    Args:
        model: Trained segmentation model
        image: Preprocessed image tensor [C, H, W]
        device: torch device
        num_classes: Number of segmentation classes
    
    Returns:
        prediction: Segmented mask [H, W] with class labels
    """
    model.eval()
    predictions = []
    
    # Test on original + 3 rotations (0°, 90°, 180°, 270°)
    angles = [0, 90, 180, 270]
    
    with torch.no_grad():
        for angle in angles:
            # Rotate image
            img_rotated = transforms.functional.rotate(image, angle)
            img_batch = img_rotated.unsqueeze(0).to(device)
            
            # Predict
            output = model(img_batch)
            output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
            output_soft = torch.softmax(output, dim=1)
            
            # Rotate prediction BACK to original orientation
            output_soft = output_soft.squeeze(0)  # Remove batch dim [C, H, W]
            
            # Rotate each class channel back
            rotated_back = []
            for c in range(output_soft.shape[0]):
                channel = output_soft[c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                channel_rotated = transforms.functional.rotate(channel, -angle, fill=0)
                rotated_back.append(channel_rotated.squeeze(0).squeeze(0))
            
            output_soft_back = torch.stack(rotated_back, dim=0)  # [C, H, W]
            predictions.append(output_soft_back)
    
    # Average all predictions
    final_pred_soft = torch.stack(predictions).mean(dim=0)  # [C, H, W]
    final_pred = torch.argmax(final_pred_soft, dim=0)  # [H, W]
    
    return final_pred.cpu().numpy()


# Example usage in your frontend:
"""
# Load model
model = ResNet50_UNet_Attention(num_classes=5).to(device)
checkpoint = torch.load('results/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Prepare image (same preprocessing as training)
from PIL import Image
image = Image.open('test_floorplan.jpg').convert('RGB')
image = image.resize((512, 512))
image_tensor = transforms.ToTensor()(image)
image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])(image_tensor)

# Predict with rotation invariance
prediction = predict_with_tta(model, image_tensor, device)

# prediction is now a numpy array [512, 512] with values 0-4
# 0=background, 1=walls, 2=doors, 3=windows, 4=floors
"""