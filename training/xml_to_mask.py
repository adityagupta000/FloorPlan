import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import re
from tqdm import tqdm
import matplotlib.pyplot as plt


def check_prerequisites(root_folder, train_txt, val_txt, test_txt):
    """Check if all required files and folders exist"""
    print("="*70)
    print("PREREQUISITES CHECK")
    print("="*70)
    
    issues = []
    
    if not os.path.exists(root_folder):
        issues.append(f"❌ Root folder does not exist: {root_folder}")
    else:
        print(f"✓ Root folder exists: {root_folder}")
        
        svg_count = sum(1 for root, dirs, files in os.walk(root_folder) if 'model.svg' in files)
        print(f"✓ Found {svg_count} folders with model.svg")
        
        if svg_count == 0:
            issues.append("❌ No model.svg files found in root folder")
        
        xml_count = sum(1 for root, dirs, files in os.walk(root_folder) if 'annotation.xml' in files)
        print(f"✓ Found {xml_count} folders with annotation.xml")
        
        if xml_count == 0:
            issues.append("❌ No annotation.xml files found! Run svg_to_xml.py first")
        elif xml_count < svg_count:
            issues.append(f"⚠ WARNING: Only {xml_count}/{svg_count} SVG files converted to XML. Run svg_to_xml.py")
    
    split_files = [
        ('train.txt', train_txt),
        ('val.txt', val_txt),
        ('test.txt', test_txt)
    ]
    
    for name, path in split_files:
        if not os.path.exists(path):
            issues.append(f"❌ {name} not found at: {path}")
        else:
            with open(path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            print(f"✓ {name} exists with {len(lines)} folders")
            
            if len(lines) == 0:
                issues.append(f"⚠ WARNING: {name} is empty")
    
    print("="*70 + "\n")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
        print("\n" + "="*70)
        print("CANNOT PROCEED - Please fix the issues above first")
        print("="*70)
        return False
    
    return True


def parse_annotation_xml(xml_path):
    """Parse annotation XML and extract polygons with class names"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        points = []
        for pt in obj.find('points').findall('point'):
            x = float(pt.find('x').text)
            y = float(pt.find('y').text)
            points.append((x, y))
        objects.append({'name': name, 'points': points})
    return objects


def validate_small_objects(mask, min_size=25, class_names=None):
    """
    Remove tiny objects that will be lost during training resize to 512x512.
    """
    if class_names is None:
        class_names = {1: 'Wall', 2: 'Door', 3: 'Window', 4: 'Floor'}
    
    removed_count = {cid: 0 for cid in [2, 3]}
    
    for class_id in [2, 3]:
        class_mask = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_size:
                mask[labels == i] = 0
                removed_count[class_id] += 1
    
    return mask


def polygons_to_mask(polygons, image_size, class_name_to_id):
    """
    Convert polygons to segmentation mask with smart layering.
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    polygons_by_class = {
        'Floor': [],
        'Wall': [],
        'Door': [],
        'Window': []
    }
    
    for poly in polygons:
        cls_name = poly['name']
        if cls_name in polygons_by_class:
            polygons_by_class[cls_name].append(poly)
    
    # Render floors
    for poly in polygons_by_class['Floor']:
        pts = np.array(poly['points'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 4)
    
    # Create wall layer
    wall_mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons_by_class['Wall']:
        pts = np.array(poly['points'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(wall_mask, [pts], isClosed=True, color=1, thickness=4)
        cv2.fillPoly(wall_mask, [pts], 1)
    
    # Create door/window priority layer
    door_window_mask = np.zeros((height, width), dtype=np.uint8)
    
    for poly in polygons_by_class['Door']:
        pts = np.array(poly['points'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(door_window_mask, [pts], 2)
    
    for poly in polygons_by_class['Window']:
        pts = np.array(poly['points'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(door_window_mask, [pts], 3)
    
    # Smart layering
    mask[wall_mask > 0] = 1
    mask[door_window_mask > 0] = door_window_mask[door_window_mask > 0]
    
    # Validate small objects
    mask = validate_small_objects(mask, min_size=25)
    
    return mask


def process_folders(folders, split_name, root_folder, output_mask_folder, 
                   output_image_folder, class_map, image_exts={'.png', '.jpg', '.jpeg'}):
    """Process folders and create masks - FIXED OVERFLOW VERSION"""
    count = 0
    skipped = 0
    # ✅ FIXED: Properly terminated string
    scaled_img_pattern = re.compile(r'.*_scaled\.png$', re.IGNORECASE)
    
    # ✅ FIXED: Use Python native int to prevent overflow
    class_pixel_counts = {cls: 0 for cls in class_map.keys()}
    
    pbar = tqdm(folders, desc=f"Processing {split_name}")
    
    for folder_rel_path in pbar:
        dirpath = os.path.join(root_folder, folder_rel_path.lstrip('/\\'))

        if not os.path.isdir(dirpath):
            skipped += 1
            continue
        
        filenames = os.listdir(dirpath)
        xml_path = os.path.join(dirpath, 'annotation.xml')
        
        if not os.path.exists(xml_path):
            skipped += 1
            continue
        
        scaled_images = [f for f in filenames if scaled_img_pattern.match(f)]
        if not scaled_images:
            skipped += 1
            continue

        scaled_img_name = scaled_images[0]
        scaled_img_path = os.path.join(dirpath, scaled_img_name)

        try:
            polygons = parse_annotation_xml(xml_path)
        except Exception as e:
            print(f"\nError parsing {xml_path}: {e}")
            skipped += 1
            continue
        
        try:
            scaled_img = Image.open(scaled_img_path)
            scaled_width, scaled_height = scaled_img.size
        except Exception as e:
            print(f"\nError opening {scaled_img_path}: {e}")
            skipped += 1
            continue

        base_mask = polygons_to_mask(polygons, (scaled_width, scaled_height), class_map)
        
        # ✅ FIXED: Convert to Python int to prevent overflow
        for cls_name, cls_id in class_map.items():
            pixel_count = int(np.sum(base_mask == cls_id))
            class_pixel_counts[cls_name] += pixel_count

        for img_name in filenames:
            if img_name.endswith('.xml') or scaled_img_pattern.match(img_name):
                continue
            
            ext = os.path.splitext(img_name)[1].lower()
            if ext not in image_exts:
                continue

            img_path = os.path.join(dirpath, img_name)
            
            try:
                image = Image.open(img_path)
                img_width, img_height = image.size
            except Exception as e:
                continue

            if (img_width, img_height) == (scaled_width, scaled_height):
                mask_to_save = base_mask
            else:
                mask_to_save = cv2.resize(base_mask, (img_width, img_height), 
                                         interpolation=cv2.INTER_NEAREST)

            relative_path = os.path.relpath(dirpath, root_folder)
            
            mask_dir = os.path.join(output_mask_folder, split_name, relative_path)
            os.makedirs(mask_dir, exist_ok=True)
            mask_name = os.path.splitext(img_name)[0] + '_mask.png'
            mask_path = os.path.join(mask_dir, mask_name)
            
            cv2.imwrite(mask_path, mask_to_save)
            
            img_output_dir = os.path.join(output_image_folder, split_name, relative_path)
            os.makedirs(img_output_dir, exist_ok=True)
            img_output_path = os.path.join(img_output_dir, img_name)
            
            if not os.path.exists(img_output_path):
                image.save(img_output_path)
            
            count += 1
            pbar.set_postfix({'processed': count, 'skipped': skipped})

    return count, skipped, class_pixel_counts


def visualize_sample_mask(mask_folder, split_name, num_samples=3):
    """Visualize sample masks to verify correctness."""
    print(f"\n{'='*70}")
    print(f"VISUALIZING SAMPLE MASKS - {split_name.upper()}")
    print(f"{'='*70}")
    
    mask_files = []
    for root, dirs, files in os.walk(os.path.join(mask_folder, split_name)):
        for file in files:
            if file.endswith('_mask.png'):
                mask_files.append(os.path.join(root, file))
    
    if not mask_files:
        print("No mask files found!")
        return
    
    import random
    sample_files = random.sample(mask_files, min(num_samples, len(mask_files)))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['Background', 'Wall', 'Door', 'Window', 'Floor']
    
    for idx, mask_path in enumerate(sample_files):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        unique_vals = np.unique(mask)
        print(f"\nMask {idx+1}: {os.path.basename(mask_path)}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Shape: {mask.shape}")
        
        if not all(val in [0,1,2,3,4] for val in unique_vals):
            print(f"  ⚠ WARNING: Invalid values detected!")
        
        for class_id in range(5):
            class_mask = (mask == class_id)
            pixel_count = class_mask.sum()
            
            axes[idx, class_id].imshow(class_mask, cmap='gray')
            axes[idx, class_id].set_title(
                f"{class_names[class_id]}\n{pixel_count:,} px",
                fontsize=10
            )
            axes[idx, class_id].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(mask_folder, f'sample_visualization_{split_name}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {viz_path}")
    plt.close()


def read_folders_from_txt(txt_path):
    """Read folder paths from text file"""
    if not os.path.exists(txt_path):
        return []
    
    with open(txt_path, 'r') as f:
        folders = [line.strip() for line in f if line.strip()]
    return folders


def print_statistics(split_name, count, skipped, class_pixel_counts, total_pixels):
    """Print processing statistics"""
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} SET STATISTICS")
    print(f"{'='*70}")
    print(f"Images processed: {count}")
    print(f"Folders skipped: {skipped}")
    print(f"\nClass distribution (pixels):")
    
    for cls_name, pixel_count in sorted(class_pixel_counts.items(), 
                                       key=lambda x: x[1], reverse=True):
        percentage = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  {cls_name:10s}: {pixel_count:12,d} ({percentage:5.2f}%)")
    
    bg_pixels = total_pixels - sum(class_pixel_counts.values())
    bg_percentage = (bg_pixels / total_pixels * 100) if total_pixels > 0 else 0
    print(f"  {'Background':10s}: {bg_pixels:12,d} ({bg_percentage:5.2f}%)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # ========== CONFIGURATION ==========
    
    root_folder = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\cubicasa5k'
    train_txt = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\train.txt'
    val_txt = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\val.txt'
    test_txt = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\test.txt'
    
    output_mask_folder = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\MasksReal'
    output_image_folder = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\ImagesReal'

    class_map = {
        'Wall': 1,
        'Door': 2,
        'Window': 3,
        'Floor': 4,
    }

    print(f"{'='*70}")
    print("FLOOR PLAN MASK GENERATION (FIXED VERSION)")
    print(f"{'='*70}")
    print(f"Root folder: {root_folder}")
    print(f"Output masks: {output_mask_folder}")
    print(f"Output images: {output_image_folder}")
    print(f"\nClass mapping:")
    for cls_name, cls_id in class_map.items():
        print(f"  {cls_id}: {cls_name}")
    print(f"\nFixes applied:")
    print(f"  ✓ Wall thickness = 4 pixels")
    print(f"  ✓ Smart layering (doors/windows priority)")
    print(f"  ✓ Small object validation (<25px removed)")
    print(f"  ✓ Overflow prevention (Python int)")
    print(f"{'='*70}\n")

    if not check_prerequisites(root_folder, train_txt, val_txt, test_txt):
        print("\n❌ FAILED - Fix the issues above and try again\n")
        exit(1)

    train_folders = read_folders_from_txt(train_txt)
    val_folders = read_folders_from_txt(val_txt)
    test_folders = read_folders_from_txt(test_txt)

    print(f"Split sizes:")
    print(f"  Train: {len(train_folders)} folders")
    print(f"  Val:   {len(val_folders)} folders")
    print(f"  Test:  {len(test_folders)} folders\n")

    if len(train_folders) == 0 and len(val_folders) == 0 and len(test_folders) == 0:
        print("❌ ERROR: All split files are empty!")
        exit(1)

    train_count, train_skipped, train_stats = process_folders(
        train_folders, 'train', root_folder, output_mask_folder, 
        output_image_folder, class_map
    )
    
    val_count, val_skipped, val_stats = process_folders(
        val_folders, 'val', root_folder, output_mask_folder, 
        output_image_folder, class_map
    )
    
    test_count, test_skipped, test_stats = process_folders(
        test_folders, 'test', root_folder, output_mask_folder, 
        output_image_folder, class_map
    )

    total_train_pixels = sum(train_stats.values())
    total_val_pixels = sum(val_stats.values())
    total_test_pixels = sum(test_stats.values())

    if total_train_pixels > 0:
        print_statistics('train', train_count, train_skipped, train_stats, total_train_pixels)
    if total_val_pixels > 0:
        print_statistics('val', val_count, val_skipped, val_stats, total_val_pixels)
    if total_test_pixels > 0:
        print_statistics('test', test_count, test_skipped, test_stats, total_test_pixels)

    print(f"\n{'='*70}")
    print("GENERATING SAMPLE VISUALIZATIONS")
    print(f"{'='*70}")
    
    if train_count > 0:
        visualize_sample_mask(output_mask_folder, 'train', num_samples=3)
    if val_count > 0:
        visualize_sample_mask(output_mask_folder, 'val', num_samples=3)
    if test_count > 0:
        visualize_sample_mask(output_mask_folder, 'test', num_samples=3)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {train_count + val_count + test_count}")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")
    print(f"\nMasks saved to: {output_mask_folder}")
    print(f"Images saved to: {output_image_folder}")
    
    if train_count + val_count + test_count > 0:
        print(f"\n✓ SUCCESS! Dataset ready for training!")
        print(f"\nUpdate training script paths:")
        print(f"  img_dir_train = r'{os.path.join(output_image_folder, 'train')}'")
        print(f"  mask_dir_train = r'{os.path.join(output_mask_folder, 'train')}'")
        print(f"  img_dir_val = r'{os.path.join(output_image_folder, 'val')}'")
        print(f"  mask_dir_val = r'{os.path.join(output_mask_folder, 'val')}'")
        print(f"{'='*70}\n")
    
    print(f"{'='*70}\n")