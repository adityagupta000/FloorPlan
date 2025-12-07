import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

SVG_NS = "http://www.w3.org/2000/svg"
NS_MAP = {'svg': SVG_NS}

def parse_svg_with_group_classes(svg_path, classes_to_extract):
    """
    Parses SVG with polygons nested inside <g> groups having class names.
    Extracts polygons with their parent's class attribute as their class name.
    
    FIXED: High precision coordinate storage (6 decimals instead of 2)
    
    Classes to extract:
    - Wall, Door, Window: Extracted as-is
    - Any class starting with 'Space': Extracted and mapped to 'Floor'
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    polygons = []
    classes_to_extract_lower = {cls.lower() for cls in classes_to_extract}

    def recurse_group(group_elem, current_class=None):
        group_class_attr = group_elem.get('class', None)
        group_class = None

        if group_class_attr:
            for cls in group_class_attr.split():
                cls_lower = cls.lower()
                if cls_lower in classes_to_extract_lower or cls_lower.startswith('space'):
                    group_class = cls
                    break

        if group_class is None:
            group_class = current_class

        for elem in group_elem:
            if elem.tag == f'{{{SVG_NS}}}polygon':
                if group_class:
                    cls_lower = group_class.lower()
                    if cls_lower in classes_to_extract_lower or cls_lower.startswith('space'):
                        points_str = elem.get('points')
                        if points_str:
                            pts = []
                            for pair in points_str.strip().split():
                                if ',' not in pair:
                                    continue
                                try:
                                    x, y = map(float, pair.split(','))
                                    pts.append((x, y))
                                except ValueError:
                                    continue
                            if pts:
                                # Normalize: all Space* -> 'Floor'
                                if group_class.lower().startswith('space'):
                                    normalized_name = 'Floor'
                                else:
                                    normalized_name = group_class
                                
                                polygons.append({
                                    'name': normalized_name,
                                    'points': pts,
                                    'original_name': group_class
                                })

            elif elem.tag == f'{{{SVG_NS}}}g':
                recurse_group(elem, group_class)

    recurse_group(root)
    return polygons


def create_annotation_xml(polygons, output_path):
    """
    Creates annotation XML with standardized class names.
    
    FIXED: High precision coordinates (6 decimals instead of 2)
    Classes: Wall, Door, Window, Floor (all Space* classes become Floor)
    """
    annotation = ET.Element('annotation')

    for poly in polygons:
        obj = ET.SubElement(annotation, 'object')

        # Use normalized name (Floor instead of Space*)
        ET.SubElement(obj, 'name').text = poly['name']
        ET.SubElement(obj, 'pose').text = 'v'

        points_el = ET.SubElement(obj, 'points')
        for x, y in poly['points']:
            point_el = ET.SubElement(points_el, 'point')
            # ✅ FIXED: 6 decimal precision (was 2)
            ET.SubElement(point_el, 'x').text = f"{x:.6f}"
            ET.SubElement(point_el, 'y').text = f"{y:.6f}"

        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        # Bounding box
        xs = [p[0] for p in poly['points']]
        ys = [p[1] for p in poly['points']]
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = f"{min(xs):.6f}"
        ET.SubElement(bndbox, 'ymin').text = f"{min(ys):.6f}"
        ET.SubElement(bndbox, 'xmax').text = f"{max(xs):.6f}"
        ET.SubElement(bndbox, 'ymax').text = f"{max(ys):.6f}"

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def process_folder(folder_path, classes_to_extract):
    """Process a single folder containing model.svg"""
    svg_file = os.path.join(folder_path, 'model.svg')
    if not os.path.exists(svg_file):
        return False, 0

    polygons = parse_svg_with_group_classes(svg_file, classes_to_extract)
    if not polygons:
        return False, 0

    # Count polygons by class
    class_counts = {}
    for poly in polygons:
        cls = poly['name']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    xml_path = os.path.join(folder_path, 'annotation.xml')
    create_annotation_xml(polygons, xml_path)
    
    return True, class_counts


def process_root_directory(root_dir, classes_to_extract):
    """Process all folders in root directory with progress bar"""
    # First, count total folders
    svg_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'model.svg' in filenames:
            svg_folders.append(dirpath)
    
    print(f"Found {len(svg_folders)} folders with model.svg")
    
    count = 0
    total_class_counts = {}
    failed_folders = []

    for dirpath in tqdm(svg_folders, desc="Converting SVG to XML"):
        success, class_counts = process_folder(dirpath, classes_to_extract)
        if success:
            count += 1
            # Accumulate class counts
            for cls, cnt in class_counts.items():
                total_class_counts[cls] = total_class_counts.get(cls, 0) + cnt
        else:
            failed_folders.append(dirpath)

    print(f"\n{'='*70}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {count} folders")
    print(f"✗ Failed: {len(failed_folders)} folders")
    print(f"\nTotal polygons extracted by class:")
    for cls, cnt in sorted(total_class_counts.items()):
        print(f"  {cls:10s}: {cnt:,} polygons")
    print(f"{'='*70}\n")

    if failed_folders and len(failed_folders) <= 10:
        print("Failed folders:")
        for folder in failed_folders:
            print(f"  - {folder}")
    elif len(failed_folders) > 10:
        print(f"Failed folders (showing first 10 of {len(failed_folders)}):")
        for folder in failed_folders[:10]:
            print(f"  - {folder}")

    return count, len(failed_folders)


if __name__ == '__main__':
    # ========== CONFIGURATION ==========
    
    # Classes to extract - matches training code requirements
    # All Space* classes will automatically be converted to 'Floor'
    classes = {
        "Wall",    # Class 1 in training
        "Door",    # Class 2 in training
        "Window",  # Class 3 in training
        # Space* classes will be converted to "Floor" (Class 4 in training)
    }

    # ✅ UPDATE THIS PATH to your CubiCasa5k dataset location
    root_folder = r'C:\Users\Aditya\OneDrive\Documents\final_project_base\cubicasa5k'
    
    # ===================================

    print(f"{'='*70}")
    print("SVG TO XML CONVERTER (FIXED VERSION)")
    print(f"{'='*70}")
    print(f"Root folder: {root_folder}")
    print(f"Extracting classes: {', '.join(classes)}")
    print(f"All 'Space*' classes will be converted to 'Floor'")
    print(f"Coordinate precision: 6 decimals (high precision)")
    print(f"{'='*70}\n")
    
    if not os.path.exists(root_folder):
        print(f"❌ ERROR: Root folder does not exist!")
        print(f"   Path: {root_folder}")
        print(f"\n   Please update the 'root_folder' variable in this script.")
        exit(1)
    
    success_count, failed_count = process_root_directory(root_folder, classes)
    
    if success_count > 0:
        print(f"\n✓ SUCCESS! {success_count} XML annotation files created.")
        print(f"  Next step: Run mask_generation.py to create segmentation masks")
    else:
        print(f"\n❌ FAILED! No XML files were created.")
        print(f"   Please check your SVG files and folder structure.")