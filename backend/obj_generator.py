import os
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LinearRing, box
from shapely.ops import unary_union
import trimesh
import traceback

# ================== CONFIG ==================
HEIGHT_SCALE = 1.5    # <--- Change this to increase overall height

PIXEL_SCALE = 0.01
MIN_AREA_PX = 50

WALL_THICKNESS_PX = 4
WALL_THICKNESS_M = WALL_THICKNESS_PX * PIXEL_SCALE

# ==== ALL HEIGHTS SCALED HERE =====
WALL_HEIGHT = 0.8 * HEIGHT_SCALE
WINDOW_BASE_HEIGHT = 0.4 * HEIGHT_SCALE
WINDOW_HEIGHT = 0.3 * HEIGHT_SCALE
DOOR_HEIGHT = 0.8 * HEIGHT_SCALE

# Frame thickness scaled
WINDOW_FRAME_THICKNESS = 1 * HEIGHT_SCALE
DOOR_FRAME_THICKNESS = 0.032 * HEIGHT_SCALE
GLASS_THICKNESS = 0.00001 * HEIGHT_SCALE
DOOR_LEAF_THICKNESS = 0.02 * HEIGHT_SCALE
WINDOW_SILL_DEPTH = 0.02 * HEIGHT_SCALE

DEFAULT_PAD = WALL_THICKNESS_M / 2.0

# Floor settings
FLOOR_THICKNESS = 0.02 * HEIGHT_SCALE
FLOOR_Z_OFFSET = -0.01 * HEIGHT_SCALE
FLOOR_BUFFER = 0.05

# Grid floor settings
TILE_SIZE = 0.3  # Size of each floor tile in meters
GROUT_WIDTH = 0.01  # Width of grout lines between tiles

# Colors RGBA (0-255)
COLORS = {
    "wall": np.array([180, 180, 180, 255], dtype=np.uint8),          # Soft neutral gray
    "door": np.array([165, 55, 45, 255], dtype=np.uint8),            # Deep brick red/brown
    "window_frame": np.array([255, 180, 30, 255], dtype=np.uint8),   # Bright amber/yellow-gold
    "glass": np.array([50, 170, 255, 120], dtype=np.uint8),          # Clear sky-blue translucent
    "floor_tile": np.array([220, 200, 170, 255], dtype=np.uint8),    # Light tile color
    "floor_grout": np.array([140, 130, 120, 255], dtype=np.uint8),   # Dark grout color
}

SIMPLIFY_TOLERANCE = 0.0005

# ================== FUNCTIONS ==================
def read_mask(mask_path):
    """Read mask robustly"""
    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read mask: {mask_path}")
    if img.ndim == 3:
        mask = img[..., 0].astype(np.uint8)
    else:
        mask = img.astype(np.uint8)
    return mask


def find_contours_for_class(mask_arr, cls_id):
    """Find contours for a specific class"""
    binary = (mask_arr == cls_id).astype(np.uint8)
    if binary.sum() == 0:
        return []
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = (binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= MIN_AREA_PX:
            pts = cnt[:, 0, :].astype(np.float64)
            if len(pts) >= 3:
                results.append(pts)
    return results


def contour_to_shapely(pts):
    """Convert contour to shapely polygon"""
    try:
        ring = LinearRing(pts)
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def scale_polygon(poly, scale):
    """Scale polygon from pixels to meters"""
    if poly is None:
        return None
    try:
        if poly.geom_type == "Polygon":
            exterior = [(x * scale, y * scale) for (x, y) in poly.exterior.coords]
            holes = [
                [(x * scale, y * scale) for (x, y) in h.coords] for h in poly.interiors
            ]
            return Polygon(exterior, holes)
        elif poly.geom_type == "MultiPolygon":
            scaled = []
            for p in poly.geoms:
                exterior = [(x * scale, y * scale) for (x, y) in p.exterior.coords]
                holes = [
                    [(x * scale, y * scale) for (x, y) in h.coords] for h in p.interiors
                ]
                scaled.append(Polygon(exterior, holes))
            return MultiPolygon(scaled)
    except Exception:
        return None
    return None


def extrude_polygon_with_color(poly, height, color):
    """Extrude polygon and apply color"""
    meshes = []
    if poly is None or poly.is_empty:
        return meshes
    
    # Handle both Polygon and MultiPolygon
    polys_to_process = []
    if poly.geom_type == "Polygon":
        polys_to_process = [poly]
    elif poly.geom_type == "MultiPolygon":
        polys_to_process = list(poly.geoms)
    
    for p in polys_to_process:
        try:
            mesh = trimesh.creation.extrude_polygon(p, height)
            try:
                mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            except Exception:
                pass
            meshes.append(mesh)
        except Exception:
            try:
                p2 = p.buffer(0)
                mesh = trimesh.creation.extrude_polygon(p2, height)
                try:
                    mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
                except Exception:
                    pass
                meshes.append(mesh)
            except Exception as e:
                print(f"Extrude failed: {e}")
    return meshes


def subtract_polygons_from_wall(wall_poly, subtract_polys):
    """Subtract all polygons (doors/windows) from one wall polygon."""
    result = wall_poly
    for p in subtract_polys:
        if p is not None and result.intersects(p):
            try:
                result = result.difference(p).buffer(0)
            except:
                pass
    return result


def create_solid_floor(floor_polys_m, wall_polys_m, verbose=False):
    """Create solid floor covering entire floorplan"""
    if not floor_polys_m:
        if verbose:
            print("No floor polygons, creating from walls...")
        if wall_polys_m:
            try:
                all_walls = unary_union(wall_polys_m)
                bounds = all_walls.bounds
                minx, miny, maxx, maxy = bounds
                floor_bbox = box(
                    minx - FLOOR_BUFFER,
                    miny - FLOOR_BUFFER,
                    maxx + FLOOR_BUFFER,
                    maxy + FLOOR_BUFFER,
                )
                return floor_bbox
            except Exception as e:
                if verbose:
                    print(f"Failed to create floor from walls: {e}")
                return None
        return None

    try:
        merged_floor = unary_union(floor_polys_m)
        if not merged_floor.is_valid:
            merged_floor = merged_floor.buffer(0)

        minx, miny, maxx, maxy = merged_floor.bounds
        floor_bbox = box(
            minx - FLOOR_BUFFER,
            miny - FLOOR_BUFFER,
            maxx + FLOOR_BUFFER,
            maxy + FLOOR_BUFFER,
        )

        solid_floor = unary_union([merged_floor, floor_bbox])

        if verbose:
            print(f"Floor area: {solid_floor.area:.2f} m²")

        return solid_floor

    except Exception as e:
        if verbose:
            print(f"Floor creation error: {e}")
        try:
            if floor_polys_m:
                return unary_union(floor_polys_m)
        except Exception:
            pass
        return None


def create_grid_floor(floor_poly, tile_size=TILE_SIZE, grout_width=GROUT_WIDTH, 
                     floor_thickness=FLOOR_THICKNESS, z_offset=FLOOR_Z_OFFSET, verbose=False):
    """
    Create a tiled floor with grid pattern (tiles + grout lines)
    """
    meshes = []
    
    if floor_poly is None or floor_poly.is_empty:
        return meshes
    
    try:
        # Get floor bounds
        minx, miny, maxx, maxy = floor_poly.bounds
        
        if verbose:
            print(f"Creating grid floor from ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")
        
        # Calculate number of tiles
        width = maxx - minx
        height = maxy - miny
        
        n_tiles_x = int(np.ceil(width / tile_size))
        n_tiles_y = int(np.ceil(height / tile_size))
        
        if verbose:
            print(f"Grid: {n_tiles_x} x {n_tiles_y} tiles")
        
        tile_count = 0
        
        # Create individual tiles
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                # Calculate tile position (with grout spacing)
                tile_x = minx + i * tile_size + grout_width/2
                tile_y = miny + j * tile_size + grout_width/2
                
                # Tile dimensions (reduced by grout width)
                tile_w = tile_size - grout_width
                tile_h = tile_size - grout_width
                
                # Create tile polygon
                tile_poly = box(tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)
                
                # Check if tile intersects with floor area
                if floor_poly.intersects(tile_poly):
                    # Clip tile to floor boundary
                    clipped_tile = tile_poly.intersection(floor_poly)
                    
                    if not clipped_tile.is_empty and clipped_tile.area > 0.001:
                        # Extrude tile
                        tile_meshes = extrude_polygon_with_color(
                            clipped_tile, 
                            floor_thickness, 
                            COLORS["floor_tile"]
                        )
                        
                        for tm in tile_meshes:
                            tm.apply_translation((0, 0, z_offset))
                            meshes.append(tm)
                            tile_count += 1
        
        # Create grout base (solid floor beneath tiles)
        grout_meshes = extrude_polygon_with_color(
            floor_poly,
            floor_thickness * 0.9,  # Slightly lower than tiles
            COLORS["floor_grout"]
        )
        
        for gm in grout_meshes:
            gm.apply_translation((0, 0, z_offset - floor_thickness * 0.1))
            meshes.insert(0, gm)  # Add grout first (below tiles)
        
        if verbose:
            print(f"✓ Grid floor created: {tile_count} tiles + grout base")
        
    except Exception as e:
        if verbose:
            print(f"Grid floor error: {e}")
            traceback.print_exc()
    
    return meshes


def create_window_frame_and_glass(
    wpoly,
    frame_thickness=WINDOW_FRAME_THICKNESS,
    frame_height=WINDOW_HEIGHT,
    base_height=WINDOW_BASE_HEIGHT,
    create_sill=True,
):
    """Build window frame (hollow - no glass pane) + sill"""
    meshes = []
    if wpoly is None or wpoly.is_empty:
        return meshes

    try:
        outer = wpoly.buffer(0)
    except:
        outer = wpoly

    # Inner cut
    try:
        inner = outer.buffer(-frame_thickness)
    except:
        inner = None

    if inner is None or inner.is_empty:
        try:
            inner = outer.buffer(-max(frame_thickness * 0.5, 0.001))
        except:
            inner = None

    # Frame = outer - inner (hollow frame only)
    if inner is not None and not inner.is_empty:
        try:
            frame = outer.difference(inner).buffer(0)
            fmeshes = extrude_polygon_with_color(
                frame, frame_height, COLORS["window_frame"]
            )
            for fm in fmeshes:
                fm.apply_translation((0, 0, base_height))
                meshes.append(fm)
        except Exception as e:
            print(f"Window frame error: {e}")

    # NO GLASS PANE - Window is completely hollow

    # Window sill
    if create_sill:
        try:
            minx, miny, maxx, maxy = outer.bounds
            sill_width = maxx - minx
            sill_depth = WINDOW_SILL_DEPTH
            sill_height = 0.02
            sill = trimesh.creation.box(extents=(sill_width, sill_depth, sill_height))
            cx = (minx + maxx) / 2.0
            sill.apply_translation(
                (cx, maxy + sill_depth / 2.0, base_height - sill_height / 2.0)
            )
            sill.visual.vertex_colors = np.tile(
                COLORS["window_frame"], (len(sill.vertices), 1)
            )
            meshes.append(sill)
        except:
            pass

    return meshes


def create_door_frame_and_leaf(
    dpoly,
    frame_thickness=DOOR_FRAME_THICKNESS,
    door_height=DOOR_HEIGHT,
    leaf_thickness=DOOR_LEAF_THICKNESS,
):
    """
    Create a FULLY SOLID DOOR (no hollow frame).
    The door polygon is extruded directly as a solid block.
    """
    meshes = []
    if dpoly is None or dpoly.is_empty:
        return meshes

    try:
        # Extrude the entire door polygon as a solid
        solid_meshes = extrude_polygon_with_color(
            dpoly, door_height, COLORS["door"]
        )
        meshes.extend(solid_meshes)

    except Exception as e:
        print(f"Solid door build error: {e}")

    return meshes


# =====================================================
# MAIN OBJ GENERATION FUNCTION
# =====================================================
def generate_obj_from_mask(mask_path, output_obj_path, verbose=True):
    """
    Generate 3D OBJ from segmentation mask with GRID FLOOR.
    """
    try:
        mask = read_mask(mask_path)
        if verbose:
            print(f"Mask read: {mask_path}, shape: {mask.shape}")
            print(f"Unique classes: {np.unique(mask)}")

        # Extract contours
        wall_conts = find_contours_for_class(mask, 1)
        door_conts = find_contours_for_class(mask, 2)
        window_conts = find_contours_for_class(mask, 3)
        floor_conts = find_contours_for_class(mask, 4)

        if verbose:
            print(
                f"Contours - walls: {len(wall_conts)}, doors: {len(door_conts)}, windows: {len(window_conts)}, floors: {len(floor_conts)}"
            )

        # Convert to shapely polygons (pixels)
        wall_polys = [contour_to_shapely(c) for c in wall_conts]
        wall_polys = [p for p in wall_polys if p]

        door_polys = [contour_to_shapely(c) for c in door_conts]
        door_polys = [p for p in door_polys if p]

        window_polys = [contour_to_shapely(c) for c in window_conts]
        window_polys = [p for p in window_polys if p]

        floor_polys = [contour_to_shapely(c) for c in floor_conts]
        floor_polys = [p for p in floor_polys if p]

        # Merge walls for processing
        if wall_polys:
            merged_walls = unary_union(wall_polys)
            wall_parts = [merged_walls] if merged_walls.geom_type == "Polygon" else list(merged_walls.geoms)
        else:
            wall_parts = []

        meshes = []

        # =====================================================
        # Process WALLS: Subtract doors and windows
        # =====================================================
        if verbose:
            print("Processing walls (subtracting openings)...")

        for wall_poly in wall_parts:
            # Collect doors and windows to subtract
            to_subtract = []
            
            for dpoly in door_polys:
                if wall_poly.contains(dpoly) or wall_poly.intersects(dpoly):
                    to_subtract.append(dpoly)
            
            for wpoly in window_polys:
                if wall_poly.contains(wpoly) or wall_poly.intersects(wpoly):
                    to_subtract.append(wpoly)

            # Subtract all openings
            wall_clean = subtract_polygons_from_wall(wall_poly, to_subtract)

            # Scale to meters
            wall_clean_m = scale_polygon(wall_clean, PIXEL_SCALE)

            if wall_clean_m and wall_clean_m.is_valid and not wall_clean_m.is_empty:
                # Extrude full wall height
                wall_meshes = extrude_polygon_with_color(wall_clean_m, WALL_HEIGHT, COLORS["wall"])
                meshes.extend(wall_meshes)

        if verbose:
            print(f"Wall meshes created: {len(meshes)}")

        # =====================================================
        # ADD WALL SEGMENTS BELOW WINDOWS (0 to window_bottom)
        # =====================================================
        if verbose:
            print(f"Adding wall segments below {len(window_polys)} windows...")

        window_bottom = WINDOW_BASE_HEIGHT

        for wpoly in window_polys:
            # Scale to meters
            wpoly_m = scale_polygon(wpoly, PIXEL_SCALE)
            
            if wpoly_m and wpoly_m.is_valid and not wpoly_m.is_empty:
                # Create wall segment from floor to window bottom
                bottom_meshes = extrude_polygon_with_color(
                    wpoly_m, window_bottom, COLORS["wall"]
                )
                meshes.extend(bottom_meshes)

        # =====================================================
        # ADD WALL SEGMENTS ABOVE WINDOWS (window_top to wall_height)
        # =====================================================
        window_top = WINDOW_BASE_HEIGHT + WINDOW_HEIGHT
        wall_above_height = WALL_HEIGHT - window_top

        if wall_above_height > 0:
            if verbose:
                print(f"Adding wall segments above windows (height: {wall_above_height}m)...")

            for wpoly in window_polys:
                # Scale to meters
                wpoly_m = scale_polygon(wpoly, PIXEL_SCALE)
                
                if wpoly_m and wpoly_m.is_valid and not wpoly_m.is_empty:
                    # Create wall segment above window
                    top_meshes = extrude_polygon_with_color(
                        wpoly_m, wall_above_height, COLORS["wall"]
                    )
                    for tm in top_meshes:
                        tm.apply_translation((0, 0, window_top))
                        meshes.append(tm)

        # =====================================================
        # WINDOWS (hollow frames only)
        # =====================================================
        if verbose:
            print(f"Adding {len(window_polys)} window frames (hollow)...")

        for wpoly in window_polys:
            wpoly_m = scale_polygon(wpoly, PIXEL_SCALE)
            
            if wpoly_m and wpoly_m.is_valid and not wpoly_m.is_empty:
                try:
                    wmeshes = create_window_frame_and_glass(
                        wpoly_m,
                        frame_thickness=WINDOW_FRAME_THICKNESS,
                        frame_height=WINDOW_HEIGHT,
                        base_height=WINDOW_BASE_HEIGHT,
                        create_sill=True,
                    )
                    meshes.extend(wmeshes)

                    if verbose and wmeshes:
                        print(f"  ✓ Hollow window created ({len(wmeshes)} parts)")

                except Exception as e:
                    if verbose:
                        print(f"  ✗ Window build error: {e}")

        # =====================================================
        # DOORS (frame + leaf)
        # =====================================================
        if verbose:
            print(f"Adding {len(door_polys)} doors...")

        for dpoly in door_polys:
            dpoly_m = scale_polygon(dpoly, PIXEL_SCALE)
            
            if dpoly_m and dpoly_m.is_valid and not dpoly_m.is_empty:
                try:
                    dmeshes = create_door_frame_and_leaf(
                        dpoly_m,
                        frame_thickness=DOOR_FRAME_THICKNESS,
                        door_height=DOOR_HEIGHT,
                        leaf_thickness=DOOR_LEAF_THICKNESS,
                    )
                    meshes.extend(dmeshes)

                except Exception as e:
                    if verbose:
                        print(f"Door build error: {e}")

        # =====================================================
        # GRID FLOOR
        # =====================================================
        if verbose:
            print("Creating grid floor...")

        floor_polys_m = [scale_polygon(p, PIXEL_SCALE) for p in floor_polys]
        floor_polys_m = [p for p in floor_polys_m if p]

        wall_polys_m = [scale_polygon(p, PIXEL_SCALE) for p in wall_polys]
        wall_polys_m = [p for p in wall_polys_m if p]

        solid_floor = create_solid_floor(floor_polys_m, wall_polys_m, verbose=verbose)

        if solid_floor:
            try:
                grid_floor_meshes = create_grid_floor(
                    solid_floor,
                    tile_size=TILE_SIZE,
                    grout_width=GROUT_WIDTH,
                    floor_thickness=FLOOR_THICKNESS,
                    z_offset=FLOOR_Z_OFFSET,
                    verbose=verbose
                )
                meshes.extend(grid_floor_meshes)
                
                if verbose:
                    print("✓ Grid floor created")
            except Exception as e:
                if verbose:
                    print(f"Floor mesh error: {e}")

        # =====================================================
        # EXPORT
        # =====================================================
        if not meshes:
            if verbose:
                print("No meshes generated.")
            return False

        if verbose:
            print(f"Combining {len(meshes)} meshes...")

        combined = trimesh.util.concatenate(meshes)

        output_dir = os.path.dirname(output_obj_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        combined.export(output_obj_path)

        if verbose:
            print(f"Export successful: {output_obj_path}")
            print(
                f"   Vertices: {len(combined.vertices):,}, Faces: {len(combined.faces):,}"
            )

        return True

    except Exception as e:
        print(f"❌ OBJ generation failed: {e}")
        traceback.print_exc()
        return False


# =====================================================
# CLI ENTRY
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate OBJ from segmentation mask")
    parser.add_argument("mask", type=str, help="Input mask png path")
    parser.add_argument("out", type=str, help="Output OBJ path")
    args = parser.parse_args()
    ok = generate_obj_from_mask(args.mask, args.out, verbose=True)
    if ok:
        print("Done.")
    else:
        print("❌ Failed.")