import os
import json
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

# Path to the folder containing the .tif files
TIF_DIR = os.path.join(os.path.dirname(__file__), '../snow_monthly_output_2018_to_2021/snod/2019')
TIF_DIR = os.path.abspath(TIF_DIR)
OUT_DIR = "./Json"
os.makedirs(OUT_DIR, exist_ok=True)


def get_tif_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith('.tif')]

def reproject_to_epsg4326(src_path, dst_path):
    with rasterio.open(src_path) as src:
        dst_crs = 'EPSG:4326'
        if src.crs == CRS.from_string(dst_crs):
            # Already in EPSG:4326
            return src_path
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        return dst_path

def get_bounds_epsg4326(tif_path):
    """Get bounds in EPSG:4326 format"""
    with rasterio.open(tif_path) as src:
        if src.crs == CRS.from_string('EPSG:4326'):
            return src.bounds
        else:
            # Need to reproject to get proper bounds
            temp_dir = os.path.join(os.path.dirname(tif_path), 'temp_bounds')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, 'temp_' + os.path.basename(tif_path))
            reproject_to_epsg4326(tif_path, temp_file)
            with rasterio.open(temp_file) as temp_src:
                bounds = temp_src.bounds
            # Clean up temp file
            os.remove(temp_file)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            return bounds

def create_bounds_geojson(tif_files_dir, output_geojson_path):
    """Create a GeoJSON file with bounds as a Polygon geometry"""
    tif_files = get_tif_files(tif_files_dir)
    
    if not tif_files:
        print("No TIF files found!")
        return
    
    # Get bounds from the first file (since all have same bounds)
    first_tif = os.path.join(tif_files_dir, tif_files[0])
    bounds = get_bounds_epsg4326(first_tif)
    
    # Create GeoJSON Polygon from bounds
    # GeoJSON coordinates are [longitude, latitude] (x, y)
    # Polygon coordinates: [[exterior_ring]]
    # Rectangle: bottom-left, bottom-right, top-right, top-left, bottom-left (closed)
    polygon_coords = [[
        [bounds.left, bounds.bottom],   # bottom-left (west, south)
        [bounds.right, bounds.bottom],  # bottom-right (east, south)
        [bounds.right, bounds.top],     # top-right (east, north)
        [bounds.left, bounds.top],      # top-left (west, north)
        [bounds.left, bounds.bottom]    # close the polygon
    ]]
    
    # Create GeoJSON structure
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "TIF Files Bounding Box",
                    "total_files": len(tif_files),
                    "sample_file": os.path.basename(first_tif),
                    "bounds": {
                        "west": bounds.left,
                        "south": bounds.bottom,
                        "east": bounds.right,
                        "north": bounds.top
                    },
                    "bounds_array": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                    "description": "Bounding box for all TIF files in the dataset"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords
                }
            }
        ]
    }
    
    # Save to GeoJSON file
    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"GeoJSON bounds saved to: {output_geojson_path}")
    print(f"Bounds: West={bounds.left:.6f}, South={bounds.bottom:.6f}, East={bounds.right:.6f}, North={bounds.top:.6f}")
    
    return geojson_data

def main():
    output_geojson = os.path.join(OUT_DIR, 'tif_bounds.geojson')
    bounds_data = create_bounds_geojson(TIF_DIR, output_geojson)
    
    # Optional: verify all files have same bounds
    print("\nVerifying all files have the same bounds...")
    tif_files = get_tif_files(TIF_DIR)
    reference_bounds = bounds_data['features'][0]['properties']['bounds_array']
    
    all_same = True
    for tif_file in tif_files[:5]:  # Check first 5 files as sample
        tif_path = os.path.join(TIF_DIR, tif_file)
        current_bounds = get_bounds_epsg4326(tif_path)
        current_bounds_list = [current_bounds.left, current_bounds.bottom, 
                              current_bounds.right, current_bounds.top]
        
        # Compare with small tolerance for floating point differences
        if not all(abs(a - b) < 1e-10 for a, b in zip(reference_bounds, current_bounds_list)):
            print(f"Warning: {tif_file} has different bounds!")
            all_same = False
    
    if all_same:
        print("✓ All checked files have the same bounds")

if __name__ == "__main__":
    main()