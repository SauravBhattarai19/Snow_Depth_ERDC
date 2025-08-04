import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import json
import geemap
import datetime
import os
import zipfile
import tempfile
import io
from typing import List, Dict, Any, Optional
import math
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.merge import merge
import numpy as np
import requests
from dateutil.relativedelta import relativedelta

# Try to import the separate auth module if available
try:
    from Tool.gee_auth import GEEAuth
except ImportError:
    # Define GEEAuth class inline if module not available
    class GEEAuth:
        """Class to handle Google Earth Engine authentication."""
        
        def __init__(self):
            """Initialize the GEE authentication handler."""
            self._initialized = False
        
        def initialize(self, project_id: str, credentials_content: Optional[str] = None) -> bool:
            """
            Initialize the Earth Engine API with the provided credentials.
            
            Args:
                project_id: The Google Cloud project ID
                credentials_content: Optional credentials file content for web apps
                
            Returns:
                bool: True if authentication was successful, False otherwise
            """
            try:
                # Method: Credentials file content (for web apps with file upload)
                if credentials_content:
                    # Handle uploaded credentials content
                    try:
                        # Validate that the JSON parses correctly
                        creds_data = json.loads(credentials_content)

                        # Check if it's a service account key (has private_key)
                        if "private_key" in creds_data and "client_email" in creds_data:
                            # Temporarily write to disk and use ServiceAccountCredentials
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_sa:
                                tmp_sa.write(credentials_content)
                                tmp_sa_path = tmp_sa.name

                            try:
                                credentials = ee.ServiceAccountCredentials(creds_data["client_email"], tmp_sa_path)
                                ee.Initialize(credentials, project=project_id)
                            finally:
                                if os.path.exists(tmp_sa_path):
                                    os.unlink(tmp_sa_path)
                        else:
                            # This is a user OAuth credentials file
                            # Persist it to ~/.config/earthengine/ so subsequent sessions work
                            ee_creds_dir = os.path.expanduser("~/.config/earthengine")
                            os.makedirs(ee_creds_dir, exist_ok=True)

                            ee_creds_path = os.path.join(ee_creds_dir, "credentials")

                            try:
                                with open(ee_creds_path, "w", encoding="utf-8") as f:
                                    f.write(credentials_content)
                            except Exception as write_err:
                                raise Exception(f"Unable to write credentials file: {write_err}")

                            # Initialize EE with the credentials
                            ee.Initialize(project=project_id)

                    except json.JSONDecodeError:
                        raise Exception("Invalid credentials file format. Please upload a valid Earth Engine credentials file.")
                
                # Try default authentication if no credentials provided
                else:
                    try:
                        ee.Initialize(project=project_id)
                    except Exception as e:
                        raise Exception(
                            "Authentication failed. Please provide:\n"
                            "1. Your Google Cloud Project ID\n"
                            "2. Your Earth Engine credentials file\n\n"
                            "To get credentials: Run 'earthengine authenticate' locally, then upload the credentials file from ~/.config/earthengine/credentials"
                        )
                    
                # Test the connection
                if self.test_connection():
                    self._initialized = True
                    return True
                else:
                    raise Exception("Authentication succeeded but connection test failed")
                    
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                self._initialized = False
                return False
                
        def is_initialized(self) -> bool:
            """Check if Earth Engine has been initialized."""
            return self._initialized
        
        @staticmethod
        def test_connection() -> bool:
            """Test the connection to Earth Engine by making a simple API call."""
            try:
                ee.Image("USGS/SRTMGL1_003").getInfo()
                return True
            except Exception:
                return False

class SnowDepthCalculator:
    def __init__(self, scale: int = 500):
        """Initialize the Snow Depth Calculator
        
        Args:
            scale: Pixel resolution in meters (default: 500)
        """
        self.scale = scale
        
    def calculate_monthly_snow_depth(self, aoi, year: int, month: int):
        """Calculate snow depth for a specific month and year"""
        # Create date range for the month
        start_date = datetime.date(year, month, 1)
        if month == 12:
            end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # MODIS Snow Cover
        snowExtent = ee.ImageCollection('MODIS/061/MOD10A1') \
            .select("NDSI_Snow_Cover") \
            .filterDate(start_date_str, end_date_str) \
            .filterBounds(aoi)
        
        meanNDSI = snowExtent.mean().divide(100)
        fsc = meanNDSI.multiply(1.45).subtract(0.01).clamp(0, 1)
        
        # GLDAS SWE
        gldasSWE = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .select("SWE_inst") \
            .filterDate(start_date_str, end_date_str) \
            .filterBounds(aoi) \
            .mean()
        
        # Resample GLDAS
        modisProjection = meanNDSI.projection()
        gldasSWE_resampled = gldasSWE \
            .resample("bilinear") \
            .reproject(crs=modisProjection, scale=self.scale)
        
        # Snow density calculation
        rho_min = 100
        rho_max = 500
        snowDensity = fsc.expression(
            "rho_min + (rho_max - rho_min) * (1 - FSC)", {
                "rho_min": rho_min,
                "rho_max": rho_max,
                "FSC": fsc
            })
        
        # Final calculations
        fineSWE = gldasSWE_resampled.multiply(fsc)
        snowDepth = fineSWE.divide(snowDensity.add(1e-6))
        
        return snowDepth.clip(aoi)

def initialize_earth_engine(project_id: str, credentials_content: Optional[str] = None) -> bool:
    """Initialize Google Earth Engine with user-provided credentials"""
    if not project_id:
        st.error("Please provide a Google Cloud Project ID")
        return False
        
    auth = GEEAuth()
    return auth.initialize(project_id, credentials_content)

def generate_month_list(start_date: datetime.date, end_date: datetime.date) -> List[tuple]:
    """Generate list of (year, month) tuples for the date range"""
    months = []
    current_date = start_date.replace(day=1)  # Start from first day of month
    
    while current_date <= end_date:
        months.append((current_date.year, current_date.month))
        # Add one month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return months

def create_chunks(bounds: List[float], max_pixels: int = 50000000) -> List[List[float]]:
    """Create chunks from bounds to avoid memory limits
    
    Args:
        bounds: [min_lon, min_lat, max_lon, max_lat]
        max_pixels: Maximum number of pixels per chunk
        
    Returns:
        List of chunk bounds
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Estimate area in degrees
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    # Estimate number of pixels at 500m resolution
    # Rough approximation: 1 degree ≈ 111 km
    approx_pixels = (lon_range * 111000) * (lat_range * 111000) / (500 * 500)
    
    if approx_pixels <= max_pixels:
        return [bounds]
    
    # Calculate number of chunks needed
    chunks_needed = math.ceil(approx_pixels / max_pixels)
    chunks_per_side = math.ceil(math.sqrt(chunks_needed))
    
    chunks = []
    lon_step = lon_range / chunks_per_side
    lat_step = lat_range / chunks_per_side
    
    for i in range(chunks_per_side):
        for j in range(chunks_per_side):
            chunk_min_lon = min_lon + i * lon_step
            chunk_max_lon = min_lon + (i + 1) * lon_step
            chunk_min_lat = min_lat + j * lat_step
            chunk_max_lat = min_lat + (j + 1) * lat_step
            
            chunks.append([chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat])
    
    return chunks

def download_chunk(calculator: SnowDepthCalculator, chunk_bounds: List[float], 
                  chunk_id: int, year: int, month: int, output_dir: str) -> str:
    """Download a single chunk for a specific month
    
    Args:
        calculator: SnowDepthCalculator instance
        chunk_bounds: [min_lon, min_lat, max_lon, max_lat]
        chunk_id: Unique identifier for the chunk
        year: Year for the calculation
        month: Month for the calculation
        output_dir: Directory to save the chunk
        
    Returns:
        Path to the downloaded file
    """
    # Create geometry from bounds
    geometry = ee.Geometry.Rectangle(chunk_bounds)
    
    # Calculate snow depth for specific month
    snow_depth = calculator.calculate_monthly_snow_depth(geometry, year, month)
    
    # Create download URL
    url = snow_depth.getDownloadURL({
        'scale': calculator.scale,
        'crs': 'EPSG:4326',
        'region': geometry,
        'format': 'GEO_TIFF'
    })
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()
    
    chunk_path = os.path.join(output_dir, f'snow_depth_{year}_{month:02d}_chunk_{chunk_id}.tif')
    with open(chunk_path, 'wb') as f:
        f.write(response.content)
    
    return chunk_path

def merge_chunks(chunk_paths: List[str], output_path: str):
    """Merge multiple GeoTIFF chunks into a single file"""
    if len(chunk_paths) == 1:
        # If only one chunk, just copy it
        import shutil
        shutil.copy2(chunk_paths[0], output_path)
        return
    
    # Open all chunk files
    src_files_to_mosaic = []
    for chunk_path in chunk_paths:
        src = rasterio.open(chunk_path)
        src_files_to_mosaic.append(src)
    
    # Merge the chunks
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Get metadata from the first chunk
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # Write the merged file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all source files
    for src in src_files_to_mosaic:
        src.close()

def process_monthly_data(calculator: SnowDepthCalculator, aoi_geom, 
                        months_list: List[tuple], progress_bar, status_text) -> bytes:
    """Process AOI and generate monthly snow depth data as a zip file"""
    
    # Get bounds from geometry
    if hasattr(aoi_geom, 'bounds'):
        bounds = list(aoi_geom.bounds().getInfo()['coordinates'][0])
        # Convert from polygon coordinates to bounds
        lons = [coord[0] for coord in bounds]
        lats = [coord[1] for coord in bounds]
        bounds = [min(lons), min(lats), max(lons), max(lats)]
    else:
        # Assume it's already in bounds format
        bounds = aoi_geom
    
    # Create chunks
    chunks = create_chunks(bounds)
    total_tasks = len(months_list) * len(chunks)
    current_task = 0
    
    # Create in-memory zip buffer
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Process each month
            for year, month in months_list:
                month_name = datetime.date(year, month, 1).strftime('%Y_%m')
                status_text.text(f"Processing {month_name} ({len(chunks)} chunks)...")
                
                chunk_paths = []
                
                # Process each chunk for this month
                for chunk_id, chunk_bounds in enumerate(chunks):
                    try:
                        current_task += 1
                        progress_bar.progress(current_task / total_tasks)
                        
                        chunk_path = download_chunk(
                            calculator, chunk_bounds, chunk_id, year, month, temp_dir
                        )
                        chunk_paths.append(chunk_path)
                        
                    except Exception as e:
                        st.warning(f"Error processing chunk {chunk_id+1} for {month_name}: {str(e)}")
                        continue
                
                if not chunk_paths:
                    st.warning(f"No chunks processed successfully for {month_name}")
                    continue
                
                # Merge chunks for this month
                status_text.text(f"Merging chunks for {month_name}...")
                monthly_output_path = os.path.join(temp_dir, f'snow_depth_{month_name}.tif')
                merge_chunks(chunk_paths, monthly_output_path)
                
                # Add to zip file
                status_text.text(f"Adding {month_name} to zip...")
                with open(monthly_output_path, 'rb') as f:
                    zip_file.writestr(f'snow_depth_{month_name}.tif', f.read())
                
                # Clean up chunk files
                for chunk_path in chunk_paths:
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
    
    return zip_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="Snow Depth Calculator",
        page_icon="❄️",
        layout="wide"
    )
    
    st.title("❄️ Snow Depth Calculator - Monthly Analysis")
    st.markdown("Calculate monthly snow depth using MODIS and GLDAS data from Google Earth Engine")
    
    # Authentication Section
    st.header("🔐 Authentication")
    
    # Check if already authenticated
    ee_initialized = False
    if 'ee_initialized' in st.session_state:
        ee_initialized = st.session_state.ee_initialized
    
    if not ee_initialized:
        st.info("Please authenticate with Google Earth Engine to continue")
        
        # Create authentication form
        with st.form("auth_form"):
            st.subheader("Google Earth Engine Authentication")
            
            # Project ID input
            project_id = st.text_input(
                "Google Cloud Project ID *",
                placeholder="your-project-id",
                help="Enter your Google Cloud Project ID that has Earth Engine enabled"
            )
            
            # Credentials file upload
            credentials_file = st.file_uploader(
                "Upload Earth Engine Credentials File *",
                type=None,  # Accept any file type
                help="Upload your Earth Engine credentials file. This file may have no extension. See instructions below for how to obtain this file."
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Authenticate", type="primary")
            
            if submitted:
                if not project_id:
                    st.error("Please provide a Google Cloud Project ID")
                elif not credentials_file:
                    st.error("Please upload your credentials file")
                else:
                    # Read credentials file
                    credentials_content = credentials_file.read().decode('utf-8')
                    
                    # Attempt authentication
                    with st.spinner("Authenticating with Google Earth Engine..."):
                        if initialize_earth_engine(project_id, credentials_content):
                            st.success("✅ Successfully authenticated with Google Earth Engine!")
                            st.session_state.ee_initialized = True
                            st.session_state.project_id = project_id
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Please check your credentials and try again.")
        
        # Instructions for getting credentials
        with st.expander("📋 How to get your credentials file"):
            st.markdown("""
            ### Step 1: Get Google Earth Engine Access
            1. Sign up for Google Earth Engine at [https://earthengine.google.com/](https://earthengine.google.com/)
            2. Wait for approval (this can take a few days for new users)
            
            ### Step 2: Create a Google Cloud Project
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the Earth Engine API for your project
            4. Note down your Project ID
            
            ### Step 3: Get Credentials File
            
            **Option A: Service Account Key (Recommended for applications)**
            1. In Google Cloud Console, go to "IAM & Admin" → "Service Accounts"
            2. Create a new service account or select an existing one
            3. Add the "Earth Engine Resource Viewer" role
            4. Create a JSON key for the service account
            5. Download the JSON file - this is your credentials file
            
            **Option B: User Credentials (For personal use)**
            1. Install Earth Engine CLI: `pip install earthengine-api`
            2. Run: `earthengine authenticate`
            3. Follow the authentication flow in your browser
            4. Find the credentials file at:
               - **Linux/Mac**: `~/.config/earthengine/credentials`
               - **Windows**: `C:\\Users\\[username]\\.config\\earthengine\\credentials`
            5. Upload this file
            
            ### Troubleshooting
            - Make sure your project has Earth Engine API enabled
            - Verify you have the necessary permissions
            - For service accounts, ensure the Earth Engine Resource Viewer role is assigned
            """)
        
        st.divider()
        st.markdown("*Complete authentication above to access the Snow Depth Calculator*")
        return
    
    # Show authenticated status
    st.success(f"✅ Authenticated with project: `{st.session_state.get('project_id', 'Unknown')}`")
    
    # Add logout button
    if st.button("🔓 Logout", help="Clear authentication and start over"):
        for key in ['ee_initialized', 'project_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Main application content starts here
    st.header("📊 Monthly Snow Depth Analysis")
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    # Date inputs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2020, 1, 1),
            max_value=datetime.date.today()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.date(2023, 12, 31),
            max_value=datetime.date.today()
        )
    
    # Scale input
    scale = st.sidebar.number_input(
        "Resolution (meters)",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        help="Pixel resolution in meters. Lower values = higher resolution but longer processing time."
    )
    
    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        st.stop()
    
    # Generate month list and show info
    months_list = generate_month_list(start_date, end_date)
    st.sidebar.info(f"📅 **Total months to process:** {len(months_list)}")
    st.sidebar.markdown(f"**Date range:** {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Define Study Area")
        
        # AOI input method
        aoi_method = st.radio(
            "Choose method to define Area of Interest:",
            ["Draw on Map", "Upload JSON File"]
        )
        
        aoi_geom = None
        
        if aoi_method == "Draw on Map":
            st.markdown("**Instructions:** Use the drawing tools to create a polygon on the map")
            
            # Create folium map
            m = folium.Map(location=[40.0, -100.0], zoom_start=4)
            
            # Add drawing tools
            from folium.plugins import Draw
            draw = Draw(
                export=True,
                filename='data.geojson',
                position='topleft',
                draw_options={
                    'polyline': False,
                    'rectangle': True,
                    'polygon': True,
                    'circle': False,
                    'marker': False,
                    'circlemarker': False,
                },
                edit_options={'edit': False}
            )
            draw.add_to(m)
            
            # Display map
            map_data = st_folium(m, width=700, height=500)
            
            # Process drawn features
            if map_data['all_drawings']:
                if len(map_data['all_drawings']) > 0:
                    # Get the last drawn feature
                    feature = map_data['all_drawings'][-1]
                    if feature['geometry']['type'] in ['Polygon', 'Rectangle']:
                        coords = feature['geometry']['coordinates'][0]
                        
                        # Convert to ee.Geometry
                        if feature['geometry']['type'] == 'Rectangle':
                            # For rectangles, we need to handle the coordinate format
                            lons = [coord[0] for coord in coords]
                            lats = [coord[1] for coord in coords]
                            aoi_geom = ee.Geometry.Rectangle([min(lons), min(lats), max(lons), max(lats)])
                        else:
                            aoi_geom = ee.Geometry.Polygon(coords)
                        
                        st.success("✅ Area of Interest defined!")
                    else:
                        st.warning("Please draw a polygon or rectangle")
        
        else:  # Upload JSON File
            uploaded_file = st.file_uploader(
                "Upload GeoJSON file",
                type=['json', 'geojson'],
                help="Upload a GeoJSON file containing your area of interest"
            )
            
            if uploaded_file is not None:
                try:
                    geojson_data = json.load(uploaded_file)
                    
                    # Extract geometry
                    if 'features' in geojson_data:
                        # FeatureCollection
                        feature = geojson_data['features'][0]
                        geometry = feature['geometry']
                    elif 'geometry' in geojson_data:
                        # Feature
                        geometry = geojson_data['geometry']
                    else:
                        # Geometry
                        geometry = geojson_data
                    
                    # Convert to ee.Geometry
                    aoi_geom = ee.Geometry(geometry)
                    st.success("✅ GeoJSON file uploaded successfully!")
                    
                    # Display uploaded area on map
                    bounds = aoi_geom.bounds().getInfo()
                    coords = bounds['coordinates'][0]
                    lons = [coord[0] for coord in coords]
                    lats = [coord[1] for coord in coords]
                    center_lat = (min(lats) + max(lats)) / 2
                    center_lon = (min(lons) + max(lons)) / 2
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
                    folium.GeoJson(geojson_data).add_to(m)
                    st_folium(m, width=700, height=300)
                    
                except Exception as e:
                    st.error(f"Error processing GeoJSON file: {str(e)}")
    
    with col2:
        st.subheader("Processing")
        
        if aoi_geom is not None:
            st.success("Ready to process!")
            
            # Show area info
            try:
                area_km2 = aoi_geom.area().divide(1000000).getInfo()
                st.metric("Area", f"{area_km2:.2f} km²")
            except:
                st.info("Area calculation not available")
            
            # Show number of files that will be generated
            st.metric("Files to generate", f"{len(months_list)} GeoTIFF files")
            
            # Processing button
            if st.button("🚀 Generate Monthly Snow Depth", type="primary"):
                # Create calculator
                calculator = SnowDepthCalculator(scale=scale)
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Process and create zip
                    with st.spinner("Processing monthly data..."):
                        zip_data = process_monthly_data(
                            calculator, aoi_geom, months_list, progress_bar, status_text
                        )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    # Provide download button
                    filename = f"snow_depth_monthly_{start_date.strftime('%Y%m')}_{end_date.strftime('%Y%m')}_{scale}m.zip"
                    st.download_button(
                        label="📥 Download Monthly Snow Depth Data (ZIP)",
                        data=zip_data,
                        file_name=filename,
                        mime="application/zip"
                    )
                    
                    st.success(f"Monthly snow depth calculation completed! Generated {len(months_list)} files.")
                    
                    # Show what's in the zip
                    with st.expander("📋 Files included in the ZIP"):
                        for year, month in months_list:
                            month_name = datetime.date(year, month, 1).strftime('%Y_%m')
                            st.text(f"snow_depth_{month_name}.tif")
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.info("Try reducing the area size, date range, or increasing the resolution scale.")
        
        else:
            st.info("Please define an Area of Interest first")
    
    # Information section
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool calculates **monthly** snow depth using:
        
        **Data Sources:**
        - **MODIS Snow Cover** (MOD10A1): Normalized Difference Snow Index
        - **GLDAS Snow Water Equivalent**: NASA Global Land Data Assimilation System
        
        **Algorithm:**
        1. For each month in the selected date range:
           - Calculate Fractional Snow Cover (FSC) from MODIS NDSI
           - Estimate snow density based on FSC
           - Combine GLDAS SWE with FSC to get fine-resolution SWE
           - Calculate snow depth = SWE / snow density
        2. Package all monthly GeoTIFF files into a single ZIP archive
        
        **Output:**
        - One GeoTIFF file per month in the date range
        - Files are named: `snow_depth_YYYY_MM.tif`
        - All files packaged in a ZIP archive for download
        
        **Notes:**
        - Large areas are automatically chunked to avoid memory limits
        - Processing time depends on area size, number of months, and resolution
        - Authentication with Google Earth Engine is required
        - Each monthly calculation uses the mean values for that specific month
        """)

if __name__ == "__main__":
    main()