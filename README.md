# Snow Depth Calculator - Setup and Usage Guide

## Prerequisites

1. **Google Earth Engine Account**: Sign up at [https://earthengine.google.com/](https://earthengine.google.com/) and wait for approval
2. **Google Cloud Project**: Create a project with Earth Engine API enabled
3. **Python Environment**: Python 3.8 or higher

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run snow_depth_app.py
   ```

3. **Authenticate through the web interface** (see Authentication section below)

## Authentication

The application uses a web-based authentication system. No command-line setup required!

### Getting Your Credentials

**Option A: Service Account Key (Recommended for applications)**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select your project
3. Enable the Earth Engine API
4. Go to "IAM & Admin" → "Service Accounts"
5. Create a service account with "Earth Engine Resource Viewer" role
6. Download the JSON key file
7. Upload this file in the app

**Option B: User Credentials (For personal use)**
1. Install Earth Engine CLI locally: `pip install earthengine-api`
2. Run: `earthengine authenticate`
3. Complete the browser authentication
4. Find the credentials file at:
   - **Linux/Mac**: `~/.config/earthengine/credentials`
   - **Windows**: `C:\Users\[username]\.config\earthengine\credentials`
5. Upload this file in the app

### In the Application
1. Enter your Google Cloud Project ID
2. Upload your credentials file (JSON format)
3. Click "Authenticate"
4. Start using the application!

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run snow_depth_app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## How to Use

### 1. Set Parameters
- **Start Date**: Beginning of the analysis period
- **End Date**: End of the analysis period  
- **Resolution**: Pixel size in meters (default: 500m)
  - Lower values = higher resolution but longer processing time
  - Recommended: 500m for regional studies, 100-250m for local studies

### 2. Define Study Area

**Option A: Draw on Map**
- Use the drawing tools on the map
- Draw a polygon or rectangle around your area of interest
- The polygon will be automatically detected

**Option B: Upload JSON File**
- Prepare a GeoJSON file with your study area
- Use QGIS, ArcGIS, or online tools like [geojson.io](http://geojson.io/)
- Upload the file using the file uploader

### 3. Process Data
- Click "Calculate Snow Depth" button
- The app will:
  - Automatically chunk large areas to avoid memory limits
  - Download data from Google Earth Engine
  - Merge chunks if necessary
  - Provide a downloadable GeoTIFF file

## Features

### Automatic Chunking
- Large areas are automatically divided into smaller chunks
- Each chunk is processed separately
- Results are merged into a single GeoTIFF file
- Prevents memory and timeout issues with Google Earth Engine

### Data Sources
- **MODIS Snow Cover (MOD10A1)**: Daily snow cover at 500m resolution
- **GLDAS SWE**: 3-hourly snow water equivalent at ~25km resolution

### Output
- **GeoTIFF file** with snow depth in meters
- **EPSG:4326** coordinate system (WGS84)
- **LZW compression** for smaller file sizes

## Troubleshooting

### Common Issues

1. **Earth Engine Authentication Error**
   ```bash
   earthengine authenticate
   ```

2. **Memory/Timeout Errors**
   - Reduce the area size
   - Increase the resolution scale (e.g., from 250m to 500m)
   - The app automatically chunks large areas, but very large regions may still timeout

3. **No Data in Results**
   - Check if your area has snow during the selected time period
   - Verify the date range is correct
   - Some areas may have persistent cloud cover

4. **Slow Processing**
   - Large areas take more time
   - High resolution (low scale values) increases processing time
   - Consider processing smaller areas or lower resolution for faster results

### Area Size Limits
- **Small areas** (< 10,000 km²): Process normally
- **Medium areas** (10,000 - 100,000 km²): May be chunked automatically
- **Large areas** (> 100,000 km²): Will be chunked, expect longer processing times

## Tips for Best Results

1. **Choose appropriate dates**: Snow season varies by location
   - Northern Hemisphere: December - March
   - Southern Hemisphere: June - September
   - High elevations: Longer snow seasons

2. **Resolution selection**:
   - 500m: Good for regional studies, faster processing
   - 250m: Good balance of detail and speed
   - 100m: High detail, slower processing

3. **Area size**: Start with smaller areas to test, then scale up

4. **Data validation**: Compare results with ground truth or other snow products

## Example Workflows

### Regional Snow Assessment
1. Draw a large polygon over your region
2. Use 500m resolution
3. Select full winter season (Dec-Mar)
4. Let the app chunk automatically

### Local Detailed Study
1. Upload precise GeoJSON boundary
2. Use 100-250m resolution
3. Select specific snow events or shorter periods
4. Manual quality control of results

### Time Series Analysis
1. Run multiple analyses with different date ranges
2. Compare results across seasons/years
3. Use consistent resolution and boundaries

## Output Interpretation

- **Values**: Snow depth in meters
- **0 values**: No snow detected
- **NoData/NaN**: Areas with no valid observations (clouds, etc.)
- **Range**: Typically 0-5 meters for most regions

## Support

For issues with:
- **Google Earth Engine**: Check [EE documentation](https://developers.google.com/earth-engine/)
- **Streamlit**: Check [Streamlit documentation](https://docs.streamlit.io/)
- **This application**: Review error messages and troubleshooting section above