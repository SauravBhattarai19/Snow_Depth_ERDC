#!/usr/bin/env python3
"""
Comprehensive Snow Depth Comparison Tool
Compares satellite snow depth data with ground truth measurements

Author: Generated for snow depth analysis
Dependencies: rasterio, numpy, matplotlib, pandas, scikit-learn, scipy
"""

import os
import glob
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    print("Warning: rasterio not available. Install with: pip install rasterio")
    RASTERIO_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.spatial.distance import cdist


class SnowDepthComparator:
    """
    Comprehensive tool for comparing satellite snow depth data with ground truth
    """
    
    def __init__(self, satellite_dir, ground_truth_dir, output_dir="comparison_results", 
                 resampling_method="auto"):
        """
        Initialize the comparator
        
        Args:
            satellite_dir (str): Directory containing satellite .tif files
            ground_truth_dir (str): Directory containing ground truth data
            output_dir (str): Directory to save results
            resampling_method (str): Resampling method to use ('auto', 'bilinear', 'cubic', 'nearest', 'lanczos')
        """
        self.satellite_dir = Path(satellite_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Resampling configuration
        self.resampling_method = resampling_method
        
        # Storage for data
        self.satellite_files = {}
        self.ground_truth_files = {}
        self.comparison_results = []
        
        # Find and organize files
        self._find_files()
        
    def _get_optimal_resampling_method(self, data_characteristics=None):
        """
        Determine the optimal resampling method based on data characteristics
        
        Args:
            data_characteristics (dict): Optional dict with data info
            
        Returns:
            Resampling enum: Best resampling method for the data
        """
        if not RASTERIO_AVAILABLE:
            return 3  # Cubic for scipy fallback
            
        from rasterio.enums import Resampling
        
        if self.resampling_method == "auto":
            # For snow depth data (continuous values), bilinear or cubic work best
            # Bilinear is faster and usually sufficient for most cases
            # Cubic provides smoother results but is slower
            # Lanczos provides best quality but is slowest
            
            if data_characteristics:
                data_range = data_characteristics.get('range', 0)
                data_variance = data_characteristics.get('variance', 0)
                
                # For high-variance data with large ranges, use higher quality
                if data_range > 2.0 and data_variance > 0.5:
                    return Resampling.lanczos  # Best quality for complex data
                elif data_range > 1.0:
                    return Resampling.cubic    # Good balance for moderate complexity
                else:
                    return Resampling.bilinear # Fast and good for simple data
            else:
                # Default for snow depth: bilinear (good balance of speed/quality)
                return Resampling.bilinear
                
        elif self.resampling_method == "nearest":
            return Resampling.nearest
        elif self.resampling_method == "bilinear":
            return Resampling.bilinear
        elif self.resampling_method == "cubic":
            return Resampling.cubic
        elif self.resampling_method == "lanczos":
            return Resampling.lanczos
        else:
            print(f"Unknown resampling method '{self.resampling_method}', using bilinear")
            return Resampling.bilinear
        
    def _find_files(self):
        """Find and organize satellite and ground truth files by date"""
        
        # Find satellite files
        satellite_pattern = self.satellite_dir / "snow_depth_*.tif"
        for file_path in glob.glob(str(satellite_pattern)):
            filename = os.path.basename(file_path)
            # Extract date from filename: snow_depth_YYYY_MM.tif
            match = re.search(r'snow_depth_(\d{4})_(\d{2})\.tif', filename)
            if match:
                year, month = match.groups()
                date_key = f"{year}_{month}"
                self.satellite_files[date_key] = file_path
        
        # Find ground truth files
        for year_dir in self.ground_truth_dir.glob("*"):
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = year_dir.name
                # Look for both .asc and .tif files
                for ext in ['*.asc', '*.tif']:
                    pattern = year_dir / f"snod_{year}_*.{ext.split('.')[-1]}"
                    for file_path in glob.glob(str(pattern)):
                        filename = os.path.basename(file_path)
                        # Extract date from filename: snod_YYYY_MM.asc or snod_YYYY_MM.tif
                        match = re.search(r'snod_(\d{4})_(\d{2})\.(asc|tif)', filename)
                        if match:
                            year_m, month, file_ext = match.groups()
                            date_key = f"{year_m}_{month}"
                            if date_key not in self.ground_truth_files:
                                self.ground_truth_files[date_key] = {}
                            self.ground_truth_files[date_key][file_ext] = file_path
        
        print(f"Found {len(self.satellite_files)} satellite files")
        print(f"Found {len(self.ground_truth_files)} ground truth date groups")
        
        # Find matching dates
        self.matching_dates = set(self.satellite_files.keys()) & set(self.ground_truth_files.keys())
        print(f"Found {len(self.matching_dates)} matching dates")
        
    def _read_raster(self, file_path, preferred_format='tif'):
        """
        Read raster data from file
        
        Args:
            file_path: Path to raster file or dict of file paths by extension
            preferred_format: Preferred file format to read
            
        Returns:
            tuple: (data_array, transform, crs)
        """
        if not RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required for reading geospatial data")
            
        # Handle case where file_path is a dict (ground truth with multiple formats)
        if isinstance(file_path, dict):
            if preferred_format in file_path:
                actual_path = file_path[preferred_format]
            else:
                # Use any available format
                actual_path = next(iter(file_path.values()))
        else:
            actual_path = file_path
            
        try:
            with rasterio.open(actual_path) as src:
                data = src.read(1)  # Read first band
                transform = src.transform
                crs = src.crs
                nodata = src.nodata
                
                # Handle nodata values with robust checking
                if nodata is not None:
                    # Handle different nodata value types
                    if isinstance(nodata, (list, tuple)):
                        # Multiple nodata values
                        for nd_val in nodata:
                            if nd_val is not None:
                                data = np.where(data == nd_val, np.nan, data)
                    else:
                        # Single nodata value
                        data = np.where(data == nodata, np.nan, data)
                
                # Ensure CRS is properly handled
                if crs is None:
                    print(f"Warning: No CRS found in {os.path.basename(actual_path)}")
                    crs = 'EPSG:4326'  # Default to WGS84 if no CRS
                
        except Exception as e:
            print(f"Error reading raster {actual_path}: {str(e)}")
            raise
            
        return data, transform, crs
    
    def _align_rasters(self, sat_data, sat_transform, sat_crs, 
                      gt_data, gt_transform, gt_crs):
        """
        Align two rasters to the same grid for comparison using proper geospatial resampling
        
        Returns:
            tuple: (aligned_sat_data, aligned_gt_data)
        """
        if not RASTERIO_AVAILABLE:
            # Fallback: simple resizing if rasterio not available
            from scipy.ndimage import zoom
            print("Warning: rasterio not available, using basic scipy resampling")
            
            sat_shape = sat_data.shape
            gt_shape = gt_data.shape
            
            if sat_shape != gt_shape:
                # Resize ground truth to match satellite
                zoom_factors = (sat_shape[0] / gt_shape[0], sat_shape[1] / gt_shape[1])
                gt_data_aligned = zoom(gt_data, zoom_factors, order=1, mode='nearest')
                print(f"Resized ground truth from {gt_shape} to {gt_data_aligned.shape} using scipy")
            else:
                gt_data_aligned = gt_data.copy()
                
            return sat_data, gt_data_aligned
        
        # Proper geospatial resampling using rasterio
        try:
            from rasterio.warp import calculate_default_transform, reproject
            from rasterio.enums import Resampling
            import rasterio
            
            # Handle CRS safely
            if sat_crs is None:
                print("Warning: Satellite CRS is None, assuming EPSG:4326")
                sat_crs = rasterio.crs.CRS.from_epsg(4326)
            if gt_crs is None:
                print("Warning: Ground truth CRS is None, assuming EPSG:4326")
                gt_crs = rasterio.crs.CRS.from_epsg(4326)
            
            # Convert string CRS to rasterio CRS objects if needed
            if isinstance(sat_crs, str):
                sat_crs = rasterio.crs.CRS.from_string(sat_crs)
            if isinstance(gt_crs, str):
                gt_crs = rasterio.crs.CRS.from_string(gt_crs)
            
            print(f"Satellite CRS: {sat_crs}")
            print(f"Ground truth CRS: {gt_crs}")
            print(f"Satellite shape: {sat_data.shape}, Ground truth shape: {gt_data.shape}")
            
            # Check if rasters already have same CRS and dimensions
            same_crs = sat_crs == gt_crs
            same_transform = np.allclose(np.array(sat_transform)[:6], np.array(gt_transform)[:6], rtol=1e-6)
            same_shape = sat_data.shape == gt_data.shape
            
            if same_crs and same_transform and same_shape:
                print("Rasters already aligned - no resampling needed")
                return sat_data, gt_data
            
            # Calculate resampling ratio to assess quality impact
            pixel_ratio = (sat_data.size) / (gt_data.size)
            print(f"Pixel ratio (target/source): {pixel_ratio:.4f}")
            
            # Determine optimal target grid based on resolution analysis
            if pixel_ratio < 0.1:  # Extreme downsampling (>90% data loss)
                print("Warning: Extreme downsampling detected (>90% data loss)")
                print("Consider using higher resolution satellite data or alternative approach")
                
                # For extreme downsampling, use the highest quality method available
                target_crs = sat_crs
                target_transform = sat_transform
                target_shape = sat_data.shape
                print(f"Proceeding with satellite grid as target: {target_shape}")
                print("Using highest quality resampling to minimize information loss")
                
            elif pixel_ratio > 10:  # Extreme upsampling
                print("Warning: Extreme upsampling detected - may introduce artifacts")
                target_crs = sat_crs
                target_transform = sat_transform
                target_shape = sat_data.shape
                
            else:  # Reasonable resampling ratio
                target_crs = sat_crs
                target_transform = sat_transform
                target_shape = sat_data.shape
                
            print(f"Resampling ground truth to match satellite grid...")
            print(f"Target shape: {target_shape}")
            
            # Analyze data characteristics for optimal resampling method selection
            gt_valid = gt_data[~np.isnan(gt_data)]
            if len(gt_valid) > 0:
                data_characteristics = {
                    'range': float(np.max(gt_valid) - np.min(gt_valid)),
                    'variance': float(np.var(gt_valid)),
                    'mean': float(np.mean(gt_valid)),
                    'pixel_ratio': pixel_ratio
                }
            else:
                data_characteristics = {'pixel_ratio': pixel_ratio}
            
            # Choose the optimal resampling method (considering extreme ratios)
            if pixel_ratio < 0.1:  # Extreme downsampling - use best quality
                from rasterio.enums import Resampling
                resampling_method = Resampling.lanczos  # Highest quality for extreme cases
                print("Using Lanczos resampling (highest quality) for extreme downsampling")
            elif pixel_ratio > 10:  # Extreme upsampling - use cubic to avoid blocky artifacts
                from rasterio.enums import Resampling
                resampling_method = Resampling.cubic
                print("Using Cubic resampling for extreme upsampling to avoid artifacts")
            else:
                resampling_method = self._get_optimal_resampling_method(data_characteristics)
            
            # Create output array for resampled ground truth
            gt_resampled = np.empty(target_shape, dtype=gt_data.dtype)
            
            print(f"Using resampling method: {resampling_method.name}")
            if data_characteristics:
                print(f"Data characteristics - Range: {data_characteristics['range']:.3f}, "
                      f"Variance: {data_characteristics['variance']:.3f}")
            
            # Perform the reproject operation
            reproject(
                source=gt_data,
                destination=gt_resampled,
                src_transform=gt_transform,
                src_crs=gt_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_method,
                src_nodata=np.nan,
                dst_nodata=np.nan
            )
            
            print(f"Resampling completed using {resampling_method.name} method")
            print(f"Original GT shape: {gt_data.shape} -> Resampled GT shape: {gt_resampled.shape}")
            
            # Validate the resampling results
            original_valid_pixels = np.sum(~np.isnan(gt_data))
            resampled_valid_pixels = np.sum(~np.isnan(gt_resampled))
            
            print(f"Valid pixels - Original GT: {original_valid_pixels:,}, Resampled GT: {resampled_valid_pixels:,}")
            
            # Additional quality checks
            if original_valid_pixels > 0 and resampled_valid_pixels > 0:
                original_stats = {
                    'mean': float(np.nanmean(gt_data)),
                    'std': float(np.nanstd(gt_data)),
                    'min': float(np.nanmin(gt_data)),
                    'max': float(np.nanmax(gt_data))
                }
                resampled_stats = {
                    'mean': float(np.nanmean(gt_resampled)),
                    'std': float(np.nanstd(gt_resampled)),
                    'min': float(np.nanmin(gt_resampled)),
                    'max': float(np.nanmax(gt_resampled))
                }
                
                print(f"Original GT stats - Mean: {original_stats['mean']:.3f}, Std: {original_stats['std']:.3f}")
                print(f"Resampled GT stats - Mean: {resampled_stats['mean']:.3f}, Std: {resampled_stats['std']:.3f}")
                
                # Check for statistical preservation
                mean_diff = abs(original_stats['mean'] - resampled_stats['mean'])
                std_diff = abs(original_stats['std'] - resampled_stats['std'])
                
                # More lenient thresholds for extreme resampling
                mean_threshold = original_stats['std'] * (0.2 if pixel_ratio < 0.1 else 0.1)
                std_threshold = original_stats['std'] * (0.4 if pixel_ratio < 0.1 else 0.2)
                
                if mean_diff > mean_threshold:
                    severity = "Expected for extreme resampling" if pixel_ratio < 0.1 else "Warning"
                    print(f"{severity}: Mean value changed significantly during resampling (diff: {mean_diff:.3f})")
                if std_diff > std_threshold:
                    severity = "Expected for extreme resampling" if pixel_ratio < 0.1 else "Warning"
                    print(f"{severity}: Standard deviation changed significantly during resampling (diff: {std_diff:.3f})")
                    
                # Calculate quality preservation score
                mean_preservation = max(0, 1.0 - mean_diff / (original_stats['std'] + 1e-10))
                std_preservation = min(original_stats['std'], resampled_stats['std']) / (max(original_stats['std'], resampled_stats['std']) + 1e-10)
                overall_quality = (mean_preservation + std_preservation) / 2
                
                print(f"Resampling quality score: {overall_quality:.3f} (0=poor, 1=perfect)")
                
                if overall_quality < 0.7 and pixel_ratio >= 0.1:
                    print("Warning: Poor resampling quality detected. Consider:")
                    print("  - Using higher resolution satellite data")
                    print("  - Alternative resampling method")
                    print("  - Pre-processing to match resolutions better")
            
            if resampled_valid_pixels == 0:
                print("Warning: Resampling resulted in all NaN values - checking for issues...")
                # Fall back to alternative method if needed
                return self._fallback_alignment(sat_data, gt_data, sat_transform, gt_transform)
            
            # Check for extreme resampling ratios that might cause quality issues
            final_pixel_ratio = resampled_valid_pixels / original_valid_pixels if original_valid_pixels > 0 else 0
            
            if final_pixel_ratio > 10:
                print(f"Warning: Extreme upsampling ratio ({final_pixel_ratio:.2f}x) - may introduce interpolation artifacts")
                print("Recommendation: Use higher resolution ground truth data")
            elif final_pixel_ratio < 0.1:
                resolution_loss = (1 - final_pixel_ratio) * 100
                print(f"Info: Significant downsampling ({final_pixel_ratio:.3f}x, {resolution_loss:.1f}% resolution loss)")
                print("This is expected when comparing high-res ground truth to low-res satellite data")
                print("Results interpretation:")
                print("  - Comparison shows how well satellite captures large-scale patterns")
                print("  - Fine-scale details in ground truth are averaged out")
                print("  - Negative R² may indicate scale mismatch rather than poor correlation")
            elif final_pixel_ratio < 0.5:
                print(f"Info: Moderate downsampling ({final_pixel_ratio:.2f}x) - some information loss expected")
            
            return sat_data, gt_resampled
                
        except Exception as e:
            print(f"Error in geospatial resampling: {str(e)}")
            print("Falling back to basic alignment method...")
            return self._fallback_alignment(sat_data, gt_data, sat_transform, gt_transform)
    
    def _fallback_alignment(self, sat_data, gt_data, sat_transform, gt_transform):
        """
        Fallback alignment method when proper resampling fails
        """
        from scipy.ndimage import zoom
        
        sat_shape = sat_data.shape
        gt_shape = gt_data.shape
        
        if sat_shape == gt_shape:
            print("Shapes already match - no resampling needed")
            return sat_data, gt_data
        
        # Calculate zoom factors
        zoom_factors = (sat_shape[0] / gt_shape[0], sat_shape[1] / gt_shape[1])
        
        # Use cubic interpolation for better quality (order=3)
        # For snow depth data, cubic gives smoother results than linear
        try:
            gt_data_aligned = zoom(gt_data, zoom_factors, order=3, mode='constant', cval=np.nan)
            interpolation_method = "cubic"
        except:
            # Fall back to bilinear if cubic fails
            gt_data_aligned = zoom(gt_data, zoom_factors, order=1, mode='constant', cval=np.nan)
            interpolation_method = "bilinear"
        
        print(f"Fallback: Resized ground truth from {gt_shape} to {gt_data_aligned.shape} using {interpolation_method} interpolation")
        
        return sat_data, gt_data_aligned
    
    def _validate_resampling_quality(self, original_data, resampled_data, method_name):
        """
        Validate the quality of resampling operation
        
        Args:
            original_data: Original data array
            resampled_data: Resampled data array
            method_name: Name of the resampling method used
            
        Returns:
            dict: Quality metrics
        """
        quality_metrics = {
            'method': method_name,
            'original_valid_pixels': int(np.sum(~np.isnan(original_data))),
            'resampled_valid_pixels': int(np.sum(~np.isnan(resampled_data))),
            'pixel_ratio': 0.0,
            'mean_preservation': 0.0,
            'std_preservation': 0.0,
            'range_preservation': 0.0,
            'quality_score': 0.0
        }
        
        orig_valid = original_data[~np.isnan(original_data)]
        resamp_valid = resampled_data[~np.isnan(resampled_data)]
        
        if len(orig_valid) > 0 and len(resamp_valid) > 0:
            quality_metrics['pixel_ratio'] = len(resamp_valid) / len(orig_valid)
            
            orig_mean = np.mean(orig_valid)
            resamp_mean = np.mean(resamp_valid)
            orig_std = np.std(orig_valid)
            resamp_std = np.std(resamp_valid)
            orig_range = np.max(orig_valid) - np.min(orig_valid)
            resamp_range = np.max(resamp_valid) - np.min(resamp_valid)
            
            # Calculate preservation scores (closer to 1.0 is better)
            quality_metrics['mean_preservation'] = 1.0 - abs(orig_mean - resamp_mean) / (orig_std + 1e-10)
            quality_metrics['std_preservation'] = min(orig_std, resamp_std) / (max(orig_std, resamp_std) + 1e-10)
            quality_metrics['range_preservation'] = min(orig_range, resamp_range) / (max(orig_range, resamp_range) + 1e-10)
            
            # Overall quality score (weighted average)
            quality_metrics['quality_score'] = (
                0.4 * quality_metrics['mean_preservation'] +
                0.3 * quality_metrics['std_preservation'] +
                0.3 * quality_metrics['range_preservation']
            )
        
        return quality_metrics
    
    def diagnose_data_issues(self, sat_data, gt_data, date_key):
        """
        Diagnose common data issues and provide helpful messages
        
        Args:
            sat_data: Satellite data array
            gt_data: Ground truth data array
            date_key: Date identifier for context
        """
        print(f"\nDiagnosing data issues for {date_key}:")
        
        # Basic array info
        print(f"  Satellite data shape: {sat_data.shape}")
        print(f"  Ground truth data shape: {gt_data.shape}")
        
        # Value ranges
        sat_valid = sat_data[~np.isnan(sat_data)]
        gt_valid = gt_data[~np.isnan(gt_data)]
        
        if len(sat_valid) > 0:
            print(f"  Satellite range: {np.min(sat_valid):.3f} to {np.max(sat_valid):.3f} m")
            print(f"  Satellite mean: {np.mean(sat_valid):.3f} m")
        else:
            print(f"  Satellite: ALL VALUES ARE NaN")
            
        if len(gt_valid) > 0:
            print(f"  Ground truth range: {np.min(gt_valid):.3f} to {np.max(gt_valid):.3f} m")
            print(f"  Ground truth mean: {np.mean(gt_valid):.3f} m")
        else:
            print(f"  Ground truth: ALL VALUES ARE NaN")
        
        # Coverage analysis
        total_pixels = sat_data.size
        sat_nan_count = np.sum(np.isnan(sat_data))
        gt_nan_count = np.sum(np.isnan(gt_data))
        both_valid_count = np.sum(~np.isnan(sat_data) & ~np.isnan(gt_data))
        
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Satellite NaN: {sat_nan_count:,} ({(sat_nan_count/total_pixels)*100:.1f}%)")
        print(f"  Ground truth NaN: {gt_nan_count:,} ({(gt_nan_count/total_pixels)*100:.1f}%)")
        print(f"  Both valid: {both_valid_count:,} ({(both_valid_count/total_pixels)*100:.1f}%)")
        
        # Potential issues
        issues = []
        if sat_nan_count == total_pixels:
            issues.append("All satellite data is NaN")
        elif sat_nan_count > total_pixels * 0.9:
            issues.append("Very high satellite data loss (>90%)")
            
        if gt_nan_count == total_pixels:
            issues.append("All ground truth data is NaN")
        elif gt_nan_count > total_pixels * 0.9:
            issues.append("Very high ground truth data loss (>90%)")
            
        if both_valid_count == 0:
            issues.append("No overlapping valid data")
        elif both_valid_count < total_pixels * 0.05:
            issues.append("Very low overlap (<5%)")
            
        # Check for constant values
        if len(sat_valid) > 1 and np.var(sat_valid) == 0:
            issues.append(f"Satellite data is constant (all values = {sat_valid[0]:.3f})")
        if len(gt_valid) > 1 and np.var(gt_valid) == 0:
            issues.append(f"Ground truth data is constant (all values = {gt_valid[0]:.3f})")
            
        if issues:
            print(f"  Identified issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  No obvious issues detected")
            
        return issues
    
    def calculate_metrics(self, sat_data, gt_data):
        """
        Calculate comparison metrics between satellite and ground truth data
        
        Args:
            sat_data: Satellite data array
            gt_data: Ground truth data array
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        # Comprehensive NaN/data quality analysis
        total_pixels = sat_data.size
        sat_nan_mask = np.isnan(sat_data)
        gt_nan_mask = np.isnan(gt_data)
        
        # Count different types of pixels
        sat_nan_count = np.sum(sat_nan_mask)
        gt_nan_count = np.sum(gt_nan_mask)
        both_nan_count = np.sum(sat_nan_mask & gt_nan_mask)
        either_nan_count = np.sum(sat_nan_mask | gt_nan_mask)
        
        # Valid data mask - both datasets have valid values
        valid_mask = ~(sat_nan_mask | gt_nan_mask)
        sat_valid = sat_data[valid_mask]
        gt_valid = gt_data[valid_mask]
        
        if len(sat_valid) == 0:
            return {
                "error": "No valid overlapping data",
                "total_pixels": total_pixels,
                "sat_nan_pixels": sat_nan_count,
                "gt_nan_pixels": gt_nan_count,
                "both_nan_pixels": both_nan_count,
                "valid_pixels": 0,
                "data_coverage_percent": 0.0
            }
        
        # Calculate coverage statistics
        valid_pixels = len(sat_valid)
        data_coverage = (valid_pixels / total_pixels) * 100
        
    def calculate_metrics(self, sat_data, gt_data):
        """
        Calculate comparison metrics between satellite and ground truth data
        
        Args:
            sat_data: Satellite data array
            gt_data: Ground truth data array
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        try:
            # Ensure data is numpy arrays
            if sat_data is None or gt_data is None:
                return {"error": "Input data is None"}
                
            sat_data = np.asarray(sat_data)
            gt_data = np.asarray(gt_data)
            
            # Comprehensive NaN/data quality analysis
            total_pixels = sat_data.size
            sat_nan_mask = np.isnan(sat_data)
            gt_nan_mask = np.isnan(gt_data)
            
            # Count different types of pixels
            sat_nan_count = np.sum(sat_nan_mask)
            gt_nan_count = np.sum(gt_nan_mask)
            both_nan_count = np.sum(sat_nan_mask & gt_nan_mask)
            either_nan_count = np.sum(sat_nan_mask | gt_nan_mask)
            
            # Valid data mask - both datasets have valid values
            valid_mask = ~(sat_nan_mask | gt_nan_mask)
            sat_valid = sat_data[valid_mask]
            gt_valid = gt_data[valid_mask]
            
            if len(sat_valid) == 0:
                return {
                    "error": "No valid overlapping data",
                    "total_pixels": total_pixels,
                    "sat_nan_pixels": sat_nan_count,
                    "gt_nan_pixels": gt_nan_count,
                    "both_nan_pixels": both_nan_count,
                    "valid_pixels": 0,
                    "data_coverage_percent": 0.0
                }
            
            # Calculate coverage statistics
            valid_pixels = len(sat_valid)
            data_coverage = (valid_pixels / total_pixels) * 100
            
            # Initialize metrics dictionary with safe defaults
            metrics = {
                # Core validation metrics
                'n_points': valid_pixels,
                'data_coverage_percent': data_coverage,
                'sat_data_availability_percent': ((total_pixels - sat_nan_count) / total_pixels) * 100,
                'gt_data_availability_percent': ((total_pixels - gt_nan_count) / total_pixels) * 100,
                'total_pixels': total_pixels,
                'valid_pixels': valid_pixels,
                'sat_nan_pixels': sat_nan_count,
                'gt_nan_pixels': gt_nan_count,
                'both_nan_pixels': both_nan_count,
                'either_nan_pixels': either_nan_count,
                'data_quality_flag': 'processing'
            }
            
            # Robust calculation of statistical metrics
            try:
                # Basic statistics
                metrics.update({
                    'sat_mean': float(np.mean(sat_valid)),
                    'sat_std': float(np.std(sat_valid)),
                    'gt_mean': float(np.mean(gt_valid)),
                    'gt_std': float(np.std(gt_valid)),
                    'sat_min': float(np.min(sat_valid)),
                    'sat_max': float(np.max(sat_valid)),
                    'gt_min': float(np.min(gt_valid)),
                    'gt_max': float(np.max(gt_valid)),
                    'bias': float(np.mean(sat_valid - gt_valid)),
                })
                
                # Check for constant values (zero variance)
                sat_var = np.var(sat_valid)
                gt_var = np.var(gt_valid)
                
                if sat_var == 0 and gt_var == 0:
                    # Both datasets are constant
                    metrics.update({
                        'rmse': 0.0,
                        'mae': 0.0,
                        'r2': 1.0 if metrics['bias'] == 0 else 0.0,
                        'correlation': 1.0 if metrics['bias'] == 0 else np.nan,
                        'data_quality_flag': 'both_constant'
                    })
                elif sat_var == 0 or gt_var == 0:
                    # One dataset is constant
                    try:
                        rmse_val = np.sqrt(mean_squared_error(gt_valid, sat_valid))
                        mae_val = mean_absolute_error(gt_valid, sat_valid)
                    except:
                        rmse_val = float(np.sqrt(np.mean((sat_valid - gt_valid) ** 2)))
                        mae_val = float(np.mean(np.abs(sat_valid - gt_valid)))
                    
                    metrics.update({
                        'rmse': rmse_val,
                        'mae': mae_val,
                        'r2': 0.0,
                        'correlation': np.nan,
                        'data_quality_flag': 'one_constant'
                    })
                else:
                    # Normal calculation
                    try:
                        rmse_val = np.sqrt(mean_squared_error(gt_valid, sat_valid))
                        mae_val = mean_absolute_error(gt_valid, sat_valid)
                    except:
                        rmse_val = float(np.sqrt(np.mean((sat_valid - gt_valid) ** 2)))
                        mae_val = float(np.mean(np.abs(sat_valid - gt_valid)))
                    
                    metrics.update({
                        'rmse': rmse_val,
                        'mae': mae_val,
                        'data_quality_flag': 'normal'
                    })
                    
                    # R² calculation with error handling
                    try:
                        r2_val = r2_score(gt_valid, sat_valid)
                        if np.isfinite(r2_val):
                            metrics['r2'] = float(r2_val)
                        else:
                            metrics['r2'] = np.nan
                            metrics['data_quality_flag'] = 'r2_invalid'
                    except:
                        # Manual R² calculation
                        try:
                            ss_res = np.sum((gt_valid - sat_valid) ** 2)
                            ss_tot = np.sum((gt_valid - np.mean(gt_valid)) ** 2)
                            if ss_tot != 0:
                                metrics['r2'] = float(1 - (ss_res / ss_tot))
                            else:
                                metrics['r2'] = np.nan
                        except:
                            metrics['r2'] = np.nan
                            metrics['data_quality_flag'] = 'r2_calculation_failed'
                    
                    # Correlation calculation with error handling
                    try:
                        if len(sat_valid) > 1 and len(gt_valid) > 1:
                            corr_matrix = np.corrcoef(sat_valid, gt_valid)
                            if (corr_matrix is not None and 
                                hasattr(corr_matrix, 'shape') and 
                                corr_matrix.shape == (2, 2)):
                                corr_val = corr_matrix[0, 1]
                                if np.isfinite(corr_val):
                                    metrics['correlation'] = float(corr_val)
                                else:
                                    metrics['correlation'] = np.nan
                                    metrics['data_quality_flag'] = 'correlation_nan'
                            else:
                                metrics['correlation'] = np.nan
                                metrics['data_quality_flag'] = 'correlation_matrix_invalid'
                        else:
                            metrics['correlation'] = np.nan
                            metrics['data_quality_flag'] = 'insufficient_data_for_correlation'
                    except Exception as e:
                        metrics['correlation'] = np.nan
                        metrics['data_quality_flag'] = f'correlation_error'
                        
            except Exception as e:
                # Fallback for any calculation errors
                print(f"Warning: Error in metric calculation: {str(e)}")
                metrics.update({
                    'rmse': np.nan,
                    'mae': np.nan,
                    'bias': np.nan,
                    'r2': np.nan,
                    'correlation': np.nan,
                    'sat_mean': np.nan,
                    'sat_std': np.nan,
                    'gt_mean': np.nan,
                    'gt_std': np.nan,
                    'sat_min': np.nan,
                    'sat_max': np.nan,
                    'gt_min': np.nan,
                    'gt_max': np.nan,
                    'data_quality_flag': f'calculation_error'
                })
                
        except Exception as e:
            # Complete fallback
            return {
                "error": f"Critical error in metrics calculation: {str(e)}",
                "data_quality_flag": "critical_error"
            }
            
        return metrics
        
        # Additional statistical tests
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(sat_valid, gt_valid)
            metrics['ks_statistic'] = ks_stat
            metrics['ks_pvalue'] = ks_pvalue
            
            # Wilcoxon signed-rank test (paired)
            if len(sat_valid) > 1:
                wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(sat_valid, gt_valid)
                metrics['wilcoxon_statistic'] = wilcoxon_stat
                metrics['wilcoxon_pvalue'] = wilcoxon_pvalue
        except:
            pass
            
    def create_data_quality_analysis(self, sat_data, gt_data, date_key):
        """
        Create detailed data quality and coverage analysis
        
        Args:
            sat_data: Satellite data array
            gt_data: Ground truth data array  
            date_key: Date identifier
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Data Quality Analysis - {date_key}', fontsize=16)
        
        # Create masks for different data conditions
        sat_valid = ~np.isnan(sat_data)
        gt_valid = ~np.isnan(gt_data)
        both_valid = sat_valid & gt_valid
        
        # 1. Data availability map
        data_status = np.zeros_like(sat_data, dtype=int)
        data_status[both_valid] = 3  # Both valid (green)
        data_status[sat_valid & ~gt_valid] = 2  # Only satellite valid (blue)
        data_status[~sat_valid & gt_valid] = 1  # Only ground truth valid (orange)
        # data_status remains 0 where both are NaN (red)
        
        colors = ['red', 'orange', 'blue', 'green']
        labels = ['Both NaN', 'Only GT valid', 'Only Sat valid', 'Both valid']
        
        im1 = axes[0, 0].imshow(data_status, cmap='viridis', vmin=0, vmax=3)
        axes[0, 0].set_title('Data Availability Map')
        
        # Create custom colorbar
        from matplotlib.patches import Rectangle
        legend_elements = [Rectangle((0,0),1,1, facecolor=colors[i], label=labels[i]) 
                          for i in range(4)]
        axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 2. Satellite data with NaN overlay
        sat_display = sat_data.copy()
        axes[0, 1].imshow(sat_display, cmap='Blues', vmin=0)
        # Overlay NaN areas in red
        nan_overlay = np.ma.masked_where(sat_valid, np.ones_like(sat_data))
        axes[0, 1].imshow(nan_overlay, cmap='Reds', alpha=0.7)
        axes[0, 1].set_title('Satellite Data (NaN in red)')
        
        # 3. Ground truth data with NaN overlay  
        gt_display = gt_data.copy()
        axes[1, 0].imshow(gt_display, cmap='Blues', vmin=0)
        # Overlay NaN areas in red
        nan_overlay_gt = np.ma.masked_where(gt_valid, np.ones_like(gt_data))
        axes[1, 0].imshow(nan_overlay_gt, cmap='Reds', alpha=0.7)
        axes[1, 0].set_title('Ground Truth (NaN in red)')
        
        # 4. Coverage statistics
        axes[1, 1].axis('off')
        
        total_pixels = sat_data.size
        sat_valid_count = np.sum(sat_valid)
        gt_valid_count = np.sum(gt_valid)
        both_valid_count = np.sum(both_valid)
        
        coverage_text = f"""
        Data Coverage Statistics:
        
        Total pixels: {total_pixels:,}
        
        Satellite coverage: {sat_valid_count:,} ({(sat_valid_count/total_pixels)*100:.1f}%)
        Ground truth coverage: {gt_valid_count:,} ({(gt_valid_count/total_pixels)*100:.1f}%)
        Both valid: {both_valid_count:,} ({(both_valid_count/total_pixels)*100:.1f}%)
        
        Data loss causes:
        - Satellite NaN: {total_pixels - sat_valid_count:,} pixels
        - Ground truth NaN: {total_pixels - gt_valid_count:,} pixels
        
        Usable for comparison: {both_valid_count:,} pixels
        ({(both_valid_count/total_pixels)*100:.1f}% of total area)
        """
        
        axes[1, 1].text(0.1, 0.9, coverage_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        quality_plot_path = self.output_dir / f'data_quality_{date_key}.png'
        plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved data quality analysis: {quality_plot_path}")
    
    def analyze_cloud_impact(self, df):
        """
        Analyze the impact of clouds/data gaps on comparison results
        
        Args:
            df: DataFrame with comparison results
        """
        if df.empty or len(df) < 2:
            print("Insufficient data for cloud impact analysis")
            return
            
        # Create cloud impact analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Impact of Data Coverage on Comparison Quality', fontsize=16)
        
        # 1. RMSE vs Data Coverage
        if 'rmse' in df.columns and 'data_coverage_percent' in df.columns:
            valid_data = df.dropna(subset=['rmse', 'data_coverage_percent'])
            if len(valid_data) > 1:
                axes[0, 0].scatter(valid_data['data_coverage_percent'], valid_data['rmse'], alpha=0.7)
                axes[0, 0].set_xlabel('Data Coverage (%)')
                axes[0, 0].set_ylabel('RMSE (m)')
                axes[0, 0].set_title('RMSE vs Data Coverage')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add trend line
                if len(valid_data) > 2:
                    z = np.polyfit(valid_data['data_coverage_percent'], valid_data['rmse'], 1)
                    p = np.poly1d(z)
                    axes[0, 0].plot(valid_data['data_coverage_percent'], p(valid_data['data_coverage_percent']), "r--", alpha=0.8)
            else:
                axes[0, 0].text(0.5, 0.5, 'Insufficient valid RMSE data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # 2. R² vs Data Coverage
        if 'r2' in df.columns and 'data_coverage_percent' in df.columns:
            valid_data = df.dropna(subset=['r2', 'data_coverage_percent'])
            if len(valid_data) > 1:
                axes[0, 1].scatter(valid_data['data_coverage_percent'], valid_data['r2'], alpha=0.7, color='green')
                axes[0, 1].set_xlabel('Data Coverage (%)')
                axes[0, 1].set_ylabel('R²')
                axes[0, 1].set_title('R² vs Data Coverage')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Insufficient valid R² data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. Data availability over time
        if 'datetime' in df.columns:
            axes[1, 0].set_title('Data Availability Over Time')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Data Availability (%)')
            
            plotted = False
            if 'sat_data_availability_percent' in df.columns:
                valid_sat = df.dropna(subset=['sat_data_availability_percent'])
                if len(valid_sat) > 0:
                    axes[1, 0].plot(valid_sat['datetime'], valid_sat['sat_data_availability_percent'], 'b-o', label='Satellite', markersize=4)
                    plotted = True
                    
            if 'gt_data_availability_percent' in df.columns:
                valid_gt = df.dropna(subset=['gt_data_availability_percent'])
                if len(valid_gt) > 0:
                    axes[1, 0].plot(valid_gt['datetime'], valid_gt['gt_data_availability_percent'], 'r-o', label='Ground Truth', markersize=4)
                    plotted = True
                    
            if 'data_coverage_percent' in df.columns:
                valid_both = df.dropna(subset=['data_coverage_percent'])
                if len(valid_both) > 0:
                    axes[1, 0].plot(valid_both['datetime'], valid_both['data_coverage_percent'], 'g-o', label='Both Valid', markersize=4)
                    plotted = True
            
            if plotted:
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No availability data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        # Calculate statistics for available data
        coverage_stats = {}
        if 'data_coverage_percent' in df.columns:
            valid_coverage = df['data_coverage_percent'].dropna()
            if len(valid_coverage) > 0:
                coverage_stats.update({
                    'mean_coverage': valid_coverage.mean(),
                    'min_coverage': valid_coverage.min(),
                    'max_coverage': valid_coverage.max(),
                    'low_coverage_count': len(valid_coverage[valid_coverage < 50]),
                    'high_coverage_count': len(valid_coverage[valid_coverage > 90]),
                    'total_dates': len(valid_coverage)
                })
        
        # Correlation between coverage and RMSE
        coverage_rmse_corr = np.nan
        if 'data_coverage_percent' in df.columns and 'rmse' in df.columns:
            valid_both = df.dropna(subset=['data_coverage_percent', 'rmse'])
            if len(valid_both) > 2:
                coverage_rmse_corr = valid_both['data_coverage_percent'].corr(valid_both['rmse'])
        
        summary_text = f"""
        Data Coverage Impact Summary:
        
        """
        
        if coverage_stats:
            summary_text += f"""Average data coverage: {coverage_stats['mean_coverage']:.1f}%
        Minimum coverage: {coverage_stats['min_coverage']:.1f}%
        Maximum coverage: {coverage_stats['max_coverage']:.1f}%
        
        Low coverage dates (< 50%): {coverage_stats['low_coverage_count']} of {coverage_stats['total_dates']}
        High coverage dates (> 90%): {coverage_stats['high_coverage_count']} of {coverage_stats['total_dates']}
        """
        else:
            summary_text += "No coverage data available\n"
        
        if not np.isnan(coverage_rmse_corr):
            summary_text += f"\nCoverage vs RMSE correlation: {coverage_rmse_corr:.3f}"
        else:
            summary_text += "\nCoverage vs RMSE correlation: N/A"
        
        # Add availability stats
        if 'sat_data_availability_percent' in df.columns:
            sat_avg = df['sat_data_availability_percent'].mean()
            if not np.isnan(sat_avg):
                summary_text += f"\nAverage satellite availability: {sat_avg:.1f}%"
                
        if 'gt_data_availability_percent' in df.columns:
            gt_avg = df['gt_data_availability_percent'].mean()
            if not np.isnan(gt_avg):
                summary_text += f"\nAverage ground truth availability: {gt_avg:.1f}%"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        cloud_impact_path = self.output_dir / 'cloud_impact_analysis.png'
        plt.savefig(cloud_impact_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved cloud impact analysis: {cloud_impact_path}")
    
    def create_comparison_plots(self, sat_data, gt_data, date_key, metrics):
        """
        Create comprehensive comparison plots
        
        Args:
            sat_data: Satellite data array
            gt_data: Ground truth data array
            date_key: Date identifier
            metrics: Calculated metrics dictionary
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Snow Depth Comparison - {date_key}', fontsize=16)
        
        # 1. Satellite data
        im1 = axes[0, 0].imshow(sat_data, cmap='Blues', vmin=0)
        axes[0, 0].set_title('Satellite Snow Depth')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0, 0], label='Snow Depth (m)')
        
        # 2. Ground truth data
        im2 = axes[0, 1].imshow(gt_data, cmap='Blues', vmin=0)
        axes[0, 1].set_title('Ground Truth Snow Depth')
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[0, 1], label='Snow Depth (m)')
        
        # 3. Difference map
        diff = sat_data - gt_data
        im3 = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=np.nanpercentile(diff, 5), 
                               vmax=np.nanpercentile(diff, 95))
        axes[0, 2].set_title('Difference (Satellite - Ground Truth)')
        axes[0, 2].set_xlabel('X (pixels)')
        axes[0, 2].set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=axes[0, 2], label='Difference (m)')
        
        # 4. Scatter plot
        mask = ~(np.isnan(sat_data) | np.isnan(gt_data))
        sat_valid = sat_data[mask]
        gt_valid = gt_data[mask]
        
        if len(sat_valid) > 0:
            axes[1, 0].scatter(gt_valid, sat_valid, alpha=0.5, s=1)
            min_val = min(np.min(gt_valid), np.min(sat_valid))
            max_val = max(np.max(gt_valid), np.max(sat_valid))
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
            axes[1, 0].set_xlabel('Ground Truth (m)')
            axes[1, 0].set_ylabel('Satellite (m)')
            axes[1, 0].set_title('Scatter Plot')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Histograms
        if len(sat_valid) > 0:
            bins = np.linspace(0, max(np.max(sat_valid), np.max(gt_valid)), 50)
            axes[1, 1].hist(gt_valid, bins=bins, alpha=0.7, label='Ground Truth', density=True)
            axes[1, 1].hist(sat_valid, bins=bins, alpha=0.7, label='Satellite', density=True)
            axes[1, 1].set_xlabel('Snow Depth (m)')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Distribution Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Metrics summary
        axes[1, 2].axis('off')
        if 'error' not in metrics:
            # Handle NaN values in display
            rmse_str = f"{metrics['rmse']:.3f}" if not np.isnan(metrics['rmse']) else "N/A"
            mae_str = f"{metrics['mae']:.3f}" if not np.isnan(metrics['mae']) else "N/A"
            bias_str = f"{metrics['bias']:.3f}" if not np.isnan(metrics['bias']) else "N/A"
            r2_str = f"{metrics['r2']:.3f}" if not np.isnan(metrics['r2']) else "N/A"
            corr_str = f"{metrics['correlation']:.3f}" if not np.isnan(metrics['correlation']) else "N/A"
            
            metrics_text = f"""
            Comparison Metrics:
            
            RMSE: {rmse_str} m
            MAE: {mae_str} m
            Bias: {bias_str} m
            R²: {r2_str}
            Correlation: {corr_str}
            
            Data Coverage:
            Valid pixels: {metrics['n_points']:,}
            Coverage: {metrics['data_coverage_percent']:.1f}%
            Sat availability: {metrics['sat_data_availability_percent']:.1f}%
            GT availability: {metrics['gt_data_availability_percent']:.1f}%
            
            Data Quality: {metrics.get('data_quality_flag', 'normal')}
            
            Satellite Stats:
            Mean: {metrics['sat_mean']:.3f} m
            Std: {metrics['sat_std']:.3f} m
            Range: {metrics['sat_min']:.3f} - {metrics['sat_max']:.3f} m
            
            Ground Truth Stats:
            Mean: {metrics['gt_mean']:.3f} m
            Std: {metrics['gt_std']:.3f} m
            Range: {metrics['gt_min']:.3f} - {metrics['gt_max']:.3f} m
            """
        else:
            metrics_text = f"""
            Error: {metrics['error']}
            
            Data Quality Issues:
            Total pixels: {metrics.get('total_pixels', 'N/A'):,}
            Valid pixels: {metrics.get('valid_pixels', 0):,}
            Coverage: {metrics.get('data_coverage_percent', 0):.1f}%
            """
            
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'comparison_{date_key}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {plot_path}")
    
    def run_comparison(self):
        """
        Run the complete comparison analysis
        """
        print("\n" + "="*60)
        print("STARTING SNOW DEPTH COMPARISON ANALYSIS")
        print("="*60)
        
        if not self.matching_dates:
            print("No matching dates found between satellite and ground truth data!")
            return
        
        # Sort dates for chronological processing
        sorted_dates = sorted(self.matching_dates)
        
        for i, date_key in enumerate(sorted_dates):
            print(f"\nProcessing {date_key} ({i+1}/{len(sorted_dates)})")
            print("-" * 40)
            
            try:
                # Read satellite data
                sat_file = self.satellite_files[date_key]
                print(f"Reading satellite: {os.path.basename(sat_file)}")
                
                if RASTERIO_AVAILABLE:
                    try:
                        sat_data, sat_transform, sat_crs = self._read_raster(sat_file)
                    except Exception as e:
                        print(f"Error reading satellite file: {str(e)}")
                        continue
                else:
                    # Fallback for demonstration
                    print("Warning: Using dummy data - install rasterio for real analysis")
                    sat_data = np.random.rand(100, 100) * 2  # Dummy satellite data
                    sat_transform, sat_crs = None, None
                
                # Read ground truth data
                gt_files = self.ground_truth_files[date_key]
                print(f"Reading ground truth: {list(gt_files.keys())}")
                
                if RASTERIO_AVAILABLE:
                    try:
                        gt_data, gt_transform, gt_crs = self._read_raster(gt_files)
                    except Exception as e:
                        print(f"Error reading ground truth file: {str(e)}")
                        continue
                else:
                    # Fallback for demonstration
                    gt_data = np.random.rand(100, 100) * 2  # Dummy ground truth data
                    gt_transform, gt_crs = None, None
                
                # Align rasters
                print("Aligning rasters...")
                try:
                    sat_aligned, gt_aligned = self._align_rasters(
                        sat_data, sat_transform, sat_crs,
                        gt_data, gt_transform, gt_crs
                    )
                except Exception as e:
                    print(f"Error aligning rasters: {str(e)}")
                    continue
                
                # Calculate metrics
                print("Calculating metrics...")
                metrics = self.calculate_metrics(sat_aligned, gt_aligned)
                
                if 'error' in metrics:
                    print(f"Error: {metrics['error']}")
                    
                    # Diagnose the data issues
                    self.diagnose_data_issues(sat_aligned, gt_aligned, date_key)
                    
                    # Still store this result for tracking
                    error_result = {
                        'date': date_key,
                        'year': date_key.split('_')[0],
                        'month': date_key.split('_')[1],
                        'error': metrics['error'],
                        'total_pixels': metrics.get('total_pixels', sat_aligned.size),
                        'data_coverage_percent': metrics.get('data_coverage_percent', 0),
                        'satellite_file': os.path.basename(sat_file),
                        'ground_truth_files': list(gt_files.keys()) if isinstance(gt_files, dict) else [os.path.basename(gt_files)]
                    }
                    self.comparison_results.append(error_result)
                    continue
                
                # Check for data quality issues even in successful comparisons
                if metrics.get('data_coverage_percent', 0) < 10:
                    print(f"Warning: Very low data coverage ({metrics['data_coverage_percent']:.1f}%)")
                    self.diagnose_data_issues(sat_aligned, gt_aligned, date_key)
                
                # Add metadata
                metrics['date'] = date_key
                metrics['year'] = date_key.split('_')[0]
                metrics['month'] = date_key.split('_')[1]
                metrics['satellite_file'] = os.path.basename(sat_file)
                metrics['ground_truth_files'] = list(gt_files.keys()) if isinstance(gt_files, dict) else [os.path.basename(gt_files)]
                
                # Store results
                self.comparison_results.append(metrics)
                
                # Create plots
                print("Creating comparison plots...")
                self.create_comparison_plots(sat_aligned, gt_aligned, date_key, metrics)
                
                # Create data quality analysis
                print("Creating data quality analysis...")
                self.create_data_quality_analysis(sat_aligned, gt_aligned, date_key)
                
                # Print summary with robust NaN handling
                rmse_str = f"{metrics['rmse']:.3f}" if not np.isnan(metrics['rmse']) else "N/A"
                mae_str = f"{metrics['mae']:.3f}" if not np.isnan(metrics['mae']) else "N/A"
                r2_str = f"{metrics['r2']:.3f}" if not np.isnan(metrics['r2']) else "N/A"
                corr_str = f"{metrics['correlation']:.3f}" if not np.isnan(metrics['correlation']) else "N/A"
                
                print(f"RMSE: {rmse_str} m, "
                      f"MAE: {mae_str} m, "
                      f"R²: {r2_str}, "
                      f"Correlation: {corr_str}, "
                      f"Coverage: {metrics['data_coverage_percent']:.1f}%, "
                      f"Quality: {metrics.get('data_quality_flag', 'normal')}")
                
            except Exception as e:
                print(f"Error processing {date_key}: {str(e)}")
                continue
        
        # Generate summary report
        self._generate_summary_report()
        
        print("\n" + "="*60)
        print("COMPARISON ANALYSIS COMPLETED")
        print("="*60)
        
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.comparison_results:
            print("No results to summarize!")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.comparison_results)
        
        # Filter out rows with critical errors (no valid data)
        valid_results = []
        problematic_results = []
        
        for result in self.comparison_results:
            if 'error' in result or result.get('n_points', 0) == 0:
                problematic_results.append(result)
            else:
                valid_results.append(result)
        
        print(f"\nSummary: {len(valid_results)} successful comparisons, {len(problematic_results)} problematic")
        
        if problematic_results:
            print("\nProblematic dates:")
            for result in problematic_results:
                date = result.get('date', 'unknown')
                if 'error' in result:
                    print(f"  {date}: {result['error']}")
                else:
                    coverage = result.get('data_coverage_percent', 0)
                    flag = result.get('data_quality_flag', 'unknown')
                    print(f"  {date}: No valid data (coverage: {coverage:.1f}%, flag: {flag})")
        
        if not valid_results:
            print("No valid results for summary analysis!")
            return
            
        # Create DataFrame from valid results only
        df_valid = pd.DataFrame(valid_results)
        
        # Save detailed results (including problematic ones)
        csv_path = self.output_dir / 'detailed_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detailed results: {csv_path}")
        
        # Create summary statistics from valid results only
        numeric_cols = ['rmse', 'mae', 'bias', 'r2', 'correlation', 'data_coverage_percent']
        available_cols = [col for col in numeric_cols if col in df_valid.columns]
        
        if available_cols:
            summary_stats = df_valid[available_cols].describe()
        else:
            print("Warning: No numeric columns available for summary statistics")
            return
        
        # Create summary plots
        self._create_summary_plots(df_valid)
        
        # Create cloud impact analysis
        if len(df_valid) > 1:
            print("Creating cloud impact analysis...")
            self.analyze_cloud_impact(df_valid)
        else:
            print("Skipping cloud impact analysis (insufficient valid data)")
        
        # Generate text report
        report_path = self.output_dir / 'summary_report.txt'
        with open(report_path, 'w') as f:
            f.write("SNOW DEPTH COMPARISON SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Comparisons Attempted: {len(df)}\n")
            f.write(f"Successful Comparisons: {len(df_valid)}\n")
            f.write(f"Problematic Comparisons: {len(problematic_results)}\n")
            
            if len(df_valid) > 0:
                f.write(f"Date Range: {df_valid['date'].min()} to {df_valid['date'].max()}\n\n")
                
                f.write("SUMMARY STATISTICS (Valid Results Only)\n")
                f.write("-" * 40 + "\n")
                f.write(summary_stats.to_string())
                f.write("\n\n")
                
                # Best/worst analysis only if we have valid RMSE values
                if 'rmse' in df_valid.columns and not df_valid['rmse'].isna().all():
                    f.write("BEST PERFORMING DATES (lowest RMSE)\n")
                    f.write("-" * 35 + "\n")
                    best_cols = ['date', 'rmse', 'mae', 'r2', 'correlation', 'data_coverage_percent']
                    best_cols = [col for col in best_cols if col in df_valid.columns]
                    best_dates = df_valid.nsmallest(5, 'rmse')[best_cols]
                    f.write(best_dates.to_string(index=False))
                    f.write("\n\n")
                    
                    f.write("WORST PERFORMING DATES (highest RMSE)\n")
                    f.write("-" * 36 + "\n")
                    worst_dates = df_valid.nlargest(5, 'rmse')[best_cols]
                    f.write(worst_dates.to_string(index=False))
                    f.write("\n\n")
                
                # Seasonal analysis if enough data
                if len(df_valid) > 12 and 'month' in df_valid.columns:
                    f.write("SEASONAL ANALYSIS\n")
                    f.write("-" * 17 + "\n")
                    seasonal_cols = [col for col in ['rmse', 'mae', 'r2', 'correlation', 'data_coverage_percent'] if col in df_valid.columns]
                    seasonal = df_valid.groupby('month')[seasonal_cols].mean()
                    f.write(seasonal.to_string())
                    f.write("\n\n")
            
            # Report problematic cases
            if problematic_results:
                f.write("PROBLEMATIC COMPARISONS\n")
                f.write("-" * 23 + "\n")
                for result in problematic_results:
                    date = result.get('date', 'unknown')
                    if 'error' in result:
                        f.write(f"{date}: {result['error']}\n")
                    else:
                        coverage = result.get('data_coverage_percent', 0)
                        flag = result.get('data_quality_flag', 'unknown')
                        f.write(f"{date}: No valid overlap (coverage: {coverage:.1f}%, issue: {flag})\n")
                f.write("\n")
        
        print(f"Saved summary report: {report_path}")
        
    def _create_summary_plots(self, df):
        """Create summary visualization plots"""
        
        if len(df) == 0:
            print("No valid data for summary plots")
            return
            
        # Time series plot
        plt.figure(figsize=(15, 10))
        
        # Convert date to datetime for plotting
        df['datetime'] = pd.to_datetime(df['date'], format='%Y_%m')
        df_sorted = df.sort_values('datetime')
        
        plots_created = 0
        
        # Plot 1: RMSE over time
        if 'rmse' in df.columns and not df['rmse'].isna().all():
            plt.subplot(2, 3, 1)
            valid_rmse = df_sorted.dropna(subset=['rmse'])
            if len(valid_rmse) > 0:
                plt.plot(valid_rmse['datetime'], valid_rmse['rmse'], 'b-o', markersize=4)
                plt.title('RMSE Over Time')
                plt.ylabel('RMSE (m)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plots_created += 1
        
        # Plot 2: MAE over time
        if 'mae' in df.columns and not df['mae'].isna().all():
            plt.subplot(2, 3, 2)
            valid_mae = df_sorted.dropna(subset=['mae'])
            if len(valid_mae) > 0:
                plt.plot(valid_mae['datetime'], valid_mae['mae'], 'g-o', markersize=4)
                plt.title('MAE Over Time')
                plt.ylabel('MAE (m)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plots_created += 1
        
        # Plot 3: R² over time
        if 'r2' in df.columns and not df['r2'].isna().all():
            plt.subplot(2, 3, 3)
            valid_r2 = df_sorted.dropna(subset=['r2'])
            if len(valid_r2) > 0:
                plt.plot(valid_r2['datetime'], valid_r2['r2'], 'r-o', markersize=4)
                plt.title('R² Over Time')
                plt.ylabel('R²')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plots_created += 1
        
        # Plot 4: Correlation over time
        if 'correlation' in df.columns and not df['correlation'].isna().all():
            plt.subplot(2, 3, 4)
            valid_corr = df_sorted.dropna(subset=['correlation'])
            if len(valid_corr) > 0:
                plt.plot(valid_corr['datetime'], valid_corr['correlation'], 'm-o', markersize=4)
                plt.title('Correlation Over Time')
                plt.ylabel('Correlation')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plots_created += 1
        
        # Plot 5: Bias over time
        if 'bias' in df.columns and not df['bias'].isna().all():
            plt.subplot(2, 3, 5)
            valid_bias = df_sorted.dropna(subset=['bias'])
            if len(valid_bias) > 0:
                plt.plot(valid_bias['datetime'], valid_bias['bias'], 'c-o', markersize=4)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.title('Bias Over Time')
                plt.ylabel('Bias (m)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plots_created += 1
        
        # Plot 6: Metric distributions
        plt.subplot(2, 3, 6)
        available_metrics = []
        box_data = []
        
        for metric in ['rmse', 'mae', 'r2', 'correlation']:
            if metric in df.columns and not df[metric].isna().all():
                available_metrics.append(metric)
                box_data.append(df[metric].dropna())
        
        if available_metrics and box_data:
            plt.boxplot(box_data, labels=available_metrics)
            plt.title('Metric Distributions')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plots_created += 1
        
        if plots_created > 0:
            plt.tight_layout()
            
            # Save plot
            summary_plot_path = self.output_dir / 'summary_plots.png'
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved summary plots: {summary_plot_path}")
        else:
            plt.close()
            print("No valid metrics available for summary plots")


def main():
    """
    Main function to run the comparison tool
    """
    # Configuration - Update these paths for your data
    satellite_dir = "snow_depth_monthly_202001_202312_500m"
    ground_truth_dir = "snow_monthly_output_2018_to_2021/snod"
    output_dir = "comparison_results"
    
    # Resampling method options:
    # "auto" - Automatically choose best method based on data characteristics (recommended)
    # "nearest" - Nearest neighbor (fastest, preserves original values)
    # "bilinear" - Bilinear interpolation (good balance of speed/quality)
    # "cubic" - Cubic interpolation (higher quality, slower)
    # "lanczos" - Lanczos resampling (highest quality, slowest)
    resampling_method = "auto"
    
    print("Snow Depth Satellite vs Ground Truth Comparison Tool")
    print("=" * 55)
    print(f"Satellite directory: {satellite_dir}")
    print(f"Ground truth directory: {ground_truth_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resampling method: {resampling_method}")
    
    # Create and run comparator with proper resampling
    comparator = SnowDepthComparator(satellite_dir, ground_truth_dir, output_dir, resampling_method)
    comparator.run_comparison()
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("- Individual comparison plots (comparison_YYYY_MM.png)")
    print("- Detailed results CSV (detailed_results.csv)")
    print("- Summary plots (summary_plots.png)")
    print("- Summary report (summary_report.txt)")


if __name__ == "__main__":
    main()