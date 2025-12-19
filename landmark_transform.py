# -*- coding: utf-8 -*-
"""
Landmark Transform Pipeline for 3D Slicer

A modular, open-access tool for transforming anatomical landmarks and segmentations
from template space to subject space using ANTs registrations.

Features:
  - Cross-platform path handling (Windows/Mac/Linux)
  - Configurable via YAML or CLI
  - Robust Slicer integration (lazy imports)
  - Comprehensive logging
  - Checkpoint/recovery system

License: MIT
Author: GeTang
"""
#%%
import os
import sys
sys.path.append(r'D:\users\getang\SANS\Slicertools')
import slicerutil_getang as su
import re
import json
import logging
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle


#%% Optional Slicer imports (graceful degradation if not in Slicer environment)
try:
    import slicer
    from slicer.util import loadVolume, loadMarkupsFiducialList, loadSegmentation, loadModel, saveNode
    IN_SLICER = True
except ImportError:
    IN_SLICER = False
    logging.warning("Not running in 3D Slicer; Slicer-specific functions will be unavailable.")

# Set up logging
log = logging.getLogger(__name__)

# ============================================================================
# Configuration & Constants
# ============================================================================

class Region(Enum):
    """Anatomical regions for centerline extraction."""
    L_OPTIC_NERVE = "L_optic_nerve"
    R_OPTIC_NERVE = "R_optic_nerve"
    L_OPTIC_NERVE_SHEATH = "L_optic_nerve_sheath_anterior_with_nerve"
    R_OPTIC_NERVE_SHEATH = "R_optic_nerve_sheath_anterior_with_nerve"

@dataclass
class RegionConfig:
    """Configuration for a single region."""
    region: str
    fids: str  # fiducial list name for endpoints
    resample_number: int = 400
    closed_curve_option: int = 0

@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    template_dir: Path
    template_pattern: str  # e.g., "T_{cohort}T1.nii.gz"
    input_dir: Path
    input_pattern: str  # e.g., "Denoised_*_T1w.nii"
    fids_dir: Path
    fids_pattern: str  # e.g., "fids_{cohort}.fcsv"
    segs_pattern: str  # e.g., "Seg_{cohort}.seg.nrrd"
    cohort: str  # e.g., "Astro02mm"
    output_subdir: str = "SANS"  # subdirectory for outputs
    save_visualization: bool = False
    demo_mode: bool = False
    verbose: bool = False

@dataclass
class CenterlineMetrics:
    """Container for centerline properties."""
    region: str
    radius: float = 0.0
    length_mm: float = 0.0
    curvature: float = 0.0
    torsion: float = 0.0
    tortuosity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with region-prefixed keys."""
        return {
            f'{self.region}_radius': self.radius,
            f'{self.region}_length_mm': self.length_mm,
            f'{self.region}_curvature': self.curvature,
            f'{self.region}_torsion': self.torsion,
            f'{self.region}_tortuosity': self.tortuosity,
        }

@dataclass
class SheathCrossSectionAreaMetrics:
    """Container for cross-section area measurements at specific gaps."""
    region: str
    gap_mm: int
    area_mm2: float = 0.0
    interpolated: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with region and gap-prefixed keys."""
        prefix = f'{self.region}_{self.gap_mm}mm'
        suffix = '_interpolated' if self.interpolated else '_raw'
        return {f'{prefix}_area{suffix}': self.area_mm2}

# ============================================================================
# Centerline Extraction (Slicer 5.6+ compatible)
# ============================================================================

class CenterlineExtractor:
    """Extract centerlines from segmented structures (Slicer 5.6+ compatible)."""
    
    def __init__(self, slicerutil_module=None):
        """
        Args:
            slicerutil_module: Reference to slicerutil (su) module
        """
        self.su = slicerutil_module
        if not IN_SLICER:
            raise RuntimeError("CenterlineExtractor requires 3D Slicer environment.")
        if not self.su:
            log.warning("slicerutil module not provided; some functions may fail")
    
    def extract_centerline(
        self,
        seg_node_name: str,
        region_name: str,
        centerline_metrics_name: str,
        fiducial_node_name: str,
        preprocess_surface: bool = True,
    ) -> Tuple[Any, CenterlineMetrics]:
        """
        Extract centerline from a segmented region.
        
        Args:
            seg_node_name: Name of segmentation node
            region_name: Name of region in segmentation (e.g., 'L_ON')
            fiducial_node_name: Name of fiducial list with endpoints
            preprocess_surface: Whether to preprocess the surface
        
        Returns:
            (centerlinePolyData, CenterlineMetrics)
        """
        if not self.su:
            raise RuntimeError("slicerutil module required for centerline extraction")
        
        try:
            log.info(f"Extracting centerline for {region_name}...")
            
            # Extract centerline using slicerutil
            centerline_polydata, _ = self.su.centerlineExtractcenterline(
                seg_node_name, region_name, fiducial_node_name,
                preprocesssurface=preprocess_surface
            )
            
            num_points = centerline_polydata.GetNumberOfPoints()
            log.debug(f"Initial centerline: {num_points} points")
            
            # Fallback: try without preprocessing if too few points
            if num_points < 10:
                log.warning(f"{region_name}: insufficient points ({num_points}); "
                           "retrying without preprocessing")
                centerline_polydata, _ = self.su.centerlineExtractcenterline(
                    seg_node_name, region_name, fiducial_node_name,
                    preprocesssurface=False
                )
                num_points = centerline_polydata.GetNumberOfPoints()
                log.info(f"Centerline after retry: {num_points} points")
            
            # Compute metrics
            metrics = self._compute_centerline_metrics(centerline_polydata, region_name)
            # log.info(f"Extracted centerline for {region_name}: {metrics.num_points} points, "
            #         f"length={metrics.length_mm:.2f}mm, curvature={metrics.mean_curvature:.4f}")
            
            return centerline_polydata, metrics
        
        except Exception as e:
            log.error(f"Failed to extract centerline for {region_name}: {e}", exc_info=True)
            raise
    
    def _compute_centerline_metrics(self, polydata: Any, region_name: str) -> CenterlineMetrics:
        """Compute metrics from centerline polydata."""
        try:
            metrics = self.su.centerlineGetcenterlineproperties(polydata, region_name)
            length = metrics.GetTable().GetColumnByName('Length').GetValue(0)
            curvature = metrics.GetTable().GetColumnByName('Curvature').GetValue(0)
            raduis = metrics.GetTable().GetColumnByName('Radius').GetValue(0)
            torsion = metrics.GetTable().GetColumnByName('Torsion').GetValue(0)
            tortuosity = metrics.GetTable().GetColumnByName('Tortuosity').GetValue(0)

            return CenterlineMetrics(
                region=region_name,
                radius=raduis,
                length_mm=length,
                curvature=curvature,
                torsion=torsion,
                tortuosity=tortuosity,
            )
        except Exception as e:
            log.error(f"Error computing metrics: {e}")
            return CenterlineMetrics(region=region_name)


    # def _compute_centerline_metrics(self, polydata: Any, region_name: str) -> CenterlineMetrics:
    #     """Compute metrics from centerline polydata."""
    #     try:
    #         num_points = polydata.GetNumberOfPoints()
            
    #         # Compute length along centerline
    #         length = 0.0
    #         if num_points > 1:
    #             points = polydata.GetPoints()
    #             for i in range(num_points - 1):
    #                 p0 = np.array(points.GetPoint(i))
    #                 p1 = np.array(points.GetPoint(i + 1))
    #                 length += float(np.linalg.norm(p1 - p0))
            
    #         # Compute mean curvature (angle-based)
    #         mean_curvature = 0.0
    #         if num_points > 2:
    #             points = polydata.GetPoints()
    #             angles = []
    #             for i in range(1, num_points - 1):
    #                 p0 = np.array(points.GetPoint(i - 1))
    #                 p1 = np.array(points.GetPoint(i))
    #                 p2 = np.array(points.GetPoint(i + 1))
                    
    #                 v1 = p1 - p0
    #                 v2 = p2 - p1
    #                 v1_norm = np.linalg.norm(v1)
    #                 v2_norm = np.linalg.norm(v2)
                    
    #                 if v1_norm > 0 and v2_norm > 0:
    #                     cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    #                     cos_angle = np.clip(cos_angle, -1, 1)
    #                     angle = np.arccos(cos_angle)
    #                     angles.append(float(angle))
                
    #             mean_curvature = float(np.mean(angles)) if angles else 0.0
            
    #         # Extract diameter scalars if available
    #         max_diameter = 0.0
    #         min_diameter = 0.0
    #         try:
    #             scalars = polydata.GetPointData().GetScalars()
    #             if scalars and scalars.GetNumberOfTuples() > 0:
    #                 diameters = [float(scalars.GetValue(i)) 
    #                            for i in range(scalars.GetNumberOfTuples())]
    #                 max_diameter = float(max(diameters))
    #                 min_diameter = float(min(diameters))
    #         except Exception as e:
    #             log.debug(f"Could not extract diameter scalars: {e}")
            
    #         return CenterlineMetrics(
    #             region=region_name,
    #             length_mm=length,
    #             num_points=num_points,
    #             mean_curvature=mean_curvature,
    #             max_diameter=max_diameter,
    #             min_diameter=min_diameter,
    #         )
        
    #     except Exception as e:
    #         log.error(f"Error computing metrics: {e}")
    #         return CenterlineMetrics(region=region_name)

    
    def save_centerline_model(
        self,
        centerline_polydata: Any,
        model_name: str,
        output_path: Path
    ) -> Path:
        """Save centerline as VTK model file."""
        if not self.su:
            raise RuntimeError("slicerutil module required")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_node = self.su.centerlineGetmodel(centerline_polydata, 1, model_name)
            saveNode(model_node, str(output_path))
            log.info(f"Saved centerline model: {output_path}")
            return output_path
        
        except Exception as e:
            log.error(f"Failed to save centerline model: {e}")
            raise

# ============================================================================
# Sheath Diameter Extraction (Improved with GetCrossSectionArea)
# ============================================================================

class SheatDiameterExtractor:
    """Extract sheath diameter using GetCrossSectionArea (simpler approach)."""
    
    def __init__(self, slicerutil_module=None):
        """
        Args:
            slicerutil_module: Reference to slicerutil (su) module
        """
        self.su = slicerutil_module
        if not IN_SLICER:
            raise RuntimeError("SheatDiameterExtractor requires 3D Slicer environment.")
        if not self.su:
            raise RuntimeError("slicerutil module required for diameter extraction")
    
    def extract_cross_section_areas(
        self,
        regions: List[Dict[str, str]],
        centerline_model_prefix: str,
        fiducial_node_name: str,
        seg_node_name: str,
        gaps_mm: List[int] = None,
        interpolate: bool = True
    ) -> Dict[str, List[SheathCrossSectionAreaMetrics]]:
        """
        Extract cross-sectional areas at specific gaps using GetCrossSectionArea.
        
        Args:
            regions: List of region configs [{'region': 'L_ON', 'segment': 'L_ONS'}, ...]
            centerline_model_prefix: Prefix for centerline model names (e.g., 'Centerlinemodel')
            fiducial_node_name: Name of fiducial list node
            seg_node_name: Name of segmentation node
            gaps_mm: List of gap distances (default: [3, 5])
            interpolate: Whether to interpolate the cross-section
        
        Returns:
            Dictionary mapping region names to lists of metrics
        """
        if gaps_mm is None:
            gaps_mm = [3, 5]
        
        log.info(f"Extracting cross-section areas for {len(regions)} regions at gaps {gaps_mm}mm")
        
        all_metrics = {}
        
        for region_cfg in regions:
            region_name = region_cfg['region']
            segment_name = region_cfg.get('segment', f'{region_name}S')  # Default to region + 'S'
            region_metrics = []
            
            try:
                centerline_model_name = f'{region_name}_{centerline_model_prefix}'
                for gap_mm in gaps_mm:
                    # Extract area with interpolation
                    area_interpolated = self.su.GetCrossSectionArea(
                        centerline_model_name, fiducial_node_name, seg_node_name,
                        segment_name, gap=gap_mm, interpolate=True
                    )
                    
                    metrics_interp = SheathCrossSectionAreaMetrics(
                        region=region_name,
                        gap_mm=gap_mm,
                        area_mm2=float(area_interpolated),
                        interpolated=True
                    )
                    region_metrics.append(metrics_interp)
                    
                    # Optionally extract without interpolation
                    if not interpolate:
                        area_raw = self.su.GetCrossSectionArea(
                            centerline_model_name, fiducial_node_name, seg_node_name,
                            segment_name, gap=gap_mm, interpolate=False
                        )
                        
                        metrics_raw = SheathCrossSectionAreaMetrics(
                            region=region_name,
                            gap_mm=gap_mm,
                            area_mm2=float(area_raw),
                            interpolated=False
                        )
                        region_metrics.append(metrics_raw)
                    
                    log.info(f"{region_name} at {gap_mm}mm: area={area_interpolated:.2f}mm²")
            
            except Exception as e:
                log.error(f"Failed to extract cross-section areas for {region_name}: {e}")
            
            all_metrics[region_name] = region_metrics
        
        return all_metrics
    
    def setup_visualization(
        self,
        seg_node_name: str,
        regions: List[Dict[str, str]],
        centerline_model_prefix: str,
        fiducial_node_name: str,
        scale_factor: float = 2.0,
        opacity_3d: float = 0.3,
        view_axis: str = 'superior',
        gaps_mm: List[int] = None
    ):
        """
        Set up 3D visualization for cross-section display.
        
        Args:
            seg_node_name: Name of segmentation node
            model_folder_id: Subject hierarchy folder ID for models (optional)
            scale_factor: Zoom scale factor
            opacity_3d: 3D opacity for segmentation
            view_axis: View axis ('superior', 'anterior', etc.)
        """
        try:
            # Set layout and zoom
            self.su.setLayout(4)
            self.su.zoom3D(scale_factor)
            
            for region_cfg in regions:
                region_name = region_cfg['region']
                segment_name = region_cfg.get('segment', f'{region_name}S')  # Default to region + 'S'
                region_metrics = []
                # Hide model folder if provided
                centerline_model_name = f'{region_name}_{centerline_model_prefix}'
                for gap_mm in gaps_mm:
                    # Optionally extract without interpolation
                    area_raw = self.su.GetCrossSectionArea(
                        centerline_model_name, fiducial_node_name, seg_node_name,
                        segment_name, gap=gap_mm, interpolate=False
                    )
                    log.info(f"Displayed {region_name} at {gap_mm}mm: area={area_raw:.2f}mm²")

            # Set segmentation visualization
            self.su.segmentationSetVisualization(seg_node_name, opacity3D=opacity_3d)
            
            # Set view
            self.su.view3D_lookFromViewAxis(view_axis)
            self.su.view3D_center()
            
            log.info("Visualization setup complete")
        
        except Exception as e:
            log.error(f"Failed to setup visualization: {e}")
    
    def capture_visualization(
        self,
        output_dir: Path,
        subject_id: str,
        fnroot: str,
        modality: str = "T2"
    ) -> Path:
        """
        Capture visualization screenshot.
        
        Args:
            output_dir: Output directory
            subject_id: Subject identifier
            fnroot: Filename root
            modality: Modality string
        
        Returns:
            Path to captured image
        """
        try:
            ff_img = output_dir / f'Crosssectionarea_{subject_id}_{fnroot}_{modality}.png'
            self.su.captureImageFromAllViews(str(ff_img))
            log.info(f"Captured visualization: {ff_img}")
            return ff_img
        
        except Exception as e:
            log.error(f"Failed to capture visualization: {e}")
            raise

# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(level)

def ensure_path(path: Path, is_dir: bool = False) -> Path:
    """Ensure path exists; create if needed."""
    path = Path(path).resolve()
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def extract_subject_id(filename: str, pattern: str = r'sub-(.+?)_ses') -> Optional[str]:
    """Extract subject ID from filename using regex pattern."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def extract_session_id(filename: str, pattern: str = r'ses-(.+?)_') -> Optional[str]:
    """Extract session ID from filename using regex pattern."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def splitext(fn: str) -> Tuple[str, str]:
    """Split filename, handling .nii.gz properly."""
    if fn.endswith('.nii.gz'):
        return (fn[:-7], '.nii.gz')
    return os.path.splitext(fn)

def locate_files(pattern: str, root_dir: Path, level: int = 3, sorted_: bool = True) -> List[Path]:
    """
    Locate files matching pattern (fnmatch style).
    Args:
        pattern: fnmatch-style pattern (e.g., "Denoised*.nii")
        root_dir: root directory to search
        level: maximum depth to search
        sorted_: whether to sort results
    """
    from fnmatch import fnmatch
    root_dir = Path(root_dir)
    matches = []
    
    def walk(d, depth):
        if depth > level:
            return
        for item in d.iterdir():
            if item.is_dir():
                walk(item, depth + 1)
            elif fnmatch(item.name, pattern):
                matches.append(item)
    
    walk(root_dir, 0)
    return sorted(matches) if sorted_ else matches

def save_dict_to_csv(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    filepath: Path | str
) -> bool:
    """
    Save dictionary data to CSV.
    
    Supports three input formats:
        1. Dict of lists: {'col1': [1, 2], 'col2': [3, 4]} -> multiple rows
        2. Dict of scalars: {'col1': 1, 'col2': 2} -> single row
        3. List of dicts: [{'col1': 1}, {'col1': 2}] -> multiple rows
    
    Args:
        data: Dictionary or list of dictionaries to save
        filepath: Path to save the CSV file
        
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    
    if not data:
        log.warning("No data to save to CSV: %s", filepath)
        return False
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to list of row dictionaries (unified format)
        rows = _normalize_to_rows(data)
        
        if not rows:
            log.warning("No rows to write to CSV: %s", filepath)
            return False
        
        keys = list(rows[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        
        log.info("Saved CSV with %d rows: %s", len(rows), filepath)
        return True
        
    except PermissionError:
        log.error("Permission denied writing to %s", filepath)
        return False
    except OSError as e:
        log.error("Failed to save CSV %s: %s", filepath, e)
        return False


def _normalize_to_rows(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert various input formats to a list of row dictionaries."""
    
    # Format 3: Already a list of dicts
    if isinstance(data, list):
        return data
    
    # Check if values are lists (Format 1) or scalars (Format 2)
    first_value = next(iter(data.values()))
    
    if isinstance(first_value, list):
        # Format 1: Dict of lists -> transpose to list of dicts
        keys = list(data.keys())
        return [
            dict(zip(keys, row_vals)) 
            for row_vals in zip(*[data[k] for k in keys])
        ]
    else:
        # Format 2: Dict of scalars -> wrap as single row
        return [data]

# def save_dict_to_csv(data: Dict[str, List[Any]], filepath: Path):
#     """Save dictionary of lists to CSV."""
#     filepath = Path(filepath)
#     try:
#         import csv
#         filepath.parent.mkdir(parents=True, exist_ok=True)
#         if not data:
#             log.warning("No data to save to CSV: %s", filepath)
#             return
        
#         keys = list(data.keys())
#         print(keys)
#         print(data)
#         with open(filepath, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=keys)
#             writer.writeheader()
#             writer.writerow(data) # Assuming single row of data
#         log.info("Saved CSV: %s", filepath)
#     except Exception as e:
#         log.error("Failed to save CSV %s: %s", filepath, e)

def save_pickle(obj: Any, filepath: Path):
    """Save object to pickle file."""
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        log.info("Saved pickle: %s", filepath)
    except Exception as e:
        log.error("Failed to save pickle %s: %s", filepath, e)

# ============================================================================
# Slicer Integration (only when IN_SLICER)
# ============================================================================

class SlicerHelper:
    """Helper class for Slicer operations (graceful fallback if not in Slicer)."""
    
    @staticmethod
    def is_available():
        return IN_SLICER
    
    @staticmethod
    def load_volume(filepath: str, return_node: bool = False):
        if not IN_SLICER:
            raise RuntimeError("Slicer environment not available.")
        return loadVolume(filepath, returnNode=return_node)
    
    @staticmethod
    def load_fiducials(filepath: str, return_node: bool = False):
        if not IN_SLICER:
            raise RuntimeError("Slicer environment not available.")
        return loadMarkupsFiducialList(filepath, returnNode=return_node)
    
    @staticmethod
    def load_segmentation(filepath: str, return_node: bool = False):
        if not IN_SLICER:
            raise RuntimeError("Slicer environment not available.")
        return loadSegmentation(filepath, returnNode=return_node)
    
    @staticmethod
    def load_model(filepath: str, return_node: bool = False):
        if not IN_SLICER:
            raise RuntimeError("Slicer environment not available.")
        return loadModel(filepath, returnNode=return_node)
    
    @staticmethod
    def save_node(node, filepath: str):
        if not IN_SLICER:
            raise RuntimeError("Slicer environment not available.")
        return saveNode(node, filepath)
    
    @staticmethod
    def close_scene():
        if not IN_SLICER:
            log.warning("Not in Slicer; cannot close scene.")
            return
        slicer.mrmlScene.Clear(False)

# ============================================================================
# Landmark Transform Pipeline
# ============================================================================

class LandmarkTransformPipeline:
    """Main pipeline for landmark transformation."""
    
    def __init__(
        self,
        config: PipelineConfig,
        slicerutil_module=None,
        extract_centerlines: bool = True,
        extract_sheath_diameter: bool = True,
        sheath_gaps: List[int] = None
    ):
        self.config = config
        self.su = slicerutil_module
        self.extract_centerlines = extract_centerlines
        self.extract_sheath_diameter = extract_sheath_diameter
        self.sheath_gaps = sheath_gaps or [3, 5]  # Default gaps in mm
        setup_logging(config.verbose)
    
        # Initialize extractors
        if IN_SLICER and self.su:
            if extract_centerlines:
                self.centerline_extractor = CenterlineExtractor(self.su)
                log.info("Initialized CenterlineExtractor")
            
            if extract_sheath_diameter:
                self.sheath_extractor = SheatDiameterExtractor(self.su)
                log.info("Initialized SheatDiameterExtractor")
        self._validate_config()

    def _validate_config(self):
        """Validate configuration paths exist."""
        for attr in ['template_dir', 'input_dir', 'fids_dir']:
            path = getattr(self.config, attr)
            if not path.exists():
                log.warning(f"Config path does not exist: {attr}={path}")
    
    def run(self):
        """Execute the full pipeline."""
        if not IN_SLICER:
            raise RuntimeError("This pipeline must run inside 3D Slicer.")
        
        log.info(f"Starting pipeline for cohort: {self.config.cohort}")
        
        # Locate input MRI files
        mri_files = locate_files(self.config.input_pattern, self.config.input_dir, level=3)
        if not mri_files:
            log.warning(f"No MRI files found matching {self.config.input_pattern}")
            return
        
        log.info(f"Found {len(mri_files)} MRI files.")
        
        for idx, mri_file in enumerate(mri_files):
            try:
                self._process_subject(idx, mri_file, len(mri_files))
            except Exception as e:
                log.error(f"Failed to process {mri_file}: {e}", exc_info=True)
                if not self.config.demo_mode:
                    continue
                else:
                    raise
    
    def _process_subject(self, idx: int, mri_file: Path, total: int):
        """Process a single subject."""
        log.info(f"[{idx+1}/{total}] Processing: {mri_file.name}")
        
        subject_dir = mri_file.parent
        pn_out = ensure_path(subject_dir / self.config.output_subdir, is_dir=True)
        
        fnroot, _ = splitext(mri_file.name)
        subject_id = extract_subject_id(mri_file.name)
        ses_id = extract_session_id(mri_file.name)
        if not subject_id:
            log.warning(f"Could not extract subject ID from {mri_file.name}")
            return
        if not ses_id:
            log.warning(f"Could not extract session ID from {mri_file.name}")
            return

        # Load template data
        template_file = self.config.template_dir / self.config.template_pattern.format(cohort=self.config.cohort)
        if not template_file.exists():
            log.error(f"Template not found: {template_file}")
            return
        
        try:
            # Load volumes and transforms
            success, n_t1 = SlicerHelper.load_volume(str(mri_file), return_node=True)
            success, n_atl = SlicerHelper.load_volume(str(template_file), return_node=True)
            
            # Load fiducials and segmentations
            fids_file = self.config.fids_dir / self.config.fids_pattern.format(cohort=self.config.cohort)
            segs_file = self.config.fids_dir / self.config.segs_pattern.format(cohort=self.config.cohort)
            
            n_fids = SlicerHelper.load_fiducials(str(fids_file), return_node=True)
            
            if segs_file.exists():
                success, n_segs = SlicerHelper.load_segmentation(str(segs_file), return_node=True)
            else:
                log.warning(f"Segmentation not found: {segs_file}")
                n_segs = None
            
            # Load transforms (affine then deformable) 'trf-Temp_to_sub-02_ses-01_DEF'
            fn_aff = f'trf-Temp_to_sub-{subject_id}_ses-{ses_id}_AFF.mat'
            fn_def = f'trf-Temp_to_sub-{subject_id}_ses-{ses_id}_DEF.nii.gz'
            ff_aff = pn_out / fn_aff
            ff_def = pn_out / fn_def
            
            if not ff_aff.exists() or not ff_def.exists():
                log.error(f"Transforms not found: {ff_aff}, {ff_def}")
                return
            
            success, n_trf_aff = slicer.util.loadTransform(str(ff_aff), returnNode=True)
            success, n_trf_def = slicer.util.loadTransform(str(ff_def), returnNode=True)
            
            # Chain transforms: deformable depends on affine
            n_trf_aff.SetAndObserveTransformNodeID(n_trf_def.GetID())
            n_atl.SetAndObserveTransformNodeID(n_trf_aff.GetID())
            n_fids.SetAndObserveTransformNodeID(n_trf_aff.GetID())
            if n_segs:
                n_segs.SetAndObserveTransformNodeID(n_trf_aff.GetID())
            
            # Harden transforms (apply and remove)
            n_fids.HardenTransform()
            if n_segs:
                n_segs.HardenTransform()
                n_segs.CreateClosedSurfaceRepresentation()
            
            # Save transformed data
            fn_fids_out = f'fids_{self.config.cohort}_on_sub-{subject_id}_ses-{ses_id}.fcsv'
            fn_segs_out = f'Segs_{self.config.cohort}_on_sub-{subject_id}_ses-{ses_id}.seg.nrrd'
            
            ff_fids_out = pn_out / fn_fids_out
            ff_segs_out = pn_out / fn_segs_out
            
            SlicerHelper.save_node(n_fids, str(ff_fids_out))
            if n_segs:
                SlicerHelper.save_node(n_segs, str(ff_segs_out))
            
            log.info(f"Saved outputs to {pn_out}")
            
            centerline_metrics_dict = {}
            diameter_metrics_dict = {}
            
            if self.extract_centerlines and self.centerline_extractor:
                centerline_metrics_dict = self._extract_centerlines(
                    pn_out, subject_id, fnroot,
                    'Segs_{}'.format(self.config.cohort),
                    'fids_{}'.format(self.config.cohort),
                    'centerlinemetrics'
                )
            
            if self.extract_sheath_diameter and self.sheath_extractor:
                diameter_metrics_dict = self._extract_sheath_diameter(
                    pn_out, subject_id, fnroot,
                    'Segs_{}'.format(self.config.cohort),
                    'fids_{}'.format(self.config.cohort)
                )
            
            # Merge and save all metrics
            all_metrics = {**centerline_metrics_dict, **diameter_metrics_dict}
            
            if all_metrics:
                fn_metrics = f'AllMetrics_{subject_id}_{fnroot}.csv'
                ff_metrics = pn_out / fn_metrics
                save_dict_to_csv(all_metrics, ff_metrics)
            if not self.config.demo_mode:
                SlicerHelper.close_scene()
        
        except Exception as e:
            log.error(f"Error processing subject {subject_id}: {e}", exc_info=True)
            raise

    def _extract_endpoints_from_fiducials(
        self,
        fid_node_name: str,
        region_name: str
    ) -> str:
        """
        Extract endpoint fiducial list name for a region.
        
        Args:
            fid_node_name: Source fiducial node name (e.g., 'fids_Cosmo02mm')
            region_name: Region name (e.g., 'L_ON')
        
        Returns:
            New fiducial list node name with only endpoints
        """
        try:
            from slicer.util import getNode
            region_letter = re.search(r'^(.)', region_name)[0]  # L or R
            
            # Get source fiducial node
            fid_node = getNode(fid_node_name)
            if not fid_node:
                raise ValueError(f"Fiducial node not found: {fid_node_name}")
            
            # Expected endpoint labels
            endpoint_labels = [
                f'nerve_tip_{region_letter}',
                f'{region_name}_start_of_bone_part'
            ]
            
            log.debug(f"Looking for endpoints: {endpoint_labels}")
            
            # Extract endpoints into new fiducial list
            endpoints_fid_name = f'{region_name}_endpoints'
            endpoints_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", endpoints_fid_name
            )
            
            found_count = 0
            for i in range(fid_node.GetNumberOfControlPoints()):
                label = fid_node.GetNthControlPointLabel(i)
                
                # Check if this label is one of our endpoints
                if label in endpoint_labels:
                    pos = fid_node.GetNthControlPointPositionWorld(i)
                    endpoints_node.AddControlPoint(
                        vtk.vtkVector3d(pos[0], pos[1], pos[2]),
                        label
                    )
                    found_count += 1
                    log.debug(f"  Added endpoint: {label}")
            
            if found_count < 2:
                log.warning(f"Only found {found_count} endpoints for {region_name}")
            
            log.info(f"Created endpoint list '{endpoints_fid_name}' with {found_count} points")
            return endpoints_fid_name
        
        except Exception as e:
            log.error(f"Failed to extract endpoints: {e}", exc_info=True)
            raise
    
    def _extract_centerlines(
        self,
        output_dir: Path,
        subject_id: str,
        fnroot: str,
        seg_node_name: str,
        fid_node_name: str,
        centerline_metrics_name: str
    ) -> Dict[str, float]:
        """Extract centerlines for all regions."""
        log.info(f"Extracting centerlines for {subject_id}")
        
        regions_for_centerline = [
            {'region': 'L_ON', 'fids': 'L_ON_endpoints'},
            {'region': 'R_ON', 'fids': 'R_ON_endpoints'},
        ]
        
        metrics_dict = {}
        
        for region_cfg in regions_for_centerline:
            region_name = region_cfg['region']
            try:
                # Step 1: Extract endpoints from fiducial list
                endpoints_fid_name = self._extract_endpoints_from_fiducials(
                    fid_node_name, region_name
                )

                # Step 2: Extract centerline using the extracted endpoints
                centerline_polydata, centerline_metrics = \
                    self.centerline_extractor.extract_centerline(
                        seg_node_name, region_name, centerline_metrics_name, 
                        endpoints_fid_name, preprocess_surface=False)
                
                # Add to metrics dict
                metrics_dict.update(centerline_metrics.to_dict())
                
                # Save centerline model
                model_name = f'{region_name}_Centerlinemodel'
                fn_model = f'{model_name}_{subject_id}_{fnroot}.vtk'
                ff_model = output_dir / fn_model
                self.centerline_extractor.save_centerline_model(
                    centerline_polydata, model_name, ff_model
                )
                
                log.info(f"Saved centerline model for {region_name}: {ff_model}")
            
            except Exception as e:
                log.error(f"Failed centerline extraction for {region_name}: {e}")
                continue
        
        return metrics_dict
    
    def _extract_sheath_diameter(
        self,
        output_dir: Path,
        subject_id: str,
        fnroot: str,
        seg_node_name: str,
        fid_node_name: str,
        model_folder_id: str = None,
        modality: str = "T2"
    ) -> Dict[str, float]:
        """Extract sheath diameter metrics using GetCrossSectionArea."""
        log.info(f"Extracting sheath diameter for {subject_id}")
        
        regions_for_diameter = [
            {'region': 'L_ON', 'segment': 'L_ONS'},
            {'region': 'R_ON', 'segment': 'R_ONS'},
        ]
        
        # Extract cross-section areas
        all_metrics = self.sheath_extractor.extract_cross_section_areas(
            regions=regions_for_diameter,
            centerline_model_prefix='Centerlinemodel',
            fiducial_node_name=fid_node_name,
            seg_node_name=seg_node_name,
            gaps_mm=self.sheath_gaps,
            interpolate=True
        )
        
        # Flatten metrics to dictionary
        metrics_dict = {}
        for region_name, region_metrics in all_metrics.items():
            for metric in region_metrics:
                metrics_dict.update(metric.to_dict())
        
        # Setup visualization
        self.sheath_extractor.setup_visualization(
            seg_node_name=seg_node_name,
            regions=regions_for_diameter,
            centerline_model_prefix='Centerlinemodel',
            fiducial_node_name=fid_node_name,
            scale_factor=2.0,
            opacity_3d=0.3,
            view_axis='superior',
            gaps_mm=self.sheath_gaps,
        )
        
        # Save cross-section areas to CSV
        fn_csv = f'CrosssectionArea_{subject_id}_{fnroot}.csv'
        ff_csv = output_dir / fn_csv
        save_dict_to_csv(metrics_dict, ff_csv)
        log.info(f"Saved cross-section areas: {ff_csv}")
        
        # Capture visualization
        if self.config.save_visualization:
            self.sheath_extractor.capture_visualization(
                output_dir=output_dir,
                subject_id=subject_id,
                fnroot=fnroot,
                modality=modality
            )
        
        return metrics_dict

# ============================================================================
# CLI Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None):
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Landmark Transform Pipeline for 3D Slicer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with explicit paths
  python landmark_transform.py \\
    --template-dir /data/templates \\
    --input-dir /data/subjects \\
    --output-subdir SANS \\
    --cohort Astro02mm

  # Run in demo mode (single subject)
  python landmark_transform.py ... --demo
        """
    )
    
    parser.add_argument("--template-dir", type=Path, required=True,
                        help="Directory containing template volumes")
    parser.add_argument("--template-pattern", type=str, default="T_{cohort}T1.nii.gz",
                        help="Filename pattern for templates (default: T_{cohort}T1.nii.gz)")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Root directory containing subject MRI files")
    parser.add_argument("--input-pattern", type=str, default="Denoised_*_T1w.nii",
                        help="Filename pattern for input MRI (default: Denoised_*_T1w.nii)")
    parser.add_argument("--fids-dir", type=Path, required=True,
                        help="Directory containing template fiducials")
    parser.add_argument("--fids-pattern", type=str, default="fids_{cohort}.fcsv",
                        help="Filename pattern for fiducials (default: fids_{cohort}.fcsv)")
    parser.add_argument("--segs-pattern", type=str, default="Seg_{cohort}.seg.nrrd",
                        help="Filename pattern for segmentations (default: Seg_{cohort}.seg.nrrd)")
    parser.add_argument("--cohort", type=str, required=True,
                        help="Cohort name (e.g., Astro02mm, Cosmo02mm)")
    parser.add_argument("--output-subdir", type=str, default="SANS",
                        help="Output subdirectory name (default: SANS)")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (single subject, verbose)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--save-visualization", action="store_true",
                        help="Save visualization screenshots")
    
    args = parser.parse_args(argv)
    
    if not IN_SLICER:
        print("ERROR: This script must be run in 3D Slicer's Python environment.")
        print("In Slicer, go to: Python Interactor → Execute script")
        return 1
    
    config = PipelineConfig(
        template_dir=args.template_dir,
        template_pattern=args.template_pattern,
        input_dir=args.input_dir,
        input_pattern=args.input_pattern,
        fids_dir=args.fids_dir,
        fids_pattern=args.fids_pattern,
        segs_pattern=args.segs_pattern,
        cohort=args.cohort,
        output_subdir=args.output_subdir,
        demo_mode=args.demo,
        verbose=args.verbose or args.demo,
        save_visualization=args.save_visualization,
    )
    
    try:
        pipeline = LandmarkTransformPipeline(config, 
                                             slicerutil_module=su,
                                             extract_centerlines=True,
                                             extract_sheath_diameter=True,
                                             sheath_gaps=[3, 5]
                                             )
        pipeline.run()
        return 0
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
