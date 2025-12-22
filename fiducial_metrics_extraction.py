# -*- coding: utf-8 -*-
"""
Fiducial Metrics Extraction Pipeline

A modular, open-access tool for extracting anatomical measurements
from 3D Slicer fiducial files.

Features:
  - Extract distances and measurements from fiducial landmarks
  - Support for T1 and T2 modalities
  - Cross-platform path handling
  - Configurable via dataclass
  - Comprehensive logging
  - CLI interface

License: MIT
Author: GeTang
"""

#%%
import sys
import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
import re

# ============================================================================
# Configuration & Constants
# ============================================================================

@dataclass
class FiducialMetricsConfig:
    """Configuration for fiducial metrics extraction."""
    project_path: Path = Path(r'/path/to/project')
    cohort: str = 'IIH02mm_T1w'
    mri_pattern: str = 'Denoised_*_{modality}.nii'
    fids_pattern: str = 'fids_{cohort}*.fcsv'
    output_subdir: str = 'SANS'
    output_format: str = 'csv'  # 'csv' or 'xlsx'
    save_results: bool = False
    demo_mode: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        self.output_dir = self.project_path / 'derivatives' / 'Summary'
        self.modality = re.split(r'_', self.cohort)[1]
# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fiducial_metrics_extraction.log', mode='w')
        ]
    )

def ensure_path(path: Path, is_dir: bool = False) -> Path:
    """Ensure path exists, creating directories if needed."""
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def extract_subject_id(filename: str, pattern: str = r'sub-(.+?)_ses') -> Optional[str]:
    """Extract subject ID from filename using regex."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def splitext(fn: str) -> Tuple[str, str]:
    """Split filename into root and extension, handling compound extensions."""
    if '.' not in fn:
        return fn, ''
    root, ext = fn.rsplit('.', 1)
    return root, f'.{ext}'

def locate_files(
    pattern: str, 
    root_dir: Path, 
    level: int = 99, 
    sorted_: bool = True,
) -> List[Path]:
    """
    Locate files matching pattern in directory tree.
    
    Parameters
    ----------
    pattern : str
        File pattern to match (e.g., '*.nii.gz', 'T1*.nii')
    root_dir : Path
        Root directory to start search from
    level : int
        Directory depth level to search
    sorted_ : bool
        Whether to sort results
    exact_level : bool
        If True, only find files at exact level
        If False, find files up to and including level
    
    Returns
    -------
    List[Path]
        List of matching file paths
    """
    import glob
    root_dir = Path(root_dir)
    files = []
    
    if level!= 99:
        # Use '*/' for exact level matching (each '*/' matches exactly one directory)
        pattern_path = root_dir / ('*/' * level) / pattern
        files = glob.glob(str(pattern_path), recursive=False)
    else:
        # Search at all levels up to specified level
        for lvl in range(level + 1):
            pattern_path = root_dir / ('*/' * lvl) / pattern
            files.extend(glob.glob(str(pattern_path), recursive=False))
    
    # Remove duplicates while preserving order
    paths = list(dict.fromkeys(Path(f) for f in files))
    
    if sorted_:
        paths.sort()
    
    return paths

def save_dict_to_csv(
    data: Dict[str, Any], 
    filepath: Path | str
) -> bool:
    """Save dictionary data to CSV."""
    import csv
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Key', 'Value'])
            for key, value in data.items():
                writer.writerow([key, value])
        return True
    except Exception as e:
        logging.error(f"Failed to save CSV {filepath}: {e}")
        return False

# ============================================================================
# Fiducial Processing Functions
# ============================================================================

def read_slicer_annotation_fiducials(filepath: Path) -> pd.DataFrame:
    """
    Read fiducial data from 3D Slicer annotation file.
    
    Args:
        filepath: Path to the .fcsv file
        
    Returns:
        DataFrame with fiducial data
    """
    try:
        fids = pd.read_csv(
            filepath,
            comment='#',
            header=None,
            names=['id','x','y','z','ow','ox','oy','oz','vis','sel',
                   'lock','label','desc','associatedNodeID', 'unknown1', 'unknown2'],
            engine='python'
        )
        return fids
    except Exception as e:
        logging.error(f"Failed to read fiducials from {filepath}: {e}")
        raise

def df_dist(df: pd.DataFrame, pt1: str, pt2: str) -> float:
    """Calculate Euclidean distance between two points in DataFrame."""
    p1 = df.loc[pt1, ['x','y','z']].values.astype(float)
    p2 = df.loc[pt2, ['x','y','z']].values.astype(float)
    return np.linalg.norm(p1 - p2)

def lstsq_plane_estimation(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate least squares plane from 3D points.
    
    Args:
        pts: Nx3 array of points
        
    Returns:
        Tuple of (normal, offset, rotation_matrix)
    """
    # Center points
    pts_centered = pts - np.mean(pts, axis=0)
    
    # SVD for plane fitting
    u, s, vh = np.linalg.svd(pts_centered[:, :3].T, full_matrices=True)
    
    normal = u[:, -1]
    # Ensure normal points towards positive z
    if normal[-1] < 0:
        normal *= -1.0
    
    R = u.copy()  # u is orthonormal
    offset = np.mean(pts[:, :3], axis=0)
    
    return normal, offset, R

def fid_measures_t1(df: pd.DataFrame, with_eye_orb_dist: bool = True) -> Dict[str, float]:
    """
    Extract fiducial measurements for T1 modality.
    
    Args:
        df: DataFrame with fiducial data (indexed by label)
        with_eye_orb_dist: Whether to include orbital distance calculations
        
    Returns:
        Dictionary of measurements
    """
    d = {}
    for side in ['L', 'R']:
        # Basic distances
        d[f'd1_{side}'] = df_dist(df, f'center_{side}_lens', f'center_{side}_eyeball')
        d[f'd2_{side}'] = df_dist(df, f'center_{side}_eyeball', f'nerve_tip_{side}')
        d[f'd3_{side}'] = df_dist(df, f'center_{side}_lens', f'eyeball_back_{side}')
        d[f'w1_{side}'] = df_dist(df, f'eyeball_midline_{side}_lat', f'eyeball_midline_{side}_med')
        d[f'w2_{side}'] = df_dist(df, f'nerve_baseline_muscle_{side}_lat', f'nerve_baseline_muscle_{side}_med')
        d[f'w3_{side}'] = df_dist(df, f'nerve_baseline_bone_{side}_lat', f'nerve_baseline_bone_{side}_med')
        d[f'h1_{side}'] = df_dist(df, f'optcanal_height_{side}_inf', f'optcanal_height_{side}_sup')
        d[f'w4_{side}'] = df_dist(df, f'optcanal_width_{side}_lat', f'optcanal_width_{side}_med')
        
        # Orbital rim distances
        if with_eye_orb_dist:
            pts_orb = df.loc[[
                f'orbital_rim_{side}_lat',
                f'orbital_rim_{side}_med',
                f'orbital_rim_{side}_sup',
                f'orbital_rim_{side}_inf'
            ], ['x','y','z']].values.astype(float)
            
            normal, offset, R = lstsq_plane_estimation(pts_orb)
            pts_orb_mean = np.mean(pts_orb, axis=0)
            
            # Eye center to plane
            pts_eye_ctr = df.loc[f'center_{side}_eyeball', ['x','y','z']].values.astype(float)
            d[f'd4_{side}'] = np.dot(normal, pts_eye_ctr - pts_orb_mean)
            
            # Lens center to plane
            pts_lens_ctr = df.loc[f'center_{side}_lens', ['x','y','z']].values.astype(float)
            d[f'd5_{side}'] = np.dot(normal, pts_lens_ctr - pts_orb_mean)
    
    return d

def fid_measures_t2(df: pd.DataFrame, with_eye_orb_dist: bool = True) -> Dict[str, float]:
    """
    Extract fiducial measurements for T2 modality.
    
    Args:
        df: DataFrame with fiducial data (indexed by label)
        with_eye_orb_dist: Whether to include orbital distance calculations
        
    Returns:
        Dictionary of measurements
    """
    d = {}
    for side in ['L', 'R']:
        # Basic distances
        d[f'd1_{side}'] = df_dist(df, f'center_{side}_lens', f'center_{side}_eyeball')
        d[f'd2_{side}'] = df_dist(df, f'center_{side}_eyeball', f'nerve_tip_{side}')
        d[f'd3_{side}'] = df_dist(df, f'center_{side}_lens', f'eyeball_back_{side}')
        d[f'w1_{side}'] = df_dist(df, f'eyeball_midline_{side}_lat', f'eyeball_midline_{side}_med')
        d[f'w2_{side}'] = df_dist(df, f'nerve_baseline_muscle_{side}_lat', f'nerve_baseline_muscle_{side}_med')
        d[f'w3_{side}'] = df_dist(df, f'nerve_baseline_bone_{side}_lat', f'nerve_baseline_bone_{side}_med')
        d[f'n1_{side}'] = df_dist(df, f'nerve_width_{side}_lat', f'nerve_width_{side}_med')
        d[f'h1_{side}'] = df_dist(df, f'optcanal_height_{side}_inf', f'optcanal_height_{side}_sup')
        d[f'w4_{side}'] = df_dist(df, f'optcanal_width_{side}_lat', f'optcanal_width_{side}_med')
        
        # Orbital rim distances
        if with_eye_orb_dist:
            pts_orb = df.loc[[
                f'orbital_rim_{side}_lat',
                f'orbital_rim_{side}_med',
                f'orbital_rim_{side}_sup',
                f'orbital_rim_{side}_inf'
            ], ['x','y','z']].values.astype(float)
            
            normal, offset, R = lstsq_plane_estimation(pts_orb)
            pts_orb_mean = np.mean(pts_orb, axis=0)
            
            # Eye center to plane
            pts_eye_ctr = df.loc[f'center_{side}_eyeball', ['x','y','z']].values.astype(float)
            d[f'd4_{side}'] = np.dot(normal, pts_eye_ctr - pts_orb_mean)
            
            # Lens center to plane
            pts_lens_ctr = df.loc[f'center_{side}_lens', ['x','y','z']].values.astype(float)
            d[f'd5_{side}'] = np.dot(normal, pts_lens_ctr - pts_orb_mean)
    
    return d

# ============================================================================
# Main Pipeline Class
# ============================================================================

class FiducialMetricsExtractionPipeline:
    """Main pipeline for extracting fiducial metrics."""
    
    def __init__(self, config: FiducialMetricsConfig):
        self.config = config
        setup_logging(config.verbose)
    
    def run(self):
        """Run the fiducial metrics extraction pipeline."""
        logging.info("Starting Fiducial Metrics Extraction Pipeline")
        logging.info(f"Config: {asdict(self.config)}")
        
        try:
            # Locate input files
            input_files_df = self._locate_input_files()
            
            if input_files_df.empty:
                logging.warning("No input files found")
                return
            
            # Extract metrics
            metrics_list = self._extract_metrics(input_files_df)
            
            # Create results DataFrame
            results_df = pd.DataFrame.from_records(metrics_list)
            
            # Save results
            if self.config.save_results:
                self._save_results(results_df)
            
            logging.info(f"Processed {len(metrics_list)} subjects")
        
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        
        logging.info("Fiducial Metrics Extraction Pipeline completed")
    
    def _locate_input_files(self) -> pd.DataFrame:
        """Locate and organize input files."""
        logging.info("Locating input files...")
        
        all_files = []
        print(self.config.modality)
        mri_pattern = self.config.mri_pattern.format(modality=self.config.modality)
        print(f"Using MRI pattern: {mri_pattern}")
        # Locate MRI files
        mri_files = locate_files(
            mri_pattern, 
            self.config.project_path, 
            level=3, 
            sorted_=True
        )
            
        for mri_file in mri_files:
            print(f"Found MRI file: {mri_file}")
            # Extract subject info from filename
            fn_splits = mri_file.name.split('_')
            if len(fn_splits) >= 3:
                subject_id = f"{fn_splits[1]}_{fn_splits[2]}"
                subject = fn_splits[1]
                session = fn_splits[2]
                
                all_files.append({
                    'ff': str(mri_file),
                    'fn': mri_file.name,
                    'fn_root': mri_file.stem,
                    'id': subject_id,
                    'sub': subject,
                    'ses': session,
                })
        return pd.DataFrame(all_files)
    
    def _extract_metrics(self, files_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract metrics from fiducial files."""
        logging.info("Extracting metrics from fiducials...")
        
        metrics_list = []
        
        for idx, row in files_df.iterrows():
            logging.info(f"Processing {idx+1}/{len(files_df)}: {row['fn']}")
            print(row)
            
            try:
                # Construct fiducial file path
                root_dir = Path(row['ff']).parents[3]

                # Get the relative path from root to the anat directory
                subject_session_anat = Path(row['ff']).relative_to(root_dir).parent
                
                # Construct sans_dir: root/derivatives/subject/session/anat/SANS
                metrics_dir = root_dir / 'derivatives' / subject_session_anat / 'SANS'

                fids_pattern = self.config.fids_pattern.format(cohort=self.config.cohort)
                fids_files = locate_files(fids_pattern, metrics_dir, level=0)

                if not fids_files:
                    logging.warning(f"No fiducial file found for {row['fn']}")
                    # Create dummy metrics
                    metrics = self._create_dummy_metrics(row)
                else:
                    fids_file = fids_files[0]
                    df_fids = read_slicer_annotation_fiducials(fids_file)
                    print(df_fids.head())
                    df_fids = df_fids.set_index('label')
                    
                    # Extract measurements based on modality
                    if self.config.modality == 'T1w':
                        metrics = fid_measures_t1(df_fids)
                    elif self.config.modality == 'T2w':
                        metrics = fid_measures_t2(df_fids)
                    else:
                        raise ValueError(f"Unknown modality: {row['modality']}")
                
                # Add metadata
                metrics.update({
                    'fn_root': row['fn_root'][9:],  # Remove prefix
                    'id': row['id'],
                    'sub': row['sub'],
                    'ses': row['ses']
                })
                
                metrics_list.append(metrics)
            
            except Exception as e:
                logging.error(f"Failed to process {row['fn']}: {e}")
                # Add dummy metrics on failure
                metrics_list.append(self._create_dummy_metrics(row))
        
        return metrics_list
    
    def _create_dummy_metrics(self, row: pd.Series) -> Dict[str, Any]:
        """Create dummy metrics for missing data."""
        dummy_metrics = {
            'd1_L': np.nan, 'd1_R': np.nan,
            'd2_L': np.nan, 'd2_R': np.nan,
            'd3_L': np.nan, 'd3_R': np.nan,
            'd4_L': np.nan, 'd4_R': np.nan,
            'd5_L': np.nan, 'd5_R': np.nan,
            'n1_L': np.nan, 'n1_R': np.nan,
            'w1_L': np.nan, 'w1_R': np.nan,
            'w2_L': np.nan, 'w2_R': np.nan,
            'w3_L': np.nan, 'w3_R': np.nan,
            'w4_L': np.nan, 'w4_R': np.nan,
            'h1_L': np.nan, 'h1_R': np.nan,
            'fn_root': row['fn_root'][9:],
            'id': row['id'],
            'sub': row['sub'],
            'ses': row['ses']
        }
        return dummy_metrics
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save results to file."""
        ensure_path(self.config.output_dir, is_dir=True)
        
        output_filename = f'allFidDistances'
        
        if self.config.output_format == 'csv':
            output_path = self.config.output_dir / f'{output_filename}.csv'
            results_df.to_csv(output_path, index=False)
        elif self.config.output_format == 'xlsx':
            output_path = self.config.output_dir / f'{output_filename}.xlsx'
            results_df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
        
        logging.info(f"Saved results to {output_path}")

# ============================================================================
# CLI Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None):
    """Command-line interface for Fiducial Metrics Extraction Pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fiducial Metrics Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python fiducial_metrics_extraction.py \\
    --project-path /data/IIH \\
    --modalities IIH02mm T1 IIH02mm T2

  # Custom output
  python fiducial_metrics_extraction.py \\
    --project-path /data/IIH \\
    --output-format xlsx \\
    --verbose
        """
    )
    
    parser.add_argument("--project-path", type=Path, required=True,
                        help="Project root directory")
    parser.add_argument("--mri-pattern", type=str, default="Denoised_*_{modality}.nii",
                        help="MRI filename pattern (default: Denoised_*_{modality}.nii)")
    parser.add_argument("--fids-pattern", type=str, default="fids_{cohort}_*.fcsv",
                        help="Fiducial filename pattern (default: fids_{cohort}_*.fcsv)")
    parser.add_argument("--output-subdir", type=str, default="SANS",
                        help="Output subdirectory name (default: SANS)")
    parser.add_argument("--output-format", type=str, default="csv", choices=["csv", "xlsx"],
                        help="Output format (default: csv)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to file")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (first modality only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args(argv)
    
    # Configurations
    config = FiducialMetricsConfig(
        project_path=args.project_path,
        mri_pattern=args.mri_pattern,
        fids_pattern=args.fids_pattern,
        output_subdir=args.output_subdir,
        output_format=args.output_format,
        save_results=True,
        demo_mode=args.demo,
        verbose=args.verbose,
    )
    
    try:
        pipeline = FiducialMetricsExtractionPipeline(config)
        pipeline.run()
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
# %%
