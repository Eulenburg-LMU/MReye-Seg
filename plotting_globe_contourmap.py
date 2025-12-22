# -*- coding: utf-8 -*-
"""
Globe Shape Analysis Plotting Pipeline

A modular, open-access tool for plotting eyeball shape projections
from globe shape analysis results.

Features:
  - 2D contour plotting for polar/orthogonal projections
  - Support for mean and variance analysis
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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import random
import pickle

# ============================================================================
# Optional VTK imports (graceful degradation if not available)
# ============================================================================

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    # Mock VTK classes for non-VTK environments
    class MockVTK:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    vtk = MockVTK()

# ============================================================================
# Configuration & Constants
# ============================================================================

@dataclass
class GlobeShapePlottingConfig:
    """Configuration for globe shape analysis plotting."""
    project_path: Path = Path(r'/Users/getang/Documents/EarthResearch/IIH')
    save_data: bool = False
    plotting: bool = True
    projection_map: str = 'Polar'  # 'Polar' or 'Orthogonal'
    degree_to_center: int = 90
    side_of_eye: List[str] = field(default_factory=lambda: ['L', 'R'])
    modality: str = 'T1'  # 'T1' or 'T2'
    condition: str = 'control'  # 'control' or 'variance'
    contour_level: int = 200
    radii: List[float] = field(default_factory=lambda: [])  # Auto-calculated
    x_offset: float = 0.02
    y_offset: float = 0.02
    fontsize: int = 18
    verbose: bool = False
    demo_mode: bool = False  # Process only subset for testing
    
    def __post_init__(self):
        # Auto-calculate radii based on degree_to_center
        if not self.radii:
            self.radii = [
                (3/270) * self.degree_to_center,
                (2/270) * self.degree_to_center,
                (1/270) * self.degree_to_center
            ]

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
            logging.FileHandler('plotting_globe_shape_analysis.log', mode='w')
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
    import re
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
# Data Loading Functions
# ============================================================================

def open_pickle(filename: str) -> Any:
    """Load data from pickle file."""
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

def load_vtk_points(filename: str) -> Any:
    """Load VTK points from file."""
    if not HAS_VTK:
        raise ImportError("VTK is required for loading VTK files")
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    poly_data = reader.GetOutput()
    return poly_data.GetPoints()

def vtk_points_to_numpy_array(vtk_points: Any) -> np.ndarray:
    """Convert VTK points to numpy array."""
    num_points = vtk_points.GetNumberOfPoints()
    data = np.zeros((num_points, 3))
    for i in range(num_points):
        point = vtk_points.GetPoint(i)
        data[i, 0] = point[0]
        data[i, 1] = point[1]
        data[i, 2] = point[2]
    return data

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_2d_contour(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    config: GlobeShapePlottingConfig,
    plotname: Optional[str] = None,
    levels: int = 1000,
    cmap: str = 'coolwarm'
) -> None:
    """
    Plot 2D contour of globe shape data.
    
    Args:
        grid_x, grid_y, grid_z: Grid data for plotting
        config: Plotting configuration
        plotname: Plot title/filename
        levels: Number of contour levels
        cmap: Colormap name
    """
    if grid_z.ndim == 2:
        # Single plot
        fig, ax = plt.subplots()
        norm = plt.Normalize(vmin=np.nanmin(grid_z), vmax=np.nanmax(grid_z))
        ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, norm=norm)
        ax.plot(0, 0, marker=".", markersize=10, markeredgecolor="red")

        if config.projection_map == 'Polar':
            for radius in config.radii:
                circle = plt.Circle((0, 0), radius=radius, fill=False, color='black', lw=1)
                ax.add_artist(circle)
                ax.text(radius - config.x_offset, 0,
                       f'{int(90*radius)}°', ha='right', va='center',
                       fontsize=config.fontsize)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Colorbar')

        title = plotname or '2D Contour Plot of back of eyeball'
        ax.set_title(title)
        
        if plotname and 'L_' in plotname:
            ax.set_xlabel('Temporal to Nasal')
        else:
            ax.set_xlabel('Nasal to Temporal')
        ax.set_ylabel('Inferior to Superior')
        
        if plotname:
            output_path = config.project_path / 'derivatives' / f'{plotname}.pdf'
            ensure_path(output_path.parent, is_dir=True)
            plt.savefig(str(output_path), format='pdf')
        
        plt.show()

    elif grid_z.ndim == 3:
        # Side-by-side plots
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

        ax1 = plt.subplot(gs[0], aspect='equal')
        ax2 = plt.subplot(gs[1], aspect='equal')

        vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
        norm = plt.Normalize(vmin, vmax)

        # Plot first dataset
        ax1.contourf(grid_x, grid_y, grid_z[:, :, 0], levels=levels, cmap=cmap, norm=norm)
        ax1.plot(0, 0, marker=".", markersize=8, markeredgecolor="red")
        ax1.set_title(plotname[0][0] if plotname else 'Dataset 1')
        ax1.set_xlabel('Temporal to Nasal')
        ax1.set_ylabel('Inferior to Superior')

        # Plot second dataset
        ax2.contourf(grid_x, grid_y, grid_z[:, :, 1], levels=levels, cmap=cmap, norm=norm)
        ax2.plot(0, 0, marker=".", markersize=8, markeredgecolor="red")
        ax2.set_title(plotname[1][0] if plotname else 'Dataset 2')
        ax2.set_xlabel('Nasal to Temporal')

        if config.projection_map == 'Polar':
            for ax in [ax1, ax2]:
                for radius in config.radii:
                    circle = Circle((0, 0), radius=radius, fill=False, color='black', lw=1)
                    ax.add_patch(circle)
                    ax.text(radius - config.x_offset, 0,
                           f'{int(90*radius)}°', ha='right', va='center',
                           fontsize=config.fontsize)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = plt.subplot(gs[2])
        step = 0.05
        ticks = np.arange(np.floor(vmin / step) * step,
                         np.ceil(vmax / step) * step + step, step)
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', ticks=ticks)
        cbar.set_label('Colorbar')
        
        if plotname:
            output_path = config.project_path / 'derivatives' / f'{plotname[0][0][2:]}.pdf'
            ensure_path(output_path.parent, is_dir=True)
            plt.savefig(str(output_path), format='pdf')
        
        plt.show()

# ============================================================================
# Main Pipeline Class
# ============================================================================

class GlobeShapePlottingPipeline:
    """Main pipeline for plotting globe shape analysis results."""
    
    def __init__(self, config: GlobeShapePlottingConfig):
        self.config = config
        setup_logging(config.verbose)
        
        # Check dependencies
        if config.projection_map == 'Polar' and not HAS_VTK:
            logging.warning("VTK not available - Polar projection loading may fail")
    
    def run(self):
        """Run the plotting pipeline."""
        logging.info("Starting Globe Shape Analysis Plotting Pipeline")
        logging.info(f"Config: {asdict(self.config)}")
        
        try:
            if self.config.save_data:
                self._save_projection_data()
            
            if self.config.plotting:
                self._create_plots()
        
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        
        logging.info("Globe Shape Analysis Plotting Pipeline completed")
    
    def _save_projection_data(self):
        """Save projection data to numpy files."""
        logging.info("Saving projection data...")
        
        data_dir = self.config.project_path / 'derivatives'
        
        pattern = f'{self.config.projection_map}_projection_*.vtp'
        files = locate_files(pattern, data_dir, level=4)
        
        if not files:
            logging.warning(f"No projection files found matching {pattern}")
            return
        
        for idx, file_path in enumerate(files):
            logging.info(f"Processing {file_path.name}")
            print(file_path)
            
            try:
                if self.config.projection_map == 'Orthogonal':
                    data = open_pickle(str(file_path))
                elif self.config.projection_map == 'Polar':
                    data = vtk_points_to_numpy_array(load_vtk_points(str(file_path)))
                else:
                    raise ValueError(f"Unknown projection_map: {self.config.projection_map}")
                
                # Create grid and interpolate
                x, y, z = data[:, 0], data[:, 1], data[:, 2]
                X, Y = np.mgrid[min(x):max(x):1000j, min(y):max(y):1000j]
                Z = griddata((x, y), z, (X, Y), method='linear')
                Z = np.expand_dims(Z, axis=-1)
                
                if idx == 0:
                    all_metrics = Z
                    X_expanded = np.expand_dims(X, axis=-1)
                    Y_expanded = np.expand_dims(Y, axis=-1)
                    grids = np.concatenate((X_expanded, Y_expanded), axis=2)
                else:
                    all_metrics = np.concatenate((all_metrics, Z), axis=2)
            
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Save files
        summary_file = data_dir / f'{self.config.projection_map}_projection_eyeball_{self.config.degree_to_center}_summary.npy'
        grid_file = data_dir / f'{self.config.projection_map}_grid_eyeball_{self.config.degree_to_center}_summary.npy'
        
        np.save(str(summary_file), all_metrics)
        np.save(str(grid_file), grids)
        logging.info(f"Saved projection data to {summary_file}")
    
    def _create_plots(self):
        """Create and save plots."""
        logging.info("Creating plots...")
        
        data_dir = self.config.project_path / 'derivatives'
        info_file = data_dir / 'Polar_projection_info.csv'
        
        if not info_file.exists():
            logging.error(f"Info file not found: {info_file}")
            return
        
        df_info = pd.read_csv(str(info_file))
        
        if len(self.config.side_of_eye) == 1:
            self._plot_single_eye(df_info, data_dir)
        elif set(self.config.side_of_eye) == {'L', 'R'}:
            self._plot_both_eyes(df_info, data_dir)
        else:
            logging.error("Invalid side_of_eye configuration")
    
    def _plot_single_eye(self, df_info: pd.DataFrame, data_dir: Path):
        """Plot data for a single eye."""
        summary_file = data_dir / f'{self.config.projection_map}_projection_eyeball_{self.config.degree_to_center}_summary.npy'
        grid_file = data_dir / f'{self.config.projection_map}_grid_eyeball_{self.config.degree_to_center}_summary.npy'
        
        if not summary_file.exists() or not grid_file.exists():
            logging.error("Summary or grid files not found")
            return
        
        metrics = np.load(str(summary_file))
        grids = np.load(str(grid_file))
        X, Y = grids[:, :, 0], grids[:, :, 1]
        
        # Filter data
        mask = (
            (df_info['group'] == self.config.condition) &
            (df_info['side_of_eyeball'] == self.config.side_of_eye[0]) &
            (df_info['modality'] == self.config.modality)
        )
        indices = df_info[mask].index.values
        
        if len(indices) == 0:
            logging.warning("No data found for specified filters")
            return
        
        metrics_filtered = metrics[:, :, indices]
        
        if self.config.condition == 'control':
            Z = np.mean(metrics_filtered, axis=2)
            cmap = cm.hot.reversed()
        elif self.config.condition != 'control':
            Z = np.var(metrics_filtered, axis=2)
            cmap = cm.hot.reversed()
        else:
            logging.error(f"Unknown condition: {self.config.condition}")
            return
        
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        plotname = f'{self.config.side_of_eye[0]}_back_of_eyeball_{self.config.projection_map}_projection_{self.config.modality}w_for_{self.config.condition}'
        
        plot_2d_contour(X, Y, Z, self.config, plotname=plotname,
                       levels=self.config.contour_level, cmap=cmap)
    
    def _plot_both_eyes(self, df_info: pd.DataFrame, data_dir: Path):
        """Plot data for both eyes side-by-side."""
        logging.info("Plotting both eyes")
        
        summary_file = data_dir / f'{self.config.projection_map}_projection_eyeball_{self.config.degree_to_center}_summary.npy'
        grid_file = data_dir / f'{self.config.projection_map}_grid_eyeball_{self.config.degree_to_center}_summary.npy'
        
        if not summary_file.exists() or not grid_file.exists():
            logging.error("Summary or grid files not found")
            return
        
        base_metrics = np.expand_dims(np.load(str(summary_file)), axis=-1)
        grids = np.load(str(grid_file))
        X, Y = grids[:, :, 0], grids[:, :, 1]
        
        plotnames = []
        all_metrics = None
        
        for eye in self.config.side_of_eye:
            mask = (
                (df_info['group'] == self.config.condition) &
                (df_info['side_of_eyeball'] == eye) &
                (df_info['modality'] == self.config.modality)
            )
            indices = df_info[mask].index.values
            
            if len(indices) == 0:
                logging.warning(f"No data found for {eye}")
                continue
            
            metrics_filtered = base_metrics[:, :, indices, :]
            
            if all_metrics is None:
                all_metrics = metrics_filtered
            else:
                all_metrics = np.concatenate((all_metrics, metrics_filtered), axis=3)
            
            plotnames.append([f'{eye}_back_of_eyeball_{self.config.projection_map}_projection_{self.config.modality}w_for_{self.config.condition}'])
        
        if all_metrics is None:
            logging.error("No valid data found for plotting")
            return
        
        if self.config.condition == 'control':
            Z = np.mean(all_metrics, axis=2)
            cmap = cm.hot.reversed()
        elif self.config.condition != 'control':
            Z = np.var(all_metrics, axis=2)
            cmap = cm.hot.reversed()
        else:
            logging.error(f"Unknown condition: {self.config.condition}")
            return
        
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        plot_2d_contour(X, Y, Z, self.config, plotname=plotnames,
                       levels=self.config.contour_level, cmap=cmap)

# ============================================================================
# CLI Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None):
    """Command-line interface for Globe Shape Analysis Plotting Pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Globe Shape Analysis Plotting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot control data for left eye
  python plotting_globe_shape_analysis.py \\
    --project-path /data/IIH \\
    --side-of-eye L \\
    --condition control

  # Save data and plot variance for both eyes
  python plotting_globe_shape_analysis.py \\
    --project-path /data/IIH \\
    --save-data \\
    --side-of-eye L R \\
    --condition variance \\
    --verbose
        """
    )
    
    parser.add_argument("--project-path", type=Path, required=True,
                        help="Project root directory")
    parser.add_argument("--save-data", action="store_true",
                        help="Save projection data to numpy files")
    parser.add_argument("--projection-map", type=str, default="Polar",
                        choices=["Polar", "Orthogonal"],
                        help="Projection type (default: Polar)")
    parser.add_argument("--degree-to-center", type=int, default=90,
                        help="Degree threshold to center (default: 90)")
    parser.add_argument("--side-of-eye", nargs='+', default=["L", "R"],
                        choices=["L", "R"],
                        help="Eye sides to plot (default: L R)")
    parser.add_argument("--modality", type=str, default="T1",
                        choices=["T1", "T2"],
                        help="MRI modality (default: T1)")
    parser.add_argument("--condition", type=str, default="control",
                        help="Analysis condition (default: control)")
    parser.add_argument("--contour-level", type=int, default=200,
                        help="Number of contour levels (default: 200)")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (limited processing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args(argv)
    
    config = GlobeShapePlottingConfig(
        project_path=args.project_path,
        save_data=args.save_data,
        plotting=True,  # Always plot unless specified otherwise
        projection_map=args.projection_map,
        degree_to_center=args.degree_to_center,
        side_of_eye=args.side_of_eye,
        modality=args.modality,
        condition=args.condition,
        contour_level=args.contour_level,
        verbose=args.verbose,
        demo_mode=args.demo,
    )
    
    try:
        pipeline = GlobeShapePlottingPipeline(config)
        pipeline.run()
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())