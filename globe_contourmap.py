# -*- coding: utf-8 -*-
"""
Globe Shape Analysis Pipeline for 3D Slicer

A modular, open-access tool for analyzing eyeball shape projections
from template space to subject space.

Features:
  - Orthogonal and polar projection analysis
  - Cross-platform path handling
  - Configurable via dataclass
  - Lazy Slicer integration
  - Comprehensive logging
  - Demo mode for testing

License: MIT
Author: GeTang
"""

#%%
import sys
import os
import re
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import numpy as np
import math
from math import sqrt, cos, sin, acos
import random

# ============================================================================
# Optional Slicer imports (graceful degradation if not in Slicer environment)
# ============================================================================

try:
    import slicer
    from slicer.util import loadMarkupsFiducialList, loadSegmentation
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    IN_SLICER = True
except ImportError:
    IN_SLICER = False
    # Mock classes for non-Slicer environments
    class MockVTK:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    vtk = MockVTK()

# ============================================================================
# Configuration & Constants
# ============================================================================

@dataclass
class GlobeShapeAnalysisConfig:
    """Configuration for globe shape analysis."""
    cohort: str  # e.g., "IIH02mmT1"
    # File patterns
    fids_pattern: str  # e.g., "fids_{cohort}.fcsv"
    segs_pattern: str  # e.g., "Seg_{cohort}.seg.nrrd"
    mri_pattern: str  # e.g., "Denoised_*_T1w.nii"
    project_path: Path = Path(r'D:\users\getang\SANS')
    orthogonal_projection: bool = False
    scaling: int = 50  # Scaling for finding intersection of lines on eyeball plane
    degree_to_center: int = 90
    sampling_points: int = int(1e4)
    save_results: bool = False
    demo: bool = True
    verbose: bool = False

# ============================================================================
# Utility Functions (shared with landmark_transform.py)
# ============================================================================

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('globe_shape_analysis.log', mode='w')
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
    """Extract subject ID from filename using regex pattern."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def extract_session_id(filename: str, pattern: str = r'ses-(.+?)_') -> Optional[str]:
    """Extract session ID from filename using regex pattern."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def splitext(fn: str) -> Tuple[str, str]:
    """Split filename into root and extension, handling compound extensions."""
    if '.' not in fn:
        return fn, ''
    root, ext = fn.rsplit('.', 1)
    return root, f'.{ext}'

def locate_files(pattern: str, root_dir: Path, level: int = 3, sorted_: bool = True) -> List[Path]:
    """Locate files matching pattern in directory tree."""
    import glob
    files = []
    for i in range(level + 1):
        pattern_path = root_dir / ('**/' * i) / pattern
        files.extend(glob.glob(str(pattern_path), recursive=True))
    
    paths = [Path(f) for f in files]
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
# Projection Analysis Classes
# ============================================================================

@dataclass
class ProjectionResult:
    """Container for projection analysis results."""
    subject_id: str
    session_id: str
    eyeball_name: str
    projection_type: str  # 'orthogonal' or 'polar'
    center: np.ndarray
    nerve_tip: np.ndarray
    num_points: int = 0
    degree_threshold: int = 90
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'eyeball_name': self.eyeball_name,
            'projection_type': self.projection_type,
            'center_x': self.center[0][0],
            'center_y': self.center[0][1],
            'center_z': self.center[0][2],
            'nerve_tip_x': self.nerve_tip[0][0],
            'nerve_tip_y': self.nerve_tip[0][1],
            'nerve_tip_z': self.nerve_tip[0][2],
            'num_points': self.num_points,
            'degree_threshold': self.degree_threshold,
        }

class ProjectionAnalyzer:
    """Analyze eyeball shape projections."""
    
    def __init__(self, config: GlobeShapeAnalysisConfig, slicerutil_module=None):
        self.config = config
        self.su = slicerutil_module
        if not IN_SLICER:
            raise RuntimeError("ProjectionAnalyzer requires 3D Slicer environment.")
        if not self.su:
            logging.warning("slicerutil module not provided; some functions may fail")
    
    def analyze_orthogonal_projection(
        self,
        pd: Any,  # vtkPolyData
        center: np.ndarray,
        nerve_tip: np.ndarray,
        eyeball_name: str,
        subject_id: str,
        session_id: str
    ) -> ProjectionResult:
        """Perform orthogonal projection analysis."""
        logging.info(f"Analyzing orthogonal projection for {eyeball_name}")
        
        try:
            # Get all points on surface
            points = np.array(pd.GetPoints().GetData())
            moving_vector = points - center
            
            # Z-axis in polar coordinate system
            axis_vector = nerve_tip - center
            polar_angles = self.su.getAngle(moving_vector, axis_vector, radians=False)
            
            # Add polar angles and lengths to info
            points_info = np.column_stack([
                points,
                polar_angles,
                self.su.getLength(moving_vector)
            ])
            
            # Filter points within degree threshold
            point_list = points_info[:, 3] < self.config.degree_to_center
            points_info = points_info[point_list]
            
            # Create projection plane
            plane = self.su.getPlane(axis_vector, center)
            eye_plane_xy = self.su.getOrCreateMarkupsPlaneNode(
                'eye_plane', center.reshape(1, 3), axis_vector.reshape(3, 1)
            )
            
            # Project points onto plane
            planecenter = center.reshape(3, 1)
            planenormal = (axis_vector / np.linalg.norm(axis_vector)).reshape(3, 1)
            pointsonplane = np.zeros((len(points_info), 3))
            
            for i in range(len(points_info)):
                point = points_info[i, :3]
                planeprojection = np.zeros((3, 1))
                plane.ProjectPoint(point.reshape(3, 1), planecenter, planenormal, planeprojection)
                pointsonplane[i] = planeprojection.reshape(1, 3)
            
            # Get coordinate system transformation
            ref_plane_xy = self.su.getOrCreateMarkupsPlaneNode(
                'reference_plane', np.array([0, 0, 0]).reshape(1, 3), np.array([0, 0, 1]).reshape(3, 1)
            )
            
            point_on_line, direction_x = self._intersection_line(ref_plane_xy, eye_plane_xy)
            
            # Adjust for left/right eye
            if eyeball_name.split('_')[0] == 'L':
                direction_x = direction_x  # Keep as is
            
            # Get axis intersections
            axis1_line = self._get_intersection_points(
                pd, direction_x * self.config.scaling + center, 
                -direction_x * self.config.scaling + center
            )
            
            orthogonal_vector = np.cross(direction_x, eye_plane_xy.GetNormal())
            axis2_line = self._get_intersection_points(
                pd, center + orthogonal_vector * 50, center - orthogonal_vector * 50
            )
            
            axis_points = np.vstack([axis1_line, axis2_line])
            
            # Coordinate transformation
            center_B = center
            x_axis_B = direction_x
            y_axis_B = -orthogonal_vector
            z_axis_B = eye_plane_xy.GetNormal()
            
            # Normalize basis vectors
            x_axis_B_norm = self._normalize(x_axis_B)
            y_axis_B_norm = self._normalize(y_axis_B)
            z_axis_B_norm = self._normalize(z_axis_B)
            
            R_A_to_B = np.column_stack([x_axis_B_norm, y_axis_B_norm, z_axis_B_norm])
            R_B_to_A = R_A_to_B.T
            
            pointsonplane_offset = pointsonplane[:, :3] - center
            rotated = np.dot(R_B_to_A, pointsonplane_offset.T).T
            rotated_xyaxis = np.dot(R_B_to_A, (axis_points - center).T).T
            
            rotated_info = np.column_stack([
                rotated[:, :2],
                self.su.getLength(moving_vector[point_list])
            ])
            
            # Save results if configured
            if self.config.save_results:
                projection_name = f'Orthogonal_projection_{eyeball_name}_{self.config.degree_to_center}_sub-{subject_id}_ses-{session_id}'
                output_path = self.config.project_path / 'derivatives' / f'sub-{subject_id}' / f'ses-{session_id}' / 'anat' / 'SANS' / f'{projection_name}.pickle'
                ensure_path(output_path.parent, is_dir=True)
                self.su.save_pickle(rotated_info, str(output_path))
            
            return ProjectionResult(
                subject_id=subject_id,
                session_id=session_id,
                eyeball_name=eyeball_name,
                projection_type='orthogonal',
                center=center,
                nerve_tip=nerve_tip,
                num_points=len(rotated_info),
                degree_threshold=self.config.degree_to_center
            )
        
        except Exception as e:
            logging.error(f"Failed orthogonal projection for {eyeball_name}: {e}", exc_info=True)
            raise
    
    def analyze_polar_projection(
        self,
        pd: Any,  # vtkPolyData
        center: np.ndarray,
        nerve_tip: np.ndarray,
        eyeball_name: str,
        subject_id: str,
        session_id: str
    ) -> ProjectionResult:
        """Perform polar projection analysis."""
        logging.info(f"Analyzing polar projection for {eyeball_name}")
        
        try:
            # Get all points on surface
            points = np.array(pd.GetPoints().GetData())
            moving_vector = points - center
            
            # Z-axis in polar coordinate system
            axis_vector = nerve_tip - center
            polar_angles = self.su.getAngle(moving_vector, axis_vector, radians=False)
            
            # Create projection planes
            plane = self.su.getPlane(axis_vector, center)
            eye_plane_xy = self.su.getPlane(axis_vector, center)
            ref_plane_xy = self.su.getPlane(np.array([0, 0, 1]).reshape(3, 1), np.array([0, 0, 0]).reshape(1, 3))
            
            point_on_line, direction_x = self._intersection_line(ref_plane_xy, eye_plane_xy)
            
            # Adjust for left/right eye
            if eyeball_name.split('_')[0] == 'L':
                direction_x = direction_x
            
            # Project points onto plane
            planecenter = center.reshape(3, 1)
            planenormal = (axis_vector / np.linalg.norm(axis_vector)).reshape(3, 1)
            pointsonplane = np.zeros((len(points), 3))
            
            for i in range(len(points)):
                point = points[i]
                planeprojection = np.zeros((3, 1))
                plane.ProjectPoint(point.reshape(3, 1), planecenter, planenormal, planeprojection)
                pointsonplane[i] = planeprojection.reshape(1, 3)
            
            projected_moving_vector = pointsonplane - center
            azimuthal_angles = self._angles_between_vectors(projected_moving_vector, direction_x.reshape(1, 3))
            
            # Create points info matrix
            points_info = np.column_stack([
                points,
                azimuthal_angles,
                polar_angles,
                self.su.getLength(moving_vector)
            ])
            
            # Select south hemisphere points
            south_point_list = points_info[:, 4] < self.config.degree_to_center
            points_info = points_info[south_point_list]
            
            # Convert to cartesian
            if eyeball_name.split('_')[0] == 'L':
                cartesian_coord = self._spherical_to_cartesian(
                    1, points_info[:, 3], points_info[:, 4], 90, points_info[:, 5]
                )
            else:
                cartesian_coord = self._spherical_to_cartesian(
                    1, points_info[:, 3], points_info[:, 4], 90, points_info[:, 5]
                )
            
            # Save results if configured
            if self.config.save_results:
                projection_name = f'Polar_projection_{eyeball_name}_{self.config.degree_to_center}_sub-{subject_id}_ses-{session_id}'
                output_path = self.config.project_path / 'derivatives' / f'sub-{subject_id}' / f'ses-{session_id}' / 'anat' / 'SANS' / f'{projection_name}.vtp'
                ensure_path(output_path.parent, is_dir=True)
                self._save_vtk_points(cartesian_coord, str(output_path))
            
            return ProjectionResult(
                subject_id=subject_id,
                session_id=session_id,
                eyeball_name=eyeball_name,
                projection_type='polar',
                center=center,
                nerve_tip=nerve_tip,
                num_points=cartesian_coord.GetNumberOfPoints(),
                degree_threshold=self.config.degree_to_center
            )
        
        except Exception as e:
            logging.error(f"Failed polar projection for {eyeball_name}: {e}", exc_info=True)
            raise
    
    # Utility methods (converted from original functions)
    def _intersection_line(self, plane1, plane2):
        """Calculate intersection line of two planes."""
        normal1 = np.array(plane1.GetNormal())
        normal2 = np.array(plane2.GetNormal())
        direction = np.cross(normal1, normal2)
        if np.linalg.norm(direction) == 0:
            return None, None
        
        origin1 = np.array(plane1.GetOrigin())
        origin2 = np.array(plane2.GetOrigin())
        A = np.vstack((normal1, normal2, direction))
        b = np.array([np.dot(normal1, origin1), np.dot(normal2, origin2), 0])
        
        try:
            point_on_line = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None, None
        
        return point_on_line, direction
    
    def _get_intersection_points(self, pd, start_point, end_point):
        """Get intersection points of line with polydata."""
        start_point = start_point.reshape(3, 1)
        end_point = end_point.reshape(3, 1)
        obb_tree = vtk.vtkOBBTree()
        obb_tree.SetDataSet(pd)
        obb_tree.BuildLocator()
        
        intersection_points = vtk.vtkPoints()
        obb_tree.IntersectWithLine(start_point, end_point, intersection_points, None)
        intersection_points_array = vtk_to_numpy(intersection_points.GetData())
        return intersection_points_array.reshape(-1, 3)
    
    def _normalize(self, v):
        """Normalize vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
    
    def _angles_between_vectors(self, matrix_a, vector_b):
        """Calculate angles between vectors."""
        angles = np.zeros(len(matrix_a))
        for i in range(matrix_a.shape[0]):
            a = matrix_a[i]
            b = vector_b[0]
            dot_prod = np.dot(a, b)
            mag_a = np.linalg.norm(a)
            mag_b = np.linalg.norm(b)
            cos_theta = dot_prod / (mag_a * mag_b)
            theta_rad = math.acos(np.clip(cos_theta, -1, 1))
            theta_deg = math.degrees(theta_rad)
            
            cross_prod = np.cross(a, b)
            if cross_prod[2] < 0:
                theta_deg = 360 - theta_deg
            angles[i] = theta_deg
        
        return angles.reshape(-1, 1)
    
    def _spherical_to_cartesian(self, radial, azimuthal_array, polar_array, max_polar_angle, real_radius, clockwise=True):
        """Convert spherical to cartesian coordinates."""
        vtk_points = vtk.vtkPoints()
        for azimuthal, polar, radius in zip(azimuthal_array, polar_array, real_radius):
            r = radial * (polar / max_polar_angle)
            if not clockwise:
                azimuthal = -azimuthal
            x = r * np.cos(np.radians(azimuthal))
            y = -r * np.sin(np.radians(azimuthal))  # Flip for consistency
            z = radius
            vtk_points.InsertNextPoint(x, y, z)
        return vtk_points
    
    def _save_vtk_points(self, vtk_points, filename):
        """Save VTK points to file."""
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(poly_data)
        writer.Write()

# ============================================================================
# Main Pipeline Class
# ============================================================================

class GlobeShapeAnalysisPipeline:
    """Main pipeline for globe shape analysis."""
    
    def __init__(self, config: GlobeShapeAnalysisConfig, slicerutil_module=None):
        self.config = config
        self.su = slicerutil_module
        setup_logging(config.verbose)
        
        if IN_SLICER and self.su:
            self.analyzer = ProjectionAnalyzer(config, self.su)
            logging.info("Initialized ProjectionAnalyzer")
        else:
            logging.warning("Running without Slicer environment - limited functionality")
    
    def run(self):
        """Run the globe shape analysis pipeline."""
        logging.info("Starting Globe Shape Analysis Pipeline")
        logging.info(f"Config: {asdict(self.config)}")
        
        # Locate input files
        input_files = locate_files(
            self.config.mri_pattern, 
            self.config.project_path, 
            level=3, 
            sorted_=True
        )
        
        if not input_files:
            logging.error(f"No input files found matching {self.config.mri_pattern}")
            return
        
        if self.config.demo:
            input_files = input_files[:1]
            logging.info("Demo: processing only first subject")
        
        results = []
        
        for idx, mri_file in enumerate(input_files):
            logging.info(f"[{idx+1}/{len(input_files)}] Processing: {mri_file.name}")
            
            try:
                subject_id = extract_subject_id(mri_file.name)
                if not subject_id:
                    logging.warning(f"Could not extract subject ID from {mri_file.name}")
                    continue
                
                session_id = extract_session_id(mri_file.name)
                if not session_id:
                    logging.warning(f"Could not extract session ID from {mri_file.name}")
                    continue

                subject_results = self._process_subject(mri_file, subject_id, session_id)
                results.extend(subject_results)
                
            except Exception as e:
                logging.error(f"Failed to process {mri_file.name}: {e}", exc_info=True)
                continue
            
            if not self.config.demo:
                self.su.closeScene()
        
        # Save summary results
        if results:
            summary_data = {}
            for result in results:
                result_dict = result.to_dict()
                for key, value in result_dict.items():
                    summary_data[f"{result.subject_id}_{result.eyeball_name}_{key}"] = value
            
            summary_path = self.config.project_path / 'derivatives' / 'globe_shape_analysis_summary.csv'
            save_dict_to_csv(summary_data, summary_path)
            logging.info(f"Saved summary to {summary_path}")
        
        logging.info("Globe Shape Analysis Pipeline completed")
    
    def _process_subject(self, mri_file: Path, subject_id: str, session_id: str) -> List[ProjectionResult]:
        """Process a single subject."""
        # Get the root directory (e.g., D:/IIH)
        root_dir = mri_file.parents[3]  # Adjust index based on your structure
        
        # Get the relative path from root to the anat directory
        subject_session_anat = mri_file.relative_to(root_dir).parent
        
        # Construct sans_dir: root/derivatives/subject/session/anat/SANS
        sans_dir = root_dir / 'derivatives' / subject_session_anat / 'SANS'
        results = []
        
        try:
            # Load fiducials
            fid_files = locate_files(self.config.fids_pattern.format(cohort=self.config.cohort), sans_dir, level=0)
            if len(fid_files) != 1:
                raise ValueError(f"Expected 1 fiducial file, found {len(fid_files)}: {fid_files}")
            nFIDS = loadMarkupsFiducialList(str(fid_files[0]), returnNode=True)
            nFIDS.SetName(f'fids_{self.config.cohort}')
            
            # Load segmentations
            seg_files = locate_files(self.config.segs_pattern.format(cohort=self.config.cohort), sans_dir, level=0)
            if len(seg_files) != 1:
                raise ValueError(f"Expected 1 segmentation file, found {len(seg_files)}: {seg_files}")
            
            success, nSEGS = loadSegmentation(str(seg_files[0]), returnNode=True)
            if not success:
                raise RuntimeError("Failed to load segmentation")
            nSEGS.SetName(f'Segs_{self.config.cohort}')
            
            # Process eyeballs
            eyeballs = ['L_eyeball', 'R_eyeball'] if not self.config.demo else ['L_eyeball']
            
            for eyeball_name in eyeballs:
                logging.info(f"Processing {eyeball_name} for {subject_id} {session_id}")
                
                # Get segmentation data
                n = self.su.getNode(f'Segs_{self.config.cohort}')
                s = n.GetSegmentation()
                ss = s.GetSegment(s.GetSegmentIdBySegmentName(eyeball_name))
                pd = ss.GetRepresentation('Closed surface')
                
                # Get nerve tip
                nerve_tip = self.su.arrayFromFiducialList(
                    f'fids_{self.config.cohort}', 
                    [f'nerve_tip_{eyeball_name.split("_")[0]}']
                )
                
                # Get center
                center = np.array(pd.GetCenter()).reshape(1, 3)
                
                # Perform projection analysis
                if self.config.orthogonal_projection:
                    result = self.analyzer.analyze_orthogonal_projection(
                        pd, center, nerve_tip, eyeball_name, subject_id, session_id
                    )
                else:
                    result = self.analyzer.analyze_polar_projection(
                        pd, center, nerve_tip, eyeball_name, subject_id, session_id
                    )
                
                results.append(result)
                
        except Exception as e:
            logging.error(f"Error processing subject {subject_id} {session_id}: {e}", exc_info=True)
        
        return results

# ============================================================================
# CLI Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None):
    """Command-line interface for Globe Shape Analysis Pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Globe Shape Analysis Pipeline for 3D Slicer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings
  python Globe_contourmap.py --project-path /data/SANS

  # Run orthogonal projection with custom parameters
  python Globe_contourmap.py \\
    --project-path /data/SANS \\
    --orthogonal-projection \\
    --scaling 100 \\
    --degree-to-center 120 \\
    --save-results

  # Run in demo mode (single subject, verbose)
  python Globe_contourmap.py --project-path /data/SANS --demo
        """
    )
    parser.add_argument("--cohort", type=str, required=True,
                        help="Cohort name (e.g., Astro02mm, Cosmo02mm)")
    parser.add_argument("--fids-pattern", type=str, default="fids_{cohort}.fcsv",
                        help="Filename pattern for fiducials (default: fids_{cohort}_*.fcsv)")
    parser.add_argument("--segs-pattern", type=str, default="Segs_{cohort}_*.seg.nrrd",
                        help="Filename pattern for segmentations (default: Segs_{cohort}_*.seg.nrrd)")
    parser.add_argument("--mri-pattern", type=str, default="Denoised_*_T1w.nii",
                        help="Filename pattern for MRI files (default: Denoised_*_T1w.nii)")
    parser.add_argument("--project-path", type=Path, required=True,
                        help="Project root directory containing data")
    parser.add_argument("--orthogonal-projection", action="store_true",
                        help="Use orthogonal projection instead of polar")
    parser.add_argument("--scaling", type=int, default=50,
                        help="Scaling for finding intersection of lines (default: 50)")
    parser.add_argument("--degree-to-center", type=int, default=90,
                        help="Degree threshold to center (default: 90)")
    parser.add_argument("--sampling-points", type=int, default=int(1e4),
                        help="Number of sampling points (default: 10000)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save projection results to files")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (single subject, verbose)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    
    args = parser.parse_args(argv)
    
    if not IN_SLICER:
        print("ERROR: This script must be run in 3D Slicer's Python environment.")
        print("In Slicer, go to: Python Interactor → Execute script")
        return 1
    
    config = GlobeShapeAnalysisConfig(
        cohort=args.cohort,
        fids_pattern=args.fids_pattern,
        segs_pattern=args.segs_pattern,
        mri_pattern=args.mri_pattern,
        project_path=args.project_path,
        orthogonal_projection=args.orthogonal_projection,
        scaling=args.scaling,
        degree_to_center=args.degree_to_center,
        sampling_points=args.sampling_points,
        save_results=args.save_results,
        demo=args.demo,
        verbose=args.verbose or args.demo
    )
    
    try:
        pipeline = GlobeShapeAnalysisPipeline(config, slicerutil_module=su)
        pipeline.run()
        return 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
