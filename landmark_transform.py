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


import numpy as np

#%% Optional Slicer imports (graceful degradation if not in Slicer environment)
try:
    import slicer
    import vtk
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
    template_pattern: str  # e.g., "T_{cohort}.nii.gz"
    input_dir: Path
    input_pattern: str  # e.g., "Denoised_*_T1w.nii"
    fids_dir: Path
    fids_pattern: str  # e.g., "fids_{cohort}.fcsv"
    segs_pattern: str  # e.g., "Seg_{cohort}.seg.nrrd"
    cohort: str  # e.g., "Astro02mm"
    output_subdir: str = "MReye"  # subdirectory for outputs
    save_visualization: bool = False
    demo_mode: bool = False
    verbose: bool = False
    keep_scene_open: bool = False  # if False, the scene is cleared after each image

@dataclass
class CenterlineMetrics:
    """Container for centerline properties."""
    region: str
    radius: float = 0.0
    length_mm: float = 0.0
    curvature: float = 0.0
    torsion: float = 0.0
    tortuosity: float = 0.0
    startpoint_x: float = 0.0
    startpoint_y: float = 0.0
    startpoint_z: float = 0.0
    endpoint_x: float = 0.0
    endpoint_y: float = 0.0
    endpoint_z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with region-prefixed keys."""
        return {
            f'{self.region}_radius': self.radius,
            f'{self.region}_length_mm': self.length_mm,
            f'{self.region}_curvature': self.curvature,
            f'{self.region}_torsion': self.torsion,
            f'{self.region}_tortuosity': self.tortuosity,
            f'{self.region}_startpoint_x': self.startpoint_x,
            f'{self.region}_startpoint_y': self.startpoint_y,
            f'{self.region}_startpoint_z': self.startpoint_z,
            f'{self.region}_endpoint_x': self.endpoint_x,
            f'{self.region}_endpoint_y': self.endpoint_y,
            f'{self.region}_endpoint_z': self.endpoint_z,
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

@dataclass
class FeretDiameterMetrics:
    """
    Container for Feret diameter measurements on one cross-section plane.

    The plane is placed *gap_mm* along the centerline from the nerve tip and
    rotated to minimise the intersection area with the sheath (see
    :class:`FeretDiameterExtractor`).
    """
    region: str
    gap_mm: int = 3
    maxferet_mm: float = 0.0
    perpendicularferet_mm: float = 0.0
    curve_area_mm2: float = 0.0
    curve_length_mm: float = 0.0
    valid_diameter: int = 1
    percentage_valid_iterations: float = 0.0
    maxferet_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    maxferet_end: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    perpendicularferet_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    perpendicularferet_end: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with region and gap-prefixed keys."""
        prefix = f'{self.region}_{self.gap_mm}mm'
        metrics = {
            f'{prefix}_maxferetdiameter': self.maxferet_mm,
            f'{prefix}_perpendicularferetdiameter': self.perpendicularferet_mm,
            f'{prefix}_curveareamm2': self.curve_area_mm2,
            f'{prefix}_curvelengthmm': self.curve_length_mm,
            f'{prefix}_ifvaliddiameter': self.valid_diameter,
            f'{prefix}_percentageofvaliditeration': self.percentage_valid_iterations,
        }
        points = {
            ('maxferet', 'Start'): self.maxferet_start,
            ('maxferet', 'End'): self.maxferet_end,
            ('perpendicularferet', 'Start'): self.perpendicularferet_start,
            ('perpendicularferet', 'End'): self.perpendicularferet_end,
        }
        for (which_feret, which_point), position in points.items():
            for axis, value in zip(('R', 'A', 'S'), position):
                metrics[f'{prefix}_{which_feret}_{which_point}PointPosition_{axis}'] = float(value)
        return metrics

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
            startpoint_x = metrics.GetTable().GetColumnByName('StartPointPosition').GetValue(0)
            startpoint_y = metrics.GetTable().GetColumnByName('StartPointPosition').GetValue(1)
            startpoint_z = metrics.GetTable().GetColumnByName('StartPointPosition').GetValue(2)
            endpoint_x = metrics.GetTable().GetColumnByName('EndPointPosition').GetValue(0)
            endpoint_y = metrics.GetTable().GetColumnByName('EndPointPosition').GetValue(1)
            endpoint_z = metrics.GetTable().GetColumnByName('EndPointPosition').GetValue(2)

            return CenterlineMetrics(
                region=region_name,
                radius=raduis,
                length_mm=length,
                curvature=curvature,
                torsion=torsion,
                tortuosity=tortuosity,
                startpoint_x=startpoint_x,
                startpoint_y=startpoint_y,
                startpoint_z=startpoint_z,
                endpoint_x=endpoint_x,
                endpoint_y=endpoint_y,
                endpoint_z=endpoint_z
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
# Feret Diameter Extraction (Slicer 5.6+ compatible)
# ============================================================================

class FeretDiameterExtractor:
    """
    Extract optic nerve sheath Feret diameters on an optimised cross-section plane.

    For each region the plane is placed *gap* mm along the centerline from the
    nerve tip. Because the centerline tangent is a noisy estimate of the true
    sheath axis, the plane is then rotated within a cone around that tangent and
    the orientation with the smallest intersection area is kept - the smallest
    cut of a tube is its perpendicular cut.

    Candidate planes are rejected when they intersect the ``*_diameterplane_check``
    segment (the plane has run into the front of the sheath), when the cut is not
    a single closed contour, or when the resulting area is implausibly small.
    """

    INVALID_AREA = 9999.0

    def __init__(
        self,
        slicerutil_module=None,
        n_iterations: int = 200,
        search_range_deg: float = 30.0,
        min_valid_area_mm2: float = 15.0,
        fast_area_search: bool = True,
        random_seed: Optional[int] = None,
        resample_step_mm: float = 1.0,
    ):
        """
        Args:
            slicerutil_module: Reference to slicerutil (su) module
            n_iterations: Number of candidate plane orientations to try
            search_range_deg: Half-angle of the search cone around the centerline tangent
            min_valid_area_mm2: Cross-sections smaller than this are treated as invalid
            fast_area_search: Score candidates with a polygon area computed directly
                from the cut outline instead of building a markups curve per
                candidate. Same ranking, orders of magnitude faster.
            random_seed: Seed for reproducible plane sampling
            resample_step_mm: Control point spacing of the final closed curve
        """
        self.su = slicerutil_module
        self.n_iterations = n_iterations
        self.search_range = np.deg2rad(search_range_deg)
        self.min_valid_area_mm2 = min_valid_area_mm2
        self.fast_area_search = fast_area_search
        self.resample_step_mm = resample_step_mm
        self.rng = np.random.default_rng(random_seed)

        if not IN_SLICER:
            raise RuntimeError("FeretDiameterExtractor requires 3D Slicer environment.")
        if not self.su:
            raise RuntimeError("slicerutil module required for Feret diameter extraction")

    # ------------------------------------------------------------------
    # Plane search
    # ------------------------------------------------------------------

    def get_cutter_planes(
        self,
        centerline_model_name: str,
        nerve_tip: List[float],
        gap_mm: int,
    ) -> Tuple[Any, Any]:
        """
        Generate candidate plane normals at *gap_mm* from the nerve tip.

        Returns:
            (planenormal_vectors, point) with shapes (N, 3) and (3,)
        """
        centerlinearray = self.su.modeltopointsarray(centerline_model_name)
        dist_to_nervetip = np.atleast_1d(self.su.getDistance(centerlinearray, nerve_tip)) - gap_mm
        point_idx = int(np.argmin(np.abs(dist_to_nervetip)))
        log.debug(f"{centerline_model_name}: centerline point {point_idx} is "
                  f"{dist_to_nervetip[point_idx] + gap_mm:.2f}mm from the nerve tip")

        direction, point = self.su.diameterGetplane(centerline_model_name, point_idx)
        planenormal_vectors = self.su.planeGetnormal(
            direction, self.n_iterations, search_range=self.search_range, rng=self.rng
        )
        return planenormal_vectors, point

    def _candidate_area(
        self,
        seg_node_name: str,
        sheath_segment: str,
        check_segment: Optional[str],
        planenormal: Any,
        point: Any,
    ) -> float:
        """
        Intersection area of one candidate plane, or ``INVALID_AREA`` if rejected.
        """
        plane = self.su.getPlane(np.asarray(planenormal, dtype=float),
                                 np.asarray(point, dtype=float))

        # Reject planes that cut into the front of the sheath.
        if check_segment:
            check_outline = self.su.diameterGetoutlinePolyData(
                seg_node_name, check_segment, plane)
            if check_outline.GetNumberOfPoints() != 0:
                return self.INVALID_AREA

        outline = self.su.diameterGetoutlinePolyData(seg_node_name, sheath_segment, plane)
        if outline.GetNumberOfLines() != 1:
            # Zero lines = plane misses the sheath; more than one = the plane is
            # so oblique it cuts the sheath in several places.
            return self.INVALID_AREA

        if self.fast_area_search:
            area, _ = self.su.polygonAreaAndPerimeter(self.su.polylineFromPolyData(outline))
            return area if area > 0 else self.INVALID_AREA

        # Slow path: reproduce the original measurement exactly, via a markups curve.
        outlinearray = np.array([outline.GetPoint(i) for i in range(outline.GetNumberOfPoints())])
        if len(outlinearray) < 3:
            return self.INVALID_AREA
        closedCurveNode = self.su.getSortedcurvepoints(
            outlinearray[0, :], outlinearray, 'CandidateDiametercurve')
        try:
            self.su.getResampledcurve(closedCurveNode.GetName(), self.resample_step_mm)
            area, _ = self.su.getClosedsurfaceareaandlength(closedCurveNode.GetName())
        finally:
            slicer.mrmlScene.RemoveNode(closedCurveNode)
        return area if area > 0 else self.INVALID_AREA

    def get_optimal_plane(
        self,
        seg_node_name: str,
        sheath_segment: str,
        check_segment: Optional[str],
        centerline_model_name: str,
        nerve_tip: List[float],
        gap_mm: int,
    ) -> Tuple[Any, Any, float]:
        """
        Find the plane orientation with the smallest valid intersection area.

        Returns:
            (optimal_planenormal, point, percentage_of_valid_iterations)
        """
        planenormal_vectors, point = self.get_cutter_planes(
            centerline_model_name, nerve_tip, gap_mm)

        areas = np.array([
            self._candidate_area(seg_node_name, sheath_segment, check_segment,
                                 planenormal_vectors[i, :], point)
            for i in range(len(planenormal_vectors))
        ])

        # Implausibly small cuts mean the plane clipped a corner of the surface.
        areas[areas < self.min_valid_area_mm2] = self.INVALID_AREA
        n_valid = int(np.sum(areas != self.INVALID_AREA))
        percentage_valid = n_valid / len(areas)

        if n_valid == 0:
            log.warning(f"{sheath_segment} at {gap_mm}mm: no valid plane orientation found "
                        f"in {len(areas)} candidates; falling back to the centerline normal")
            return planenormal_vectors[0, :], point, 0.0

        optimal_planenormal = planenormal_vectors[int(np.argmin(areas)), :]
        log.info(f"{sheath_segment} at {gap_mm}mm: min area={np.min(areas):.2f}mm² "
                 f"({n_valid}/{len(areas)} valid orientations)")
        return optimal_planenormal, point, percentage_valid

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def extract_feret_diameters(
        self,
        regions: List[Dict[str, str]],
        centerline_model_prefix: str,
        fiducial_node_name: str,
        seg_node_name: str,
        gaps_mm: List[int] = None,
    ) -> Tuple[Dict[str, List[FeretDiameterMetrics]], Dict[str, Dict[int, Any]]]:
        """
        Measure Feret diameters for all regions at all gaps.

        Args:
            regions: [{'region': 'L_ON', 'segment': 'L_ONS', 'check': 'L_diameterplane_check'}, ...]
            centerline_model_prefix: Suffix of the centerline model names (e.g. 'Centerlinemodel')
            fiducial_node_name: Name of the fiducial list node holding the nerve tips
            seg_node_name: Name of the segmentation node
            gaps_mm: Distances from the nerve tip (default: [3])

        Returns:
            (metrics_by_region, nodes_by_region) where nodes_by_region maps
            region -> gap -> (outlineModel, closedCurveNode, maxferetLine)
        """
        if gaps_mm is None:
            gaps_mm = [3]

        log.info(f"Extracting Feret diameters for {len(regions)} regions at gaps {gaps_mm}mm")

        all_metrics: Dict[str, List[FeretDiameterMetrics]] = {}
        all_nodes: Dict[str, Dict[int, Any]] = {}

        for region_cfg in regions:
            region_name = region_cfg['region']
            side = re.search(r'^(.)', region_name)[0]
            sheath_segment = region_cfg.get('segment', f'{region_name}S')
            check_segment = region_cfg.get('check', f'{side}_diameterplane_check')
            centerline_model_name = f'{region_name}_{centerline_model_prefix}'

            if check_segment and not self.su.segmentationHasSegment(seg_node_name, check_segment):
                log.info(f"Segment '{check_segment}' not in '{seg_node_name}'; "
                         "skipping the diameter-plane validity check")
                check_segment = None

            region_metrics: List[FeretDiameterMetrics] = []
            region_nodes: Dict[int, Any] = {}

            nerve_tip = self.su.getFiducialPosition(fiducial_node_name, f'nerve_tip_{side}')

            for gap_mm in gaps_mm:
                try:
                    metrics, nodes = self._measure_at_gap(
                        seg_node_name, sheath_segment, check_segment,
                        centerline_model_name, nerve_tip, region_name, gap_mm
                    )
                    region_metrics.append(metrics)
                    region_nodes[gap_mm] = nodes
                except Exception as e:
                    log.error(f"Failed Feret diameter for {region_name} at {gap_mm}mm: {e}",
                              exc_info=True)
                    region_metrics.append(FeretDiameterMetrics(
                        region=region_name, gap_mm=gap_mm, valid_diameter=0))

            all_metrics[region_name] = region_metrics
            all_nodes[region_name] = region_nodes

        return all_metrics, all_nodes

    def _measure_at_gap(
        self,
        seg_node_name: str,
        sheath_segment: str,
        check_segment: Optional[str],
        centerline_model_name: str,
        nerve_tip: List[float],
        region_name: str,
        gap_mm: int,
    ) -> Tuple[FeretDiameterMetrics, Tuple[Any, Any, Any]]:
        """Measure one region at one gap and return metrics plus the created nodes."""
        optimal_planenormal, point, percentage_valid = self.get_optimal_plane(
            seg_node_name, sheath_segment, check_segment,
            centerline_model_name, nerve_tip, gap_mm
        )

        outlineModel, closedCurveNode, maxferetLine, perpendicularferetLine = \
            self.su.getClosedCurveModel(
                seg_node_name, sheath_segment, optimal_planenormal, point, gap=gap_mm,
                namePrefix=region_name,
                resample_step=self.resample_step_mm,
            )

        maxferet_points = self.su.markupsLineGetEndpoints(maxferetLine)
        maxferet_mm = float(self.su.getDistance(maxferet_points[0], maxferet_points[1]))

        if perpendicularferetLine is not None:
            perp_points = self.su.markupsLineGetEndpoints(perpendicularferetLine)
            perpendicularferet_mm = float(self.su.getDistance(perp_points[0], perp_points[1]))
        else:
            perp_points = np.full((2, 3), np.nan)
            perpendicularferet_mm = float('nan')

        curveareamm2, curvelengthmm = self.su.getClosedsurfaceareaandlength(
            closedCurveNode.GetName())

        # The chosen plane is valid only if it did not have to fall back.
        valid_diameter = 1 if percentage_valid > 0 else 0

        metrics = FeretDiameterMetrics(
            region=region_name,
            gap_mm=gap_mm,
            maxferet_mm=maxferet_mm,
            perpendicularferet_mm=perpendicularferet_mm,
            curve_area_mm2=float(curveareamm2),
            curve_length_mm=float(curvelengthmm),
            valid_diameter=valid_diameter,
            percentage_valid_iterations=percentage_valid,
            maxferet_start=tuple(maxferet_points[0]),
            maxferet_end=tuple(maxferet_points[1]),
            perpendicularferet_start=tuple(perp_points[0]),
            perpendicularferet_end=tuple(perp_points[1]),
        )
        log.info(f"{region_name} at {gap_mm}mm: maxferet={maxferet_mm:.2f}mm, "
                 f"perpendicular={perpendicularferet_mm:.2f}mm, area={curveareamm2:.2f}mm²")
        return metrics, (outlineModel, closedCurveNode, maxferetLine)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_diameter_nodes(
        self,
        nodes: Dict[str, Dict[int, Any]],
        output_dir: Path,
        subject_id: str,
        fnroot: str,
    ) -> None:
        """Save the outline models and diameter curves for each region and gap."""
        for region_name, gaps in nodes.items():
            for gap_mm, node_tuple in gaps.items():
                outlineModel, closedCurveNode, _ = node_tuple
                try:
                    name_model = f'{region_name}_{gap_mm}mm_Diametermodel'
                    name_curve = f'{region_name}_{gap_mm}mm_Diametercurve'
                    saveNode(outlineModel, str(output_dir / f'{name_model}_{subject_id}_{fnroot}.vtk'))
                    saveNode(closedCurveNode,
                             str(output_dir / f'{name_curve}_{subject_id}_{fnroot}.mrk.json'))
                except Exception as e:
                    log.error(f"Failed to save diameter nodes for {region_name} "
                              f"at {gap_mm}mm: {e}")

    def capture_diameter_views(
        self,
        nodes: Dict[str, Dict[int, Any]],
        output_dir: Path,
        subject_id: str,
        fnroot: str,
        scale_factor: float = 16.0,
    ) -> None:
        """
        Capture the in-plane view of each measured cross-section, plus a
        whole-head view centred on it.
        """
        for region_name, gaps in nodes.items():
            for gap_mm, node_tuple in gaps.items():
                _, _, maxferetLine = node_tuple
                try:
                    self.su.setLayout(8)  # Green slice only, showing the cut plane
                    self.su.zoom2D(scale_factor, zoomRed=False, zoomYellow=False)
                    self.su.captureImageFromAllViews(str(
                        output_dir / f'img_{region_name}_{gap_mm}mm_perpendicularview_'
                                     f'{subject_id}_{fnroot}.png'))
                    self.su.zoom2D(1 / scale_factor, zoomRed=False, zoomYellow=False)

                    endpoints = self.su.markupsLineGetEndpoints(maxferetLine)
                    center = endpoints.mean(axis=0)
                    self.su.setLayout(3)
                    self.su.setSliceOffsetToFiducial(fidArray=center)
                    self.su.captureImageFromAllViews(str(
                        output_dir / f'img_{region_name}_{gap_mm}mm_perpendicularwholeview_'
                                     f'{subject_id}_{fnroot}.png'))
                except Exception as e:
                    log.error(f"Failed to capture diameter view for {region_name} "
                              f"at {gap_mm}mm: {e}")
        self.su.setViewtoDefault('Green')

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

def extract_acq_id(filename: str, pattern: str = r'acq-(.+?)[_.]') -> Optional[str]:
    """Extract acquisition ID from filename using regex pattern (None if absent)."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def build_entity_string(subject_id: str, ses_id: str, acq_id: Optional[str] = None) -> str:
    """
    Build the BIDS entity string used in derivative filenames.

    Returns ``sub-<id>_ses-<id>`` , with ``_acq-<id>`` appended when the input
    filename carries an acquisition entity:
    ``sub-01_ses-20260507_acq-155445``
    """
    entities = f'sub-{subject_id}_ses-{ses_id}'
    if acq_id:
        entities += f'_acq-{acq_id}'
    return entities

def parse_gap_list(values: Any, default: Optional[List[int]] = None) -> List[int]:
    """
    Parse a list of gap distances in mm from user input.

    Accepts any of these spellings, so the same value works from a shell,
    from ``main([...])`` inside Slicer, and from Python::

        parse_gap_list([3, 5, 7])          # native list
        parse_gap_list(['3', '5', '7'])    # --feret-gaps 3 5 7
        parse_gap_list('3,5,7')            # --feret-gaps 3,5,7
        parse_gap_list('[3, 5, 7]')        # --feret-gaps "[3,5,7]"
        parse_gap_list(5)                  # single value

    Duplicates are dropped and the result is sorted ascending.

    Raises:
        ValueError: if a token is not a positive number.
    """
    if values is None or (isinstance(values, (list, tuple)) and not values):
        return list(default) if default else []
    if not isinstance(values, (list, tuple)):
        values = [values]

    tokens: List[str] = []
    for value in values:
        if isinstance(value, (int, float)):
            tokens.append(str(value))
        else:
            # Strip bracket/quote decoration, then split on commas or whitespace
            tokens.extend(re.split(r'[,\s]+', str(value).strip("[](){}'\" ")))

    gaps: List[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            gap = int(float(token))
        except ValueError:
            raise ValueError(f"Invalid gap distance: {token!r} (expected a number in mm)")
        if gap <= 0:
            raise ValueError(f"Gap distance must be positive, got {gap}")
        if gap not in gaps:
            gaps.append(gap)

    if not gaps:
        return list(default) if default else []
    return sorted(gaps)

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
        extract_volume_metrics: bool = True,
        extract_feret_diameter: bool = True,
        sheath_gaps: List[int] = None,
        feret_gaps: List[int] = None,
        feret_iterations: int = 200,
        feret_random_seed: Optional[int] = None,
    ):
        self.config = config
        self.su = slicerutil_module
        self.extract_centerlines = extract_centerlines
        self.extract_sheath_diameter = extract_sheath_diameter
        self.extract_volume_metrics = extract_volume_metrics
        self.extract_feret_diameter = extract_feret_diameter
        # Gaps accept [3, 5, 7], '3,5,7' or '[3,5,7]' - see parse_gap_list()
        self.sheath_gaps = parse_gap_list(sheath_gaps, default=[3, 5])
        self.feret_gaps = parse_gap_list(feret_gaps, default=[3])
        self.centerline_extractor = None
        self.sheath_extractor = None
        self.feret_extractor = None
        setup_logging(config.verbose)

        # Initialize extractors
        if IN_SLICER and self.su:
            if extract_centerlines:
                self.centerline_extractor = CenterlineExtractor(self.su)
                log.info("Initialized CenterlineExtractor")

            if extract_sheath_diameter:
                self.sheath_extractor = SheatDiameterExtractor(self.su)
                log.info("Initialized SheatDiameterExtractor")

            if extract_feret_diameter:
                self.feret_extractor = FeretDiameterExtractor(
                    self.su,
                    n_iterations=feret_iterations,
                    random_seed=feret_random_seed,
                )
                log.info(f"Initialized FeretDiameterExtractor (gaps: "
                         f"{', '.join(f'{g}mm' for g in self.feret_gaps)})")
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
        acq_id = extract_acq_id(mri_file.name)
        if not subject_id:
            log.warning(f"Could not extract subject ID from {mri_file.name}")
            return
        if not ses_id:
            log.warning(f"Could not extract session ID from {mri_file.name}")
            return
        if not acq_id:
            log.info(f"No acquisition entity in {mri_file.name}; "
                     "filenames will be built without acq-")

        # e.g. 'sub-01_ses-20260507_acq-155445'
        entities = build_entity_string(subject_id, ses_id, acq_id)

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
            
            # Load transforms (affine then deformable)
            # e.g. 'trf-Temp_to_sub-01_ses-20260507_acq-155445_DEF.nii.gz'
            fn_aff = f'trf-Temp_to_{entities}_AFF.mat'
            fn_def = f'trf-Temp_to_{entities}_DEF.nii.gz'
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
            fn_fids_out = f'fids_{self.config.cohort}_on_{entities}.fcsv'
            fn_segs_out = f'Segs_{self.config.cohort}_on_{entities}.seg.nrrd'
            
            ff_fids_out = pn_out / fn_fids_out
            ff_segs_out = pn_out / fn_segs_out
            
            SlicerHelper.save_node(n_fids, str(ff_fids_out))
            if n_segs:
                SlicerHelper.save_node(n_segs, str(ff_segs_out))
            
            log.info(f"Saved outputs to {pn_out}")
            
            seg_node_name = 'Segs_{}'.format(self.config.cohort)
            fid_node_name = 'fids_{}'.format(self.config.cohort)

            centerline_metrics_dict = {}
            diameter_metrics_dict = {}
            feret_metrics_dict = {}
            volume_metrics_dict = {}

            if self.extract_centerlines and self.centerline_extractor:
                centerline_metrics_dict = self._extract_centerlines(
                    pn_out, subject_id, fnroot,
                    seg_node_name, fid_node_name,
                    'centerlinemetrics'
                )

            if self.extract_sheath_diameter and self.sheath_extractor:
                diameter_metrics_dict = self._extract_sheath_diameter(
                    pn_out, subject_id, fnroot,
                    seg_node_name, fid_node_name
                )

            # Feret diameters reuse the centerline models created above.
            if self.extract_feret_diameter and self.feret_extractor:
                feret_metrics_dict = self._extract_feret_diameter(
                    pn_out, subject_id, fnroot,
                    seg_node_name, fid_node_name
                )

            if self.extract_volume_metrics and n_segs:
                volume_metrics_dict = self._extract_volume_metrics(
                    pn_out, subject_id, fnroot,
                    n_segs, n_t1
                )

            # Merge and save all metrics
            all_metrics = {
                **centerline_metrics_dict,
                **diameter_metrics_dict,
                **feret_metrics_dict,
                **volume_metrics_dict,
            }
            
            if all_metrics:
                fn_metrics = f'AllMetrics_{subject_id}_{fnroot}.csv'
                ff_metrics = pn_out / fn_metrics
                save_dict_to_csv(all_metrics, ff_metrics)

        except Exception as e:
            log.error(f"Error processing subject {subject_id}: {e}", exc_info=True)
            raise

        finally:
            # Always clear the scene, including after a failure: leftover nodes
            # would otherwise be picked up by name when the next image loads.
            if self.config.keep_scene_open:
                log.info("Keeping the scene open (--keep-scene-open)")
            else:
                log.info(f"Closing the scene after {mri_file.name}")
                SlicerHelper.close_scene()

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
    
    def _extract_volume_metrics(
        self,
        output_dir: Path,
        subject_id: str,
        fnroot: str,
        seg_node: Any,
        volume_node: Any = None,
    ) -> Dict[str, float]:
        """Extract volume/shape metrics from segmentation using SegmentStatistics."""
        log.info(f"Extracting volume metrics for {subject_id}")
        try:
            stats = self.su.segmentationGetsegmentstatistics(seg_node, volume_node)
            metrics_dict = self.su.segmentationGetVolumemetric(stats)

            # Save volume metrics to separate CSV
            fn_csv = f'VolumeMetrics_{subject_id}_{fnroot}.csv'
            ff_csv = output_dir / fn_csv
            save_dict_to_csv(metrics_dict, ff_csv)
            log.info(f"Saved volume metrics: {ff_csv}")

            return metrics_dict
        except Exception as e:
            log.error(f"Failed volume metrics extraction for {subject_id}: {e}")
            return {}

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

    def _extract_feret_diameter(
        self,
        output_dir: Path,
        subject_id: str,
        fnroot: str,
        seg_node_name: str,
        fid_node_name: str,
    ) -> Dict[str, float]:
        """Extract Feret diameters of the sheath on optimised cross-section planes."""
        log.info(f"Extracting Feret diameters for {subject_id}")

        regions_for_feret = [
            {'region': 'L_ON', 'segment': 'L_ONS', 'check': 'L_diameterplane_check'},
            {'region': 'R_ON', 'segment': 'R_ONS', 'check': 'R_diameterplane_check'},
        ]

        all_metrics, all_nodes = self.feret_extractor.extract_feret_diameters(
            regions=regions_for_feret,
            centerline_model_prefix='Centerlinemodel',
            fiducial_node_name=fid_node_name,
            seg_node_name=seg_node_name,
            gaps_mm=self.feret_gaps,
        )

        # Flatten metrics to dictionary
        metrics_dict = {}
        for region_metrics in all_metrics.values():
            for metric in region_metrics:
                metrics_dict.update(metric.to_dict())

        # Save the outline models and diameter curves
        self.feret_extractor.save_diameter_nodes(all_nodes, output_dir, subject_id, fnroot)

        # Save Feret diameters to CSV
        fn_csv = f'FeretdiameterMetrics_{subject_id}_{fnroot}.csv'
        ff_csv = output_dir / fn_csv
        save_dict_to_csv(metrics_dict, ff_csv)
        log.info(f"Saved Feret diameters: {ff_csv}")

        if self.config.save_visualization:
            self.feret_extractor.capture_diameter_views(
                all_nodes, output_dir, subject_id, fnroot)

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
    --output-subdir MReye \\
    --cohort Astro02mm

  # Measure Feret diameters at several distances from the nerve tip
  python landmark_transform.py ... --feret-gaps 3 5 7

  # Run in demo mode (single subject)
  python landmark_transform.py ... --demo

Note: the Slicer scene is cleared after every image; pass --keep-scene-open
to inspect the result of the last one.
        """
    )
    
    parser.add_argument("--template-dir", type=Path, required=True,
                        help="Directory containing template volumes")
    parser.add_argument("--template-pattern", type=str, default="T_{cohort}.nii.gz",
                        help="Filename pattern for templates (default: T_{cohort}.nii.gz)")
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
    parser.add_argument("--output-subdir", type=str, default="MReye",
                        help="Output subdirectory name (default: MReye)")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (single subject, verbose)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--save-visualization", action="store_true",
                        help="Save visualization screenshots")
    parser.add_argument("--keep-scene-open", action="store_true",
                        help="Do not clear the Slicer scene after each image "
                             "(default: the scene is closed after every image)")
    parser.add_argument("--no-feret-diameter", action="store_true",
                        help="Skip Feret diameter extraction")
    parser.add_argument("--feret-gaps", type=str, nargs="+", default=["3"],
                        help="Any number of distances from the nerve tip in mm for the "
                             "Feret diameter planes, e.g. '3 5 7', '3,5,7' or '[3,5,7]' "
                             "(default: 3)")
    parser.add_argument("--sheath-gaps", type=str, nargs="+", default=["3", "5"],
                        help="Any number of distances from the nerve tip in mm for the "
                             "cross-section areas (default: 3 5)")
    parser.add_argument("--feret-iterations", type=int, default=200,
                        help="Candidate plane orientations to search per Feret "
                             "measurement (default: 200)")
    parser.add_argument("--feret-seed", type=int, default=None,
                        help="Random seed for the Feret plane search (reproducible runs)")

    args = parser.parse_args(argv)

    try:
        feret_gaps = parse_gap_list(args.feret_gaps, default=[3])
        sheath_gaps = parse_gap_list(args.sheath_gaps, default=[3, 5])
    except ValueError as e:
        parser.error(str(e))

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
        keep_scene_open=args.keep_scene_open,
    )
    
    try:
        pipeline = LandmarkTransformPipeline(config,
                                             slicerutil_module=su,
                                             extract_centerlines=True,
                                             extract_sheath_diameter=True,
                                             extract_volume_metrics=True,
                                             extract_feret_diameter=not args.no_feret_diameter,
                                             sheath_gaps=sheath_gaps,
                                             feret_gaps=feret_gaps,
                                             feret_iterations=args.feret_iterations,
                                             feret_random_seed=args.feret_seed,
                                             )
        pipeline.run()
        return 0
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
