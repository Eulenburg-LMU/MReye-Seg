# -*- coding: utf-8 -*-
"""
MReye-Seg Slicer Utilities
===========================

Project-specific 3D Slicer utility functions for the MReye-Seg pipeline.
Provides wrappers for centerline extraction, cross-section analysis,
visualization, geometry, and fiducial/markup operations.

Used by:
  - landmark_transform.py  (centerline, cross-section, visualization)
  - globe_contourmap.py    (geometry, markups, serialization)

Requires: 3D Slicer >= 5.6 with VMTK / ExtractCenterline extension.

License: MIT
Author: GeTang
"""

import re
import pickle
import logging

import numpy as np

log = logging.getLogger(__name__)

# ============================================================================
# Optional Slicer imports (graceful degradation outside 3D Slicer)
# ============================================================================

try:
    import slicer
    import vtk
    import ctk
    from vtk.util import numpy_support
    IN_SLICER = True
except ImportError:
    IN_SLICER = False
    log.warning("Not running in 3D Slicer; Slicer-specific functions will be unavailable.")


# ============================================================================
# Node Access & Creation
# ============================================================================

def getNode(name):
    """Get a Slicer MRML node by name, returning None if not found."""
    try:
        node = slicer.util.getNode(name)
    except slicer.util.MRMLNodeNotFoundException:
        node = None
    return node


def getOrCreateNode(name, createFn=lambda n: None, performOnNode=lambda x: None, addToScene=True):
    """Get an existing node or create a new one."""
    node = getNode(name)
    if node is None:
        node = createFn(name)
        node.SetName(name)
        performOnNode(node)
        if addToScene:
            slicer.mrmlScene.AddNode(node)
    return node


def getOrCreateFiducialListNode(name, performOnNode=lambda x: None):
    """Get or create a markups fiducial node."""
    return getOrCreateNode(
        name,
        createFn=lambda n: slicer.vtkMRMLMarkupsFiducialNode(),
        performOnNode=performOnNode,
    )


# ============================================================================
# Markup & Plane Utilities
# ============================================================================

def getOrCreateMarkupsPlaneNode(namePlaneNode, centerpoint, planenormal):
    """
    Create a markups plane node in the Slicer scene.

    Parameters
    ----------
    namePlaneNode : str
        Name of the plane node.
    centerpoint : np.ndarray
        Center point (3-element, any shape).
    planenormal : np.ndarray
        Normal vector (3-element, any shape).

    Returns
    -------
    vtkMRMLMarkupsPlaneNode
    """
    if centerpoint.shape != (3,):
        centerpoint = centerpoint.reshape(3, 1)
    if planenormal.shape != (3,):
        planenormal = planenormal.reshape(3, 1)

    planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", namePlaneNode)
    planeNode.SetNormal(planenormal)
    planeNode.SetOrigin(centerpoint)

    planeNode.SetSizeMode(1)  # Absolute mode
    planeNode.SetPlaneBounds(np.array([-200, 200, 200, -200, 200, -200]))
    visFid_SetVisibility(
        planeNode.GetName(), visibility=0.4, locked=1, textScale=0,
        color=(255, 255, 0), glyph_scale=0,
    )
    return planeNode


def visFid_SetVisibility(fids, visibility=None, locked=None, textScale=None,
                         color=None, glyph_scale=None, line_thickness=None,
                         line_opacity=None):
    """Set display properties on a markups node."""
    if isinstance(fids, str):
        nfid = getNode(fids)
    else:
        nfid = fids
    dn = nfid.GetDisplayNode()

    if visibility is not None:
        for i in range(nfid.GetNumberOfControlPoints()):
            nfid.SetNthControlPointVisibility(i, visibility)
    if locked is not None:
        for i in range(nfid.GetNumberOfControlPoints()):
            nfid.SetNthControlPointLocked(i, locked)
    if textScale is not None:
        dn.SetPropertiesLabelVisibility(False)
        dn.SetTextScale(textScale)
    if color is not None:
        dn.SetSelectedColor(color)
    if glyph_scale is not None:
        dn.SetGlyphScale(glyph_scale)
    if line_thickness is not None:
        dn.SetLineThickness(line_thickness)
    if line_opacity is not None:
        dn.SetVisibility3D(1)
        dn.SetOpacity(line_opacity)


def labelIndexInFidList(nameFidList, label):
    """Return the index of a control point by label, or -1 if not found."""
    fl = getNode(nameFidList)
    if fl is not None:
        labels = [fl.GetNthControlPointLabel(i) for i in range(fl.GetNumberOfControlPoints())]
        idxs = np.where(np.array(labels) == label)
        if idxs[0].shape[0] > 0:
            return int(idxs[0][0])
    return -1


def getFiducialPosition(nameFidList, nameFid):
    """Get the world position of a fiducial by label name."""
    fids = getNode(nameFidList)
    if fids is None:
        log.warning("Fiducial list '%s' does not exist.", nameFidList)
        return [0.0, 0.0, 0.0]

    idx = labelIndexInFidList(nameFidList, nameFid)
    if idx == -1:
        log.warning("No fiducial '%s' in list '%s'", nameFid, nameFidList)
        return [0.0, 0.0, 0.0]

    pos = [0.0, 0.0, 0.0]
    fids.GetNthControlPointPositionWorld(idx, pos)
    return pos[0:3]


def arrayFromFiducialList(nameFidList, listFidNames=None):
    """
    Extract 3D coordinates from a fiducial list node.

    Parameters
    ----------
    nameFidList : str
        Name of the markups fiducial node.
    listFidNames : list[str] or None
        If given, only extract these labels. Otherwise extract all.

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)`` array of RAS coordinates.
    """
    fids = getOrCreateFiducialListNode(nameFidList)
    if listFidNames is None:
        n = fids.GetNumberOfControlPoints()
        P = []
        for i in range(n):
            pos = [0.0, 0.0, 0.0]
            fids.GetNthControlPointPositionWorld(i, pos)
            P.append(pos)
    else:
        P = [getFiducialPosition(nameFidList, name) for name in listFidNames]
    return np.array(P)


# ============================================================================
# Visualization & Layout
# ============================================================================

def setLayout(layoutid):
    """
    Set the Slicer layout.

    Parameters
    ----------
    layoutid : int
        SlicerLayoutFourUpView = 3, SlicerLayoutOneUp3DView = 4,
        SlicerLayoutOneUpRedSliceView = 6, etc.
    """
    slicer.app.layoutManager().setLayout(layoutid)


def zoom3D(factor):
    """Zoom the 3D view by *factor*."""
    camera = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCameraNode').GetCamera()
    camera.Zoom(factor)


def view3D_lookFromViewAxis(direction='right'):
    """Set the 3D view orientation by axis name."""
    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    axis_map = {
        'right':     ctk.ctkAxesWidget.Right,
        'left':      ctk.ctkAxesWidget.Left,
        'inferior':  ctk.ctkAxesWidget.Inferior,
        'superior':  ctk.ctkAxesWidget.Superior,
        'anterior':  ctk.ctkAxesWidget.Anterior,
        'posterior':  ctk.ctkAxesWidget.Posterior,
    }
    axis = axis_map.get(direction.lower())
    if axis is not None:
        view.lookFromViewAxis(axis)


def view3D_center():
    """Reset the 3D view focal point to center on visible content."""
    threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
    threeDView.resetFocalPoint()


def segmentationSetVisualization(nameSeg, opacity2D=None, opacity3D=None,
                                 show2DFill=None, show2DOutline=None,
                                 visibility3D=None):
    """Set display properties on a segmentation node."""
    nseg = getNode(nameSeg)
    nsegdisp = nseg.GetDisplayNode()
    nsegdisp.SetAllSegmentsOpacity3D(0.5)
    if show2DFill is not None:
        nsegdisp.SetAllSegmentsVisibility2DFill(show2DFill)
    if show2DOutline is not None:
        nsegdisp.SetAllSegmentsVisibility2DOutline(show2DOutline)
    if opacity2D is not None:
        nsegdisp.SetAllSegmentsOpacity(opacity2D)
    if opacity3D is not None:
        nsegdisp.SetOpacity3D(opacity3D)
    if visibility3D is not None:
        nsegdisp.SetVisibility3D(visibility3D)


def captureImageFromAllViews(filename):
    """Capture a screenshot of all Slicer views and save to *filename*."""
    import ScreenCapture
    cap = ScreenCapture.ScreenCaptureLogic()
    cap.showViewControllers(False)
    cap.captureImageFromView(None, filename)
    cap.showViewControllers(True)


# ============================================================================
# Centerline Extraction (requires VMTK / ExtractCenterline extension)
# ============================================================================

def getPreprocessedsurface(nameSeg, nameRegion, targetpoints=5000, decimation=4):
    """Pre-process a segmentation surface for centerline extraction."""
    import ExtractCenterline
    ec = ExtractCenterline.ExtractCenterlineLogic()
    n = getNode(nameSeg)
    s = n.GetSegmentation()
    ss = s.GetSegment(s.GetSegmentIdBySegmentName(nameRegion)).GetRepresentation('Closed surface')
    preprocessedPolyData = ec.preprocess(ss, targetpoints, decimation, False)
    return preprocessedPolyData


def centerlineExtractcenterline(nameSeg, nameRegion, nameFids, preprocesssurface=False):
    """
    Extract a centerline from a segmented region using VMTK.

    Parameters
    ----------
    nameSeg : str
        Name of the segmentation node.
    nameRegion : str
        Segment name within the segmentation.
    nameFids : str
        Name of the fiducial node with start/end points.
    preprocesssurface : bool
        Whether to downsample the surface before extraction.

    Returns
    -------
    tuple
        ``(centerlinePolyData, voronoiDiagramPolyData)``
    """
    import ExtractCenterline
    ec = ExtractCenterline.ExtractCenterlineLogic()
    n = getNode(nameSeg)
    s = n.GetSegmentation()
    ss = s.GetSegment(s.GetSegmentIdBySegmentName(nameRegion)).GetRepresentation('Closed surface')
    fids = getNode(nameFids)

    if preprocesssurface:
        preprocessedPolyData = getPreprocessedsurface(nameSeg, nameRegion)
        centerlinePolyData, voronoiDiagramPolyData = ec.extractCenterline(preprocessedPolyData, fids)
    else:
        centerlinePolyData, voronoiDiagramPolyData = ec.extractCenterline(ss, fids)
    return centerlinePolyData, voronoiDiagramPolyData


def centerlineGetcenterlineproperties(centerlinePolyData, nameTable=None):
    """
    Compute centerline properties (length, curvature, torsion, etc.).

    Returns
    -------
    vtkMRMLTableNode
        Slicer table node with computed metrics.
    """
    import ExtractCenterline
    ec = ExtractCenterline.ExtractCenterlineLogic()
    centerlinepropertiesTableNode = slicer.vtkMRMLTableNode()
    centerlinepropertiesTableNode.SetName(nameTable)
    ec.createCurveTreeFromCenterline(
        centerlinePolyData,
        centerlineCurveNode=None,
        centerlinePropertiesTableNode=centerlinepropertiesTableNode,
    )
    slicer.mrmlScene.AddNode(centerlinepropertiesTableNode)
    return centerlinepropertiesTableNode


def centerlineGetmodel(centerlinePolyData, opacity=1, nameModel=None):
    """
    Create a Slicer model node from centerline polydata.

    Returns
    -------
    vtkMRMLModelNode
    """
    model = slicer.vtkMRMLModelNode()
    model.SetAndObservePolyData(centerlinePolyData)
    model.SetName(nameModel)
    modelDisplay = slicer.vtkMRMLModelDisplayNode()
    modelDisplay.SetColor(1, 0, 0)
    modelDisplay.BackfaceCullingOff()
    modelDisplay.SetOpacity(opacity)
    modelDisplay.SetPointSize(3)
    modelDisplay.SetVisibility2D(True)
    modelDisplay.SetVisibility(True)
    slicer.mrmlScene.AddNode(modelDisplay)
    model.SetAndObserveDisplayNodeID(modelDisplay.GetID())
    modelDisplay.SetInputPolyDataConnection(model.GetPolyDataConnection())
    slicer.mrmlScene.AddNode(model)
    return model


# ============================================================================
# Cross-Section Analysis
# ============================================================================

def getDistance(a, b, metric='euclidean'):
    """
    Compute pairwise distance between point sets.

    Parameters
    ----------
    a, b : array_like
        Points in 1D, 2D, or 3D.
    metric : str
        ``'euclidean'`` or ``'sqeuclidean'``.

    Returns
    -------
    np.ndarray
        Distance array.
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr


def findPointIndexByDistance(tableNode, targetDistance, nerve_tip):
    """Find the centerline point index closest to *targetDistance* from *nerve_tip*."""
    table = tableNode.GetTable()
    numRows = table.GetNumberOfRows()

    coordinateRASCol = -1
    for i in range(table.GetNumberOfColumns()):
        if "ras" in table.GetColumnName(i).lower():
            coordinateRASCol = i
            break
    if coordinateRASCol == -1:
        log.error("Coordinate column not found in table")
        return None

    distances = []
    for row in range(numRows):
        point = table.GetValue(row, coordinateRASCol).ToArray().GetTuple(0)
        distance = getDistance(point, nerve_tip)
        distances.append(distance)
    distances = np.array(distances)

    closestIndex = np.argmin(np.abs(distances - targetDistance))
    return closestIndex


def interpolateCrossSectionAreaAtDistance(tableNode, targetDistance, nerve_tip):
    """Interpolate cross-sectional area at an exact distance along the centerline."""
    table = tableNode.GetTable()
    numRows = table.GetNumberOfRows()

    coordinateRASCol = -1
    areaCol = -1
    for i in range(table.GetNumberOfColumns()):
        colName = table.GetColumnName(i).lower()
        if "ras" in colName:
            coordinateRASCol = i
        elif "cross-section area" in colName or "area" in colName:
            areaCol = i
    if coordinateRASCol == -1 or areaCol == -1:
        log.error("Could not find distance or area columns")
        return None

    distances = []
    areas = []
    for row in range(numRows):
        point = table.GetValue(row, coordinateRASCol).ToArray().GetTuple(0)
        area = table.GetValue(row, areaCol).ToDouble()
        distance = getDistance(point, nerve_tip)
        distances.append(distance)
        areas.append(area)
    distances = np.array(distances)
    areas = np.array(areas)

    if targetDistance < distances.min() or targetDistance > distances.max():
        log.warning(
            "Target distance %.2f mm outside range [%.2f, %.2f] mm",
            targetDistance, distances.min(), distances.max(),
        )
        return None

    if targetDistance in distances:
        index = np.where(distances == targetDistance)[0][0]
        interpolatedArea = areas[index]
        log.debug("Exact match found at point %d", index)
    else:
        sortedIndices = np.argsort(distances)
        sortedDistances = distances[sortedIndices]
        sortedAreas = areas[sortedIndices]
        insertIndex = np.searchsorted(sortedDistances, targetDistance)
        if insertIndex == 0:
            interpolatedArea = sortedAreas[0]
        elif insertIndex == len(sortedDistances):
            interpolatedArea = sortedAreas[-1]
        else:
            d1, d2 = sortedDistances[insertIndex - 1], sortedDistances[insertIndex]
            a1, a2 = sortedAreas[insertIndex - 1], sortedAreas[insertIndex]
            weight = (targetDistance - d1) / (d2 - d1)
            interpolatedArea = a1 + weight * (a2 - a1)
    return interpolatedArea


def GetCrossSectionArea(nameCenterlineModel, nameFids, nameSeg, nameRegion,
                        gap=3, interpolate=True):
    """
    Compute cross-section area of a segment at a given distance from the nerve tip.

    Uses the CrossSectionAnalysis Slicer module.

    Parameters
    ----------
    nameCenterlineModel : str
        Name of the centerline model node.
    nameFids : str
        Name of the fiducial node (nerve tip reference).
    nameSeg : str
        Name of the segmentation node.
    nameRegion : str
        Name of the segment.
    gap : int
        Distance (mm) along centerline from nerve tip.
    interpolate : bool
        Whether to interpolate the area at the exact distance.

    Returns
    -------
    float
        Cross-section area in mm².
    """
    import CrossSectionAnalysis
    cs = CrossSectionAnalysis.CrossSectionAnalysisLogic()
    centerline = getNode(nameCenterlineModel)
    segmentation = getNode(nameSeg)
    segmentation.CreateClosedSurfaceRepresentation()
    segmentID = segmentation.GetSegmentation().GetSegmentIdBySegmentName(nameRegion)

    outputTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    outputTableNode.SetName(f'{nameRegion}_CrossSectionResults')

    nerve_tip = getFiducialPosition(nameFids, 'nerve_tip_%s' % (re.search('^(.)', nameRegion)[0]))
    cs.setInputCenterlineNode(centerline)
    cs.setLumenSurface(segmentation, segmentID)
    cs.setOutputTableNode(outputTableNode)
    cs.run()

    if interpolate:
        area = interpolateCrossSectionAreaAtDistance(outputTableNode, gap, nerve_tip)
    else:
        idx_point = findPointIndexByDistance(outputTableNode, gap, nerve_tip)
        area = cs.getCrossSectionArea(idx_point)
        crossSectionPolyData = cs.computeLumenCrossSectionPolydata(idx_point)
        model = slicer.modules.models.logic().AddModel(crossSectionPolyData)
        model.SetName(f'{nameRegion}_{gap}mmCrossSection')
    return area


# ============================================================================
# Geometry Utilities
# ============================================================================

def getAngle(vector1, vector2, radians=True):
    """
    Compute angles between row vectors in *vector1* and a single *vector2*.

    Parameters
    ----------
    vector1 : np.ndarray
        ``(N, 3)`` array of vectors.
    vector2 : np.ndarray
        ``(1, 3)`` or ``(3,)`` reference vector.
    radians : bool
        If True return radians, otherwise degrees.

    Returns
    -------
    np.ndarray
        ``(N, 1)`` array of angles.
    """
    unit_vector1 = vector1 / np.linalg.norm(vector1, axis=1)[:, None]
    unit_vector2 = (vector2 / np.linalg.norm(vector2)).T
    dot_product = np.dot(unit_vector1, unit_vector2)
    if radians:
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    else:
        angle = np.rad2deg(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    return angle


def getLength(vector):
    """
    Compute Euclidean length for each row of *vector*.

    Parameters
    ----------
    vector : np.ndarray
        ``(N, 3)`` array of vectors.

    Returns
    -------
    np.ndarray
        ``(N, 1)`` array of lengths.
    """
    return np.sqrt(np.sum(np.square(vector), axis=1)).reshape(len(vector), 1)


def getPlane(planenormal, point):
    """
    Create a ``vtkPlane`` from a normal vector and point.

    Parameters
    ----------
    planenormal : np.ndarray
        Normal vector (3-element).
    point : np.ndarray
        A point on the plane (3-element).

    Returns
    -------
    vtk.vtkPlane
    """
    if planenormal.shape != (3,):
        planenormal = planenormal.reshape(3, 1)
    if point.shape != (3,):
        point = point.reshape(3, 1)

    plane = vtk.vtkPlane()
    plane.SetOrigin(point[0], point[1], point[2])
    plane.SetNormal(planenormal[0], planenormal[1], planenormal[2])
    return plane

# ============================================================================
# SEGMENT STATISTICS UTILS
# ============================================================================
def segmentationGetsegmentstatistics(nodeSeg, nodevolume=None, *filename):
    import SegmentStatistics
    n = nodeSeg
    closedsurface = True
    scalarvolume = False
    segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
    if closedsurface:
        n.CreateClosedSurfaceRepresentation()
        segStatLogic.getParameterNode().SetParameter("Segmentation", n.GetID())
    if nodevolume is not None:
        scalarvolume = True
        vol = nodevolume
        segStatLogic.getParameterNode().SetParameter("ScalarVolume", vol.GetID())
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.centroid_ras.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_x.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_y.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_axis_z.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.principal_moments.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.feret_diameter_mm.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.surface_mm2.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.elongation.enabled",str(True))    
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.flatness.enabled",str(True))
    segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.roundness.enabled",str(True))    
    segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled",str(scalarvolume))    
    segStatLogic.getParameterNode().SetParameter("ClosedSurfaceSegmentStatisticsPlugin.enabled",str(closedsurface))
    segStatLogic.computeStatistics()
    stats = segStatLogic.getStatistics()   
    if list(filename): 
        save_pickle(stats, filename)
    return stats

def segmentationGetVolumemetric(stats):
    volumemetric = dict()
    for segmentId in stats['SegmentIDs']:
        seg_name = stats[segmentId, 'Segment']        
        for measureInfo in stats['MeasurementInfo']:
            new_measureInfo = measureInfo.replace('SegmentStatisticsPlugin.', '')
            volume_metric_name = f'{seg_name}_{new_measureInfo}'
            volumemetric[f'{volume_metric_name}'] = stats[segmentId, measureInfo]                        
    return volumemetric

# ============================================================================
# Serialization & Scene Management
# ============================================================================

def save_pickle(data, filename):
    """Save data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def closeScene():
    """Close (clear) the current Slicer scene."""
    slicer.mrmlScene.Clear(0)
