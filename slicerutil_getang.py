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
    centerpoint = np.asarray(centerpoint, dtype=float).reshape(3)
    planenormal = np.asarray(planenormal, dtype=float).reshape(3)

    planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", namePlaneNode)
    planeNode.SetNormal(planenormal)
    planeNode.SetOrigin(centerpoint)

    # Slicer >= 5: use the named enum instead of the literal 1 used in Slicer 4.11.
    planeNode.SetSizeMode(slicer.vtkMRMLMarkupsPlaneNode.SizeModeAbsolute)
    planeNode.SetPlaneBounds(np.array([-200, 200, 200, -200, 200, -200]))
    visFid_SetVisibility(
        planeNode.GetName(), visibility=0.4, locked=1, textScale=0,
        color=(1, 1, 0), glyph_scale=0,
    )
    return planeNode


def getOrCreateMarkupsLineNode(nameLineNode, controlpoints, color=(1, 1, 1)):
    """
    Create a markups line node from two control points.

    Parameters
    ----------
    nameLineNode : str
        Name of the created node.
    controlpoints : np.ndarray
        ``(2, 3)`` array of RAS coordinates.
    color : tuple
        Selected colour as 0-1 RGB.

    Returns
    -------
    vtkMRMLMarkupsLineNode
    """
    controlpoints = np.asarray(controlpoints, dtype=float).reshape(-1, 3)
    lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", nameLineNode)
    for i in range(controlpoints.shape[0]):
        v = vtk.vtkVector3d(controlpoints[i, 0], controlpoints[i, 1], controlpoints[i, 2])
        if lineNode.AddControlPoint(v) == -1:
            raise RuntimeError(f"Could not add control point {i} to line node '{nameLineNode}'")
    visFid_SetVisibility(
        lineNode.GetName(), visibility=1, locked=1, textScale=0, color=color,
        glyph_scale=0.5, line_thickness=1.5, line_opacity=1,
    )
    return lineNode


def markupsLineGetEndpoints(lineNode):
    """
    Return the two endpoints of a markups line node as a ``(2, 3)`` array.

    Slicer 4.11 code read these through ``GetCurvePoints().GetPoint(i)``. In
    Slicer 5.x the curve point buffer of a line node is regenerated lazily and
    may hold interpolated points, so read the control points directly instead.
    """
    if isinstance(lineNode, str):
        lineNode = getNode(lineNode)
    n = lineNode.GetNumberOfControlPoints()
    if n < 2:
        raise ValueError(f"Line node '{lineNode.GetName()}' has {n} control points, expected 2")
    pts = np.zeros((2, 3))
    for i in range(2):
        pos = [0.0, 0.0, 0.0]
        lineNode.GetNthControlPointPositionWorld(i, pos)
        pts[i] = pos
    return pts


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


def getRandom3Dvector(rng=None):
    """
    Generate a random 3D unit vector with a uniform spherical distribution.

    Returns
    -------
    np.ndarray
        ``(1, 3)`` unit vector.
    """
    rng = np.random.default_rng() if rng is None else rng
    phi = rng.uniform(0, np.pi * 2)
    costheta = rng.uniform(-1, 1)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array((x, y, z)).reshape(1, 3)


def planeGetnormal(center_vector, numberofplanenormalvectors, search_range=np.pi / 6, rng=None):
    """
    Sample candidate plane normals within a cone around *center_vector*.

    The first returned vector is always *center_vector* itself, so the search
    can never do worse than the plain perpendicular plane.

    Parameters
    ----------
    center_vector : array_like
        Centre of the search cone (typically the centerline tangent).
    numberofplanenormalvectors : int
        Total number of normals to return (including the centre vector).
    search_range : float
        Half-angle of the search cone in radians. Default 30 degrees.
    rng : np.random.Generator or None
        Pass a seeded generator for reproducible searches.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array of candidate plane normals.
    """
    rng = np.random.default_rng() if rng is None else rng
    start_vector = np.asarray(center_vector, dtype=float).reshape(1, 3)
    vectors = [start_vector]
    while len(vectors) < max(int(numberofplanenormalvectors), 1):
        vector = getRandom3Dvector(rng)
        if getAngle(start_vector, vector) < search_range:
            vectors.append(vector)
    return np.concatenate(vectors, axis=0)


# ============================================================================
# Slice View Control (2D)
# ============================================================================

def getSliceNode(slicename):
    """Return the qMRMLSliceWidget for 'Red' / 'Green' / 'Yellow'."""
    if slicename not in ('Red', 'Green', 'Yellow'):
        raise ValueError("slicename must be one of 'Red' / 'Green' / 'Yellow'")
    return slicer.app.layoutManager().sliceWidget(slicename)


def zoom2D(factor, zoomRed=True, zoomGreen=True, zoomYellow=True):
    """Zoom the selected slice views by *factor* (field of view is divided by it)."""
    for name, enabled in (('Red', zoomRed), ('Green', zoomGreen), ('Yellow', zoomYellow)):
        if not enabled:
            continue
        sn = getSliceNode(name).mrmlSliceNode()
        fov = sn.GetFieldOfView()
        sn.SetFieldOfView(fov[0] / factor, fov[1] / factor, fov[2])


def setViewtoDefault(which_view):
    """Reset a slice view to its default (axial/sagittal/coronal) orientation."""
    slicer.app.layoutManager().sliceWidget(which_view).mrmlSliceNode().SetOrientationToDefault()


def setSliceOffset(slicename, offset, mode='offset'):
    """Jump a slice view to a RAS position, by offsetting or by centering."""
    sn = getSliceNode(slicename).mrmlSliceNode()
    if mode == 'center':
        sn.JumpSliceByCentering(offset[0], offset[1], offset[2])
    elif mode == 'offset':
        sn.JumpSliceByOffsetting(offset[0], offset[1], offset[2])
    else:
        raise ValueError(f"Unknown slice jump mode '{mode}'; use 'offset' or 'center'")


def setSliceOffsetToFiducial(nameFidList=None, fidName=None, fidArray=None, mode='offset'):
    """Jump all slice views to a fiducial position or an explicit RAS point."""
    if fidArray is not None:
        pt = np.asarray(fidArray, dtype=float).ravel()
    else:
        pt = arrayFromFiducialList(nameFidList, [fidName]).ravel()
    for slicename in ('Green', 'Red', 'Yellow'):
        setSliceOffset(slicename, pt, mode=mode)


def sliceViewSetToPlane(planenormal, point, slicename='Green'):
    """
    Orient a slice view onto an arbitrary plane.

    Replaces the Slicer 4.11 habit of driving the Green view through
    ``CrossSectionAnalysisLogic``; ``SetSliceToRASByNTP`` is the supported
    way to do this in Slicer 5.x.
    """
    planenormal = np.asarray(planenormal, dtype=float).reshape(3)
    point = np.asarray(point, dtype=float).reshape(3)
    # Any vector perpendicular to the normal works as the in-plane X axis.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, planenormal / np.linalg.norm(planenormal))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    planeX = np.cross(planenormal, seed)
    sliceNode = slicer.app.layoutManager().sliceWidget(slicename).mrmlSliceNode()
    sliceNode.SetSliceToRASByNTP(
        planenormal[0], planenormal[1], planenormal[2],
        planeX[0], planeX[1], planeX[2],
        point[0], point[1], point[2], 0,
    )


def UpdateSlicePlane(markups, sliceNode):
    """Orient *sliceNode* onto the plane spanned by the first three control points."""
    if isinstance(markups, str):
        markups = getNode(markups)
    if markups.GetNumberOfControlPoints() < 3:
        return
    points = slicer.util.arrayFromMarkupsControlPoints(markups)
    planePosition = points.mean(axis=0)
    planeNormal = np.cross(points[1] - points[0], points[2] - points[0])
    planeX = points[1] - points[0]
    sliceNode.SetSliceToRASByNTP(
        planeNormal[0], planeNormal[1], planeNormal[2],
        planeX[0], planeX[1], planeX[2],
        planePosition[0], planePosition[1], planePosition[2], 0,
    )


# ============================================================================
# Feret Diameter / Cross-Section Outline
# ============================================================================

def _closedSurfaceRepresentationName():
    """Return the closed-surface representation name for the running Slicer version."""
    conv = slicer.vtkSegmentationConverter
    for attr in ('GetClosedSurfaceRepresentationName',
                 'GetSegmentationClosedSurfaceRepresentationName'):
        fn = getattr(conv, attr, None)
        if fn is not None:
            return fn()
    return 'Closed surface'


def segmentationHasSegment(nameSeg, nameRegion):
    """True if *nameRegion* exists in segmentation *nameSeg*."""
    n = getNode(nameSeg)
    if n is None:
        return False
    return bool(n.GetSegmentation().GetSegmentIdBySegmentName(nameRegion))


def segmentationGetClosedSurface(nameSeg, nameRegion):
    """
    Return the closed-surface polydata of a segment.

    Creates the closed-surface representation on demand; in Slicer 5.x a
    segmentation loaded from disk only carries its binary labelmap until
    the surface is requested, and ``GetRepresentation`` silently returns
    ``None`` in that state.
    """
    n = getNode(nameSeg)
    if n is None:
        raise ValueError(f"Segmentation node '{nameSeg}' not found")
    s = n.GetSegmentation()
    segmentId = s.GetSegmentIdBySegmentName(nameRegion)
    if not segmentId:
        raise ValueError(f"Segment '{nameRegion}' not found in segmentation '{nameSeg}'")
    repName = _closedSurfaceRepresentationName()
    if not s.ContainsRepresentation(repName):
        n.CreateClosedSurfaceRepresentation()
    surface = s.GetSegment(segmentId).GetRepresentation(repName)
    if surface is None:
        raise RuntimeError(f"Could not build closed surface for segment '{nameRegion}'")
    return surface


def diameterGetplane(nameModel, idx_point, tangent_window=1):
    """
    Get the cross-section plane (tangent direction + point) at a centerline index.

    Slicer 4.11 obtained this through ``CrossSectionAnalysisLogic.pathArray`` /
    ``fillPathArray()`` / ``process()``. Those attributes no longer exist in
    Slicer 5.x, so the tangent is computed directly from the model points with
    a central difference (the old code used a forward difference, which raised
    an IndexError on the last centerline point).

    Parameters
    ----------
    nameModel : str
        Name of the centerline model node.
    idx_point : int
        Index of the point along the centerline.
    tangent_window : int
        Half-width of the central difference used for the tangent.

    Returns
    -------
    tuple
        ``(direction, point)``, both ``(3,)`` arrays; *direction* is normalised.
    """
    points = modeltopointsarray(nameModel)
    n = len(points)
    if n < 2:
        raise ValueError(f"Centerline model '{nameModel}' has {n} points")
    idx_point = int(np.clip(idx_point, 0, n - 1))
    i0 = max(idx_point - tangent_window, 0)
    i1 = min(idx_point + tangent_window, n - 1)
    if i0 == i1:
        i0, i1 = max(idx_point - 1, 0), min(idx_point + 1, n - 1)
    direction = points[i1] - points[i0]
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError(f"Degenerate centerline tangent at index {idx_point} of '{nameModel}'")
    return direction / norm, points[idx_point]


def diameterGetoutlinePolyData(nameSeg, nameRegion, cutterplane):
    """
    Cut a segment surface with a plane and return the resulting outline polydata.

    No MRML node is created, which makes this cheap enough to call inside the
    optimal-plane search loop.
    """
    surface = segmentationGetClosedSurface(nameSeg, nameRegion)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(cutterplane)
    cutter.SetInputData(surface)
    cutter.Update()
    # The stripper joins the cut line segments into ordered polylines.
    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(cutter.GetOutputPort())
    stripper.Update()
    outline = vtk.vtkPolyData()
    outline.DeepCopy(stripper.GetOutput())
    return outline


def diameterGetoutline(nameSeg, nameRegion, cutterplane, nameModel=None):
    """Cut a segment surface with a plane and add the outline to the scene as a model."""
    outline = diameterGetoutlinePolyData(nameSeg, nameRegion, cutterplane)
    outlinemodel = slicer.modules.models.logic().AddModel(outline)
    outlinemodel.SetName(nameModel if nameModel else f'{nameRegion}_Diametermodel')
    outlinemodel.SetDisplayVisibility(0)
    return outlinemodel


def polylineFromPolyData(polydata):
    """
    Return the ordered points of the longest polyline in *polydata*.

    Returns ``None`` when the polydata holds no lines.
    """
    lines = polydata.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return None
    lines.InitTraversal()
    idList = vtk.vtkIdList()
    longest = None
    while lines.GetNextCell(idList):
        ids = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
        if longest is None or len(ids) > len(longest):
            longest = ids
    if longest is None:
        return None
    return np.array([polydata.GetPoint(i) for i in longest])


def polygonAreaAndPerimeter(points):
    """
    Area and perimeter of a closed planar polygon given its ordered vertices.

    Uses the vector (3D shoelace) formula ``A = |sum(P_i x P_i+1)| / 2``, which
    is origin independent for a closed polygon.

    Returns
    -------
    tuple
        ``(area_mm2, perimeter_mm)``.
    """
    if points is None or len(points) < 3:
        return 0.0, 0.0
    pts = np.asarray(points, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0, 0.0
    closed = np.vstack((pts, pts[0]))
    area = 0.5 * float(np.linalg.norm(np.cross(closed[:-1], closed[1:]).sum(axis=0)))
    perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    return area, perimeter


def modeltopointsarray(nameModel):
    """Return the point coordinates of a model node as an ``(N, 3)`` array."""
    node = nameModel if not isinstance(nameModel, str) else getNode(nameModel)
    if node is None:
        raise ValueError(f"Model node '{nameModel}' not found")
    return slicer.util.arrayFromModelPoints(node)


def diameterGetmaxferet(dist_arr, outline):
    """
    Maximum Feret diameter of an outline.

    Parameters
    ----------
    dist_arr : np.ndarray
        Pairwise distance matrix of the outline points.
    outline : np.ndarray
        ``(N, 3)`` outline point coordinates.

    Returns
    -------
    tuple
        ``(point0max, point1max, maxferetdiameter)``.
    """
    idx0, idx1 = np.unravel_index(np.argmax(dist_arr), dist_arr.shape)
    return outline[idx0, :], outline[idx1, :], float(dist_arr[idx0, idx1])


def diameterGetperpendicularferet(line_curve, point0max, point1max):
    """
    Diameter perpendicular to the maximum Feret diameter, through its midpoint.

    Parameters
    ----------
    line_curve : vtk.vtkPolyData
        The outline curve.
    point0max, point1max : np.ndarray
        The two endpoints of the maximum Feret diameter.

    Returns
    -------
    tuple
        ``(point0perpendicular, point1perpendicular, perpendicularferetdiameter)``.
    """
    perpendicular_dir = np.asarray(point0max, dtype=float) - np.asarray(point1max, dtype=float)
    point = (np.asarray(point0max, dtype=float) + np.asarray(point1max, dtype=float)) / 2

    plane = vtk.vtkPlane()
    plane.SetOrigin(point[0], point[1], point[2])
    plane.SetNormal(perpendicular_dir[0], perpendicular_dir[1], perpendicular_dir[2])

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(line_curve)
    cutter.Update()

    output = cutter.GetOutput()
    nPoints = output.GetNumberOfPoints()
    if nPoints < 2:
        log.warning("Perpendicular cut returned %d intersection point(s); "
                    "perpendicular Feret diameter is undefined", nPoints)
        nan3 = np.full(3, np.nan)
        return nan3, nan3.copy(), float('nan')

    points = np.array([output.GetPoint(i) for i in range(nPoints)])
    if nPoints > 2:
        # A non-convex outline can be cut in more than two places; keep the
        # widest pair so the measurement stays the perpendicular diameter.
        dists = getDistance(points, points)
        idx0, idx1 = np.unravel_index(np.argmax(dists), dists.shape)
        point0perpendicular, point1perpendicular = points[idx0], points[idx1]
    else:
        point0perpendicular, point1perpendicular = points[0], points[1]

    perpendicularferetdiameter = float(getDistance(point0perpendicular, point1perpendicular))
    return point0perpendicular, point1perpendicular, perpendicularferetdiameter


def getSortedcurvepoints(point1, outline, nameCurveNode):
    """
    Build a closed markups curve by nearest-neighbour ordering of outline points.

    Parameters
    ----------
    point1 : np.ndarray
        Point to start the ordering from.
    outline : np.ndarray
        ``(N, 3)`` unordered outline points.
    nameCurveNode : str
        Name for the created closed curve node.

    Returns
    -------
    vtkMRMLMarkupsClosedCurveNode
    """
    closedCurveNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLMarkupsClosedCurveNode", nameCurveNode)
    remaining = np.asarray(outline, dtype=float).copy()
    processingpoint = np.asarray(point1, dtype=float).reshape(1, 3)
    sorted_array = np.zeros((remaining.shape[0], 3))
    for i in range(sorted_array.shape[0]):
        dist_arr = np.atleast_1d(getDistance(remaining, processingpoint))
        min_idx = int(np.argmin(dist_arr))
        sorted_array[i] = remaining[min_idx, :]
        processingpoint = remaining[min_idx, :].reshape(1, 3)
        remaining = np.delete(remaining, min_idx, 0)
    slicer.util.updateMarkupsControlPointsFromArray(closedCurveNode, sorted_array)
    return closedCurveNode


def getResampledcurve(nameCurveNode, samplestep):
    """Resample a markups curve so control points sit *samplestep* mm apart."""
    curve = nameCurveNode if not isinstance(nameCurveNode, str) else getNode(nameCurveNode)
    curve.ResampleCurveWorld(samplestep)
    return curve


def getClosedsurfaceareaandlength(nameCurveNode):
    """Return ``(area_mm2, length_mm)`` of a closed markups curve."""
    curve = nameCurveNode if not isinstance(nameCurveNode, str) else getNode(nameCurveNode)
    curveareamm2 = slicer.modules.markups.logic().GetClosedCurveSurfaceArea(curve)
    curvelengthmm = curve.GetCurveLengthWorld()
    return curveareamm2, curvelengthmm


def getClosedCurveModel(nameSeg, nameRegion, planenormal, point, gap=3,
                        namePrefix=None, nameCurveNode=None, nameModel=None,
                        resample_step=1.0, update_slice_view=True, sliceViewName='Green'):
    """
    Cut a segment with a plane and build the outline model, closed curve and
    the two Feret diameter lines.

    Parameters
    ----------
    nameSeg : str
        Name of the segmentation node.
    nameRegion : str
        Segment to cut (the optic nerve sheath, e.g. ``'L_ONS'``).
    planenormal : array_like
        Normal of the cutting plane.
    point : array_like
        Point the cutting plane passes through.
    gap : int
        Distance from the nerve tip in mm; part of every created node name, so
        several gaps can be measured in one scene without name collisions.
    namePrefix : str or None
        Prefix for the created node names. Defaults to *nameRegion*.
    resample_step : float
        Control point spacing of the resampled closed curve, in mm.
    update_slice_view : bool
        Orient *sliceViewName* onto the cutting plane.

    Returns
    -------
    tuple
        ``(outlinemodel, closedCurveNode, maxferetlineNode, perpendicularferetlineNode)``
    """
    planenormal = np.asarray(planenormal, dtype=float).reshape(3)
    point = np.asarray(point, dtype=float).reshape(3)
    plane = getPlane(planenormal, point)
    prefix = namePrefix if namePrefix else nameRegion

    outlinemodel = diameterGetoutline(
        nameSeg, nameRegion, plane,
        nameModel=nameModel if nameModel else f'{prefix}_{gap}mmDiametermodel')
    outlinearray = modeltopointsarray(outlinemodel.GetName())
    if outlinearray is None or len(outlinearray) < 3:
        raise RuntimeError(f"Cutting '{nameRegion}' at {gap}mm produced an empty outline")

    dist_arr = getDistance(outlinearray, outlinearray, metric='euclidean')
    point0max, point1max, _ = diameterGetmaxferet(dist_arr, outlinearray)
    point0perpendicular, point1perpendicular, _ = diameterGetperpendicularferet(
        outlinemodel.GetPolyData(), point0max, point1max)

    closedCurveNode = getSortedcurvepoints(
        point0max, outlinearray,
        nameCurveNode if nameCurveNode else f'{prefix}_{gap}mmDiametercurve')
    getResampledcurve(closedCurveNode.GetName(), resample_step)

    if update_slice_view:
        sliceViewSetToPlane(planenormal, point, sliceViewName)

    visFid_SetVisibility(closedCurveNode.GetName(), visibility=1, locked=1, textScale=0,
                         glyph_scale=0.6, line_thickness=0.6)
    maxferetlineNode = getOrCreateMarkupsLineNode(
        f'{prefix}_{gap}mmferetmax',
        np.concatenate((point0max.reshape(1, 3), point1max.reshape(1, 3)), axis=0),
        color=(0, 1, 0))
    if np.isnan(point0perpendicular).any() or np.isnan(point1perpendicular).any():
        # NaN control points would break rendering; report the failure instead.
        perpendicularferetlineNode = None
    else:
        perpendicularferetlineNode = getOrCreateMarkupsLineNode(
            f'{prefix}_{gap}mmferetperpendicular',
            np.concatenate((point0perpendicular.reshape(1, 3),
                            point1perpendicular.reshape(1, 3)), axis=0),
            color=(0, 1, 1))
    return outlinemodel, closedCurveNode, maxferetlineNode, perpendicularferetlineNode


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
