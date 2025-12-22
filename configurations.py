# -*- coding: utf-8 -*-
"""
Configuration Examples for MReye Pipelines

This file contains example configurations and command-line invocations
for the MReye landmark transformation and globe shape analysis pipelines.

These examples demonstrate how to run the pipelines with different
parameters. Replace placeholder paths with your actual file paths.

License: MIT
Author: GeTang
"""

# ============================================================================
# Template transform : Run from command line
# ============================================================================
"""
python3 template_transform.py 
  --template {TEMPLATE_DIR}
  --input-dir {INPUT_DIR}
  --run 
  --verbose

"""

# ============================================================================
# Landmarks transform : Run from 3D Slicer
# ============================================================================

# Example 1: Basic landmark transformation
"""
exec(open("{script_dir}").read())
main([
    '--template-dir', {template_dir},
    '--template-pattern', 'T_{cohort}.nii.gz',
    '--input-dir', {input_dir},
    '--input-pattern', 'Denoised_*_T2w.nii',
    '--fids-dir', {fids_dir},
    '--fids-pattern', 'fids_{cohort}.fcsv',
    '--segs-pattern', 'Segs_{cohort}.seg.nrrd',
    '--cohort', {cohort},
    '--output-subdir', {project_name},
    '--demo',
    '--verbose',
    '--save-visualization',
])

"""

# ============================================================================
# Extract fiducial metrics: Run from command line
# ============================================================================
"""
python3 fiducial_metrics_extraction.py \
  --project-path {PROJECT_PATH} \
  --demo \
  --verbose
"""

# ============================================================================
# Globe Contour Map : Run from 3D Slicer
# ============================================================================

# Example 1: Basic globe shape analysis
# Run from command line:
"""
exec(open("{script_dir}/globe_contourmap.py").read())

main([
    '--cohort', 'IIH02mmT1',
    '--fids-pattern', 'fids_{cohort}_*.fcsv',
    '--segs-pattern', 'Segs_{cohort}_*.seg.nrrd',
    '--project-path', '{project_path}',
    '--save-results',
    '--demo',
    '--verbose',
])
"""

# ============================================================================
# Plotting Globe Contour Map : Run from command line
# ============================================================================

# Example 1: Plot control data for left eye
# Run from command line:
"""
python plotting_globe_contourmap.py \
  --project-path {PROJECT_PATH} \
  --save-data \
  --side-of-eye L \
  --condition control \
  --modality T1 \
  --verbose
"""
