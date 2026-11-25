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
python landmark_transform.py \
  --template-dir {TEMPLATE_DIR} \
  --template-pattern "T_{{cohort}}T1.nii.gz" \
  --input-dir {INPUT_DIR} \
  --input-pattern "Denoised_*_T1w.nii" \
  --fids-dir {FIDS_DIR} \
  --fids-pattern "fids_{{cohort}}.fcsv" \
  --segs-pattern "Segs_{{cohort}}.seg.nrrd" \
  --cohort "TestCohort" \
  --output-subdir "MReye-Seg" \
  --verbose
"""

python3 template_transform.py 
  --template /mnt/d/users/getang/MReye-Seg/template/High_resolution_template/T_IIH02mmT1.nii.gz 
  --input-dir  /mnt/d/users/getang/MReye-Seg/test 
  --run 
  --verbose

# ============================================================================
# Landmark transform : Run from 3D Slicer
# ============================================================================

exec(open("D:/users/getang/tmp/landmark_transform.py").read())
main([
    '--template-dir', 'D:/users/getang/MReye-Seg/template/High_resolution_template',
    '--template-pattern', 'T_{cohort}.nii.gz',
    '--input-dir', 'D:/users/getang/MReye-Seg/test/sub-02_ses-01',
    '--input-pattern', 'Denoised_*_T2w.nii',
    '--fids-dir', 'D:/users/getang/MReye-Seg/data/Segmentations',
    '--fids-pattern', 'fids_{cohort}.fcsv',
    '--segs-pattern', 'Segs_{cohort}.seg.nrrd',
    '--cohort', 'IIH02mmT2',
    '--output-subdir', 'MReye-Seg',
    '--demo',
    '--verbose',
    '--save-visualization',
])

# ============================================================================
# Extract fiducial metrics: Run from command line
# ============================================================================

python3 fiducial_metrics_extraction.py \
  --project-path /mnt/d/users/getang/MReye-Seg/test \
  --demo \
  --verbose

# ============================================================================
# Globe contour map 3D Slicer module
# ============================================================================

exec(open("D:/users/getang/tmp/globe_contourmap.py").read())

main([
    '--cohort', 'IIH02mmT1',
    '--fids-pattern', 'fids_{cohort}_*.fcsv',
    '--segs-pattern', 'Segs_{cohort}_*.seg.nrrd',
    '--project-path', 'D:/users/getang/MReye-Seg/test',
    '--save-results',
    '--demo',
    '--verbose',
])

# ============================================================================
# Plotting the Globe contour map: Running in the terminal
# ============================================================================

python3 plotting_globe_contourmap.py \
  --project-path /mnt/d/users/getang/MReye-Seg/test \
  --save-data \
  --side-of-eye L \
  --condition control \
  --modality T1 \
  --verbose



#%%
input_files = locate_files(
            'Denoised_*_T1w.nii', 
            Path('D:/users/getang/MReye-Seg/test/sub-02_ses-01'), 
            level=0, 
            sorted_=True
        )