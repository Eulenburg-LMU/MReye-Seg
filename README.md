# MReye-Seg

An open-access pipeline for MRI-based eye globe segmentation, landmark transformation, fiducial metric extraction, and globe shape (contour map) analysis.

**License:** MIT
**Author:** Ge Tang

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Pipeline Steps](#pipeline-steps)
   - [Step 1: Template Building](#step-1-template-building)
   - [Step 2: Template Registration](#step-2-template-registration)
   - [Step 3: Landmark Transformation](#step-3-landmark-transformation)
   - [Step 4: Fiducial Metrics Extraction](#step-4-fiducial-metrics-extraction)
   - [Step 5: Globe Contour Map Analysis](#step-5-globe-contour-map-analysis)
   - [Step 6: Plotting Globe Contour Maps](#step-6-plotting-globe-contour-maps)
5. [File Format Reference](#file-format-reference)
6. [Fiducial Landmark Reference](#fiducial-landmark-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

MReye-Seg provides a series of tools for analyzing eye globe anatomy from MRI data:

1. **Template Building** — Construct a population-specific MRI template using ANTs
2. **Template Registration** — Register the template to individual subject MRI volumes
3. **Landmark Transformation** — Transform fiducial landmarks and segmentations from template space to subject space (runs in 3D Slicer)
4. **Fiducial Metrics Extraction** — Extract anatomical measurements (distances, widths, depths) from transformed fiducial landmarks
5. **Globe Contour Map Analysis** — Compute polar or orthogonal projections of the eyeball surface (runs in 3D Slicer)
6. **Contour Map Plotting** — Generate 2D contour map visualizations of globe shape

---

## Prerequisites

### Software

| Software | Required By | Version |
|----------|-------------|---------|
| [ANTs](https://github.com/ANTsX/ANTs) | Template building & registration | >= 2.3 |
| [3D Slicer](https://www.slicer.org/) | Landmark transform, Globe contour map | >= 5.6 |
| Python 3 | All scripts | >= 3.8 |

### Python Packages

```
numpy
pandas
scipy
matplotlib
SimpleITK
nipype
```

For 3D Slicer scripts (`landmark_transform.py`, `globe_contourmap.py`), the Slicer built-in Python environment provides `vtk` and `slicer` modules automatically.

### Additional Modules

- `slicerutil_getang` — A custom 3D Slicer utility module (must be on the Python path when running in Slicer)

---

## Data Preparation

### Directory Structure

Organize your data following a BIDS-like structure:

```
project_root/
├── sub-01_ses-01/
│   └── anat/
│       ├── Denoised_sub-01_ses-01_T1w.nii      # Denoised T1-weighted MRI
│       └── Denoised_sub-01_ses-01_T2w.nii      # Denoised T2-weighted MRI (optional)
├── sub-01_ses-02/
│   └── anat/
│       ├── Denoised_sub-01_ses-02_T1w.nii
│       └── Denoised_sub-01_ses-02_T2w.nii
├── sub-02_ses-01/
│   └── anat/
│       └── ...
├── derivatives/
│   ├── sub-01_ses-01/
│   │   └── anat/
│   │       └── MReye-Seg/                            # Pipeline outputs per subject
│   │           ├── fids_{cohort}_*.fcsv          # Transformed fiducials
│   │           ├── Segs_{cohort}_*.seg.nrrd      # Transformed segmentations
│   │           ├── Polar_projection_*.vtp         # Globe projection data
│   │           └── Orthogonal_projection_*.pickle
│   └── Summary/
│       └── allFidDistances.csv                   # Aggregated fiducial metrics
└── template/
    └── High_resolution_template/
        └── T_{cohort}T1.nii.gz                   # Population template
```

### Input MRI Files

- **Format:** NIfTI (`.nii` or `.nii.gz`)
- **Naming convention:** `Denoised_{subject}_{session}_{modality}.nii`
  - Example: `Denoised_sub-01_ses-01_T1w.nii`
- **Preprocessing:** Images should be denoised before entering the pipeline. The filename prefix `Denoised_` is expected by default.

### Template Fiducials (`.fcsv`)

Template fiducials are 3D Slicer markup fiducial files that define anatomical landmarks on the template volume. These are placed manually in 3D Slicer and saved as `.fcsv` files.

- **Naming convention:** `fids_{cohort}.fcsv`
  - Example: `fids_IIH02mmT1.fcsv`

### Template Segmentations (`.seg.nrrd`)

Template segmentations define anatomical structures (eyeballs, optic nerves, optic nerve sheaths) on the template volume.

- **Naming convention:** `Segs_{cohort}.seg.nrrd`
  - Example: `Segs_IIH02mmT1.seg.nrrd`
- **Expected segments:** `L_eyeball`, `R_eyeball`, `L_optic_nerve`, `R_optic_nerve`, `L_optic_nerve_sheath_anterior_with_nerve`, `R_optic_nerve_sheath_anterior_with_nerve`

### Cohort Naming

The `cohort` string encodes the study group, resolution, and modality. Examples:

- `IIH02mmT1` — IIH cohort, 0.2mm resolution, T1-weighted
- `IIH02mmT2` — IIH cohort, 0.2mm resolution, T2-weighted

---

## Pipeline Steps

### Step 1: Template Building

Build a population-specific MRI template using ANTs `antsMultivariateTemplateConstruction2.sh`.

**Run from the terminal:**

```bash
# Navigate to your input directory containing Denoised*.nii files
cd /path/to/your/images

# Run template building (single modality, T1w)
bash run_template_building.sh /path/to/ANTs/bin/
```

**Key parameters in `run_template_building.sh`:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `-d` | 3 | 3D image |
| `-o` | `${outputPath}T_` | Output prefix |
| `-g` | 0.2 | Gradient step |
| `-i` | 5 | Number of iterations |
| `-k` | 1 | Number of modalities (1 for single, 2 for dual T1+T2) |
| `-q` | 120x100x70x40 | Max iterations per level |
| `-m` | CC[4] | Cross-correlation metric with radius 4 |
| `-t` | BSplineSyN[0.1,26,0] | Transformation model |

**Output:** Template volume at `{input_dir}/Template/T_template0.nii.gz`

For dual-modality (T1w + T2w) template building, uncomment the second command block in `run_template_building.sh` and set `-k 2`.

---

### Step 2: Template Registration

Register the template to individual subject MRI volumes using ANTs.

**Run from the terminal:**

```bash
python3 template_transform.py \
  --template /path/to/template/T_IIH02mmT1.nii.gz \
  --input-dir /path/to/project_root \
  --run \
  --verbose
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--template`, `-t` | Yes | — | Path to template volume file (`.nii.gz`) |
| `--input-dir`, `-i` | Yes | — | Directory to search for subject files (searches recursively for `Denoised*T1w.nii*`) |
| `--output-root`, `-o` | No | Subject folder | Root output directory |
| `--run` | No | Dry-run | Actually execute registrations (omit for dry-run) |
| `--verbose`, `-v` | No | Off | Verbose logging |

**Output per subject:**
- `out_volATLDeformed.nii.gz` — Warped template
- `trf-Temp_to_sub-{id}_ses-{ses}_AFF.mat` — Affine transform
- `trf-Temp_to_sub-{id}_ses-{ses}_DEF.nii.gz` — Deformation field
- `trf-Temp_to_sub-{id}_ses-{ses}_AFFinv.mat` — Inverse affine transform
- `trf-Temp_to_sub-{id}_ses-{ses}_DEFinv.nii.gz` — Inverse deformation field

---

### Step 3: Landmark Transformation

Transform template fiducials and segmentations into each subject's native space using the registration transforms from Step 2.

**This script must be run inside 3D Slicer's Python environment.**

**Run from 3D Slicer's Python Interactor:**

```python
exec(open("/path/to/landmark_transform.py").read())
main([
    '--template-dir', '/path/to/template/High_resolution_template',
    '--template-pattern', 'T_{cohort}.nii.gz',
    '--input-dir', '/path/to/project_root',
    '--input-pattern', 'Denoised_*_T1w.nii',
    '--fids-dir', '/path/to/template/Segmentations',
    '--fids-pattern', 'fids_{cohort}.fcsv',
    '--segs-pattern', 'Segs_{cohort}.seg.nrrd',
    '--cohort', 'IIH02mmT1',
    '--output-subdir', 'MReye-Seg',
    '--verbose',
])
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--template-dir` | Yes | — | Directory containing template volumes |
| `--template-pattern` | No | `T_{cohort}T1.nii.gz` | Template filename pattern (`{cohort}` is replaced) |
| `--input-dir` | Yes | — | Root directory with subject MRI files |
| `--input-pattern` | No | `Denoised_*_T1w.nii` | Input MRI filename glob pattern |
| `--fids-dir` | Yes | — | Directory containing template fiducial files |
| `--fids-pattern` | No | `fids_{cohort}.fcsv` | Fiducial filename pattern |
| `--segs-pattern` | No | `Seg_{cohort}.seg.nrrd` | Segmentation filename pattern |
| `--cohort` | Yes | — | Cohort name (e.g., `IIH02mmT1`) |
| `--output-subdir` | No | `MReye-Seg` | Output subdirectory name |
| `--demo` | No | Off | Process only the first subject |
| `--verbose`, `-v` | No | Off | Verbose logging |
| `--save-visualization` | No | Off | Save visualization screenshots |

**Output per subject** (in `derivatives/{sub}_{ses}/anat/MReye-Seg/`):
- `fids_{cohort}_*.fcsv` — Transformed fiducial landmarks
- `Segs_{cohort}_*.seg.nrrd` — Transformed segmentations
- Centerline metrics and cross-section measurements (CSV)

---

### Step 4: Fiducial Metrics Extraction

Extract distance measurements from the transformed fiducial landmarks.

**Run from the terminal:**

```bash
python3 fiducial_metrics_extraction.py \
  --project-path /path/to/project_root \
  --save-results \
  --verbose
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--project-path` | Yes | — | Project root directory |
| `--mri-pattern` | No | `Denoised_*_{modality}.nii` | MRI filename pattern |
| `--fids-pattern` | No | `fids_{cohort}_*.fcsv` | Fiducial filename pattern |
| `--output-subdir` | No | `MReye-Seg` | Output subdirectory name |
| `--output-format` | No | `csv` | Output format: `csv` or `xlsx` |
| `--save-results` | No | Off | Save results to file |
| `--demo` | No | Off | Process first modality only |
| `--verbose`, `-v` | No | Off | Verbose logging |

**Output:** `derivatives/Summary/allFidDistances.csv` — Table of per-subject anatomical measurements.

---

### Step 5: Globe Contour Map Analysis

Compute polar or orthogonal projections of the eyeball surface for shape analysis.

**This script must be run inside 3D Slicer's Python environment.**

**Run from 3D Slicer's Python Interactor:**

```python
exec(open("/path/to/globe_contourmap.py").read())
main([
    '--cohort', 'IIH02mmT1',
    '--fids-pattern', 'fids_{cohort}_*.fcsv',
    '--segs-pattern', 'Segs_{cohort}_*.seg.nrrd',
    '--project-path', '/path/to/project_root',
    '--save-results',
    '--verbose',
])
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--cohort` | Yes | — | Cohort name |
| `--fids-pattern` | No | `fids_{cohort}.fcsv` | Fiducial filename pattern |
| `--segs-pattern` | No | `Segs_{cohort}_*.seg.nrrd` | Segmentation filename pattern |
| `--mri-pattern` | No | `Denoised_*_T1w.nii` | MRI filename pattern |
| `--project-path` | Yes | — | Project root directory |
| `--orthogonal-projection` | No | Off (polar) | Use orthogonal instead of polar projection |
| `--scaling` | No | 50 | Scaling for intersection line computation |
| `--degree-to-center` | No | 90 | Polar angle threshold (degrees from optic nerve axis) |
| `--sampling-points` | No | 10000 | Number of sampling points |
| `--save-results` | No | Off | Save projection data |
| `--demo` | No | Off | Process single subject only |
| `--verbose`, `-v` | No | Off | Verbose logging |

**Output per subject** (in `derivatives/{sub}_{ses}/anat/MReye-Seg/`):
- `Polar_projection_{eyeball}_{degree}_sub-{id}_ses-{ses}.vtp` — VTK polydata (polar mode)
- `Orthogonal_projection_{eyeball}_{degree}_sub-{id}_ses-{ses}.pickle` — Pickle data (orthogonal mode)

---

### Step 6: Plotting Globe Contour Maps

Generate 2D contour map visualizations from the projection results.

**Run from the terminal:**

```bash
python3 plotting_globe_contourmap.py \
  --project-path /path/to/project_root \
  --save-data \
  --side-of-eye L R \
  --condition control \
  --modality T1 \
  --verbose
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--project-path` | Yes | — | Project root directory |
| `--save-data` | No | Off | Pre-process and save projection data to `.npy` files |
| `--projection-map` | No | `Polar` | Projection type: `Polar` or `Orthogonal` |
| `--degree-to-center` | No | 90 | Degree threshold |
| `--side-of-eye` | No | `L R` | Eye sides to plot: `L`, `R`, or both |
| `--modality` | No | `T1` | MRI modality: `T1` or `T2` |
| `--condition` | No | `control` | Analysis condition (e.g., `control` for mean, other values for variance) |
| `--contour-level` | No | 200 | Number of contour levels |
| `--demo` | No | Off | Limited processing for testing |
| `--verbose`, `-v` | No | Off | Verbose logging |

**Prerequisites:** Requires a `Polar_projection_info.csv` file in the `derivatives/` directory that contains columns `group`, `side_of_eyeball`, and `modality` for filtering subjects.

**Output:**
- `derivatives/{plotname}.pdf` — Contour map PDF plots
- `derivatives/Polar_projection_eyeball_{degree}_summary.npy` — Aggregated projection data
- `derivatives/Polar_grid_eyeball_{degree}_summary.npy` — Grid coordinates

---

## File Format Reference

### Fiducial Files (`.fcsv`)

3D Slicer markup fiducial CSV files. The header lines begin with `#`. Data columns:

```
id, x, y, z, ow, ox, oy, oz, vis, sel, lock, label, desc, associatedNodeID
```

- **x, y, z** — 3D coordinates in RAS (Right-Anterior-Superior) space
- **label** — Anatomical landmark name (see [Fiducial Landmark Reference](#fiducial-landmark-reference))

### Segmentation Files (`.seg.nrrd`)

3D Slicer segmentation files in NRRD format. Each segment is a labeled region. Expected segment names include:

- `L_eyeball`, `R_eyeball`
- `L_optic_nerve`, `R_optic_nerve`
- `L_optic_nerve_sheath_anterior_with_nerve`, `R_optic_nerve_sheath_anterior_with_nerve`

### Transform Files

Generated by ANTs registration:
- `.mat` — Affine transform matrices
- `.nii.gz` — Deformation fields (warp and inverse warp)

---

## Fiducial Landmark Reference

The following fiducial landmark labels are expected in the `.fcsv` files. Each landmark is defined for both left (`L`) and right (`R`) sides.

### T1 and T2 Modalities (shared landmarks)

| Label | Description |
|-------|-------------|
| `center_{L/R}_lens` | Center of the lens |
| `center_{L/R}_eyeball` | Center of the eyeball |
| `nerve_tip_{L/R}` | Tip of the optic nerve |
| `eyeball_back_{L/R}` | Posterior pole of the eyeball |
| `eyeball_midline_{L/R}_lat` | Lateral point at the eyeball midline (equator) |
| `eyeball_midline_{L/R}_med` | Medial point at the eyeball midline (equator) |
| `nerve_baseline_muscle_{L/R}_lat` | Lateral nerve baseline at muscle insertion |
| `nerve_baseline_muscle_{L/R}_med` | Medial nerve baseline at muscle insertion |
| `nerve_baseline_bone_{L/R}_lat` | Lateral nerve baseline at bony orbit |
| `nerve_baseline_bone_{L/R}_med` | Medial nerve baseline at bony orbit |
| `optcanal_height_{L/R}_inf` | Inferior optic canal margin |
| `optcanal_height_{L/R}_sup` | Superior optic canal margin |
| `optcanal_width_{L/R}_lat` | Lateral optic canal margin |
| `optcanal_width_{L/R}_med` | Medial optic canal margin |
| `orbital_rim_{L/R}_lat` | Lateral orbital rim |
| `orbital_rim_{L/R}_med` | Medial orbital rim |
| `orbital_rim_{L/R}_sup` | Superior orbital rim |
| `orbital_rim_{L/R}_inf` | Inferior orbital rim |

### T2 Modality Only (additional landmark)

| Label | Description |
|-------|-------------|
| `nerve_width_{L/R}_lat` | Lateral optic nerve width |
| `nerve_width_{L/R}_med` | Medial optic nerve width |

### Extracted Measurements

| Metric | Description |
|--------|-------------|
| `d1` | Distance: lens center to eyeball center |
| `d2` | Distance: eyeball center to nerve tip |
| `d3` | Distance: lens center to posterior eyeball |
| `d4` | Distance: eyeball center to orbital rim plane |
| `d5` | Distance: lens center to orbital rim plane |
| `w1` | Width: eyeball midline (lateral to medial) |
| `w2` | Width: nerve baseline at muscle (lateral to medial) |
| `w3` | Width: nerve baseline at bone (lateral to medial) |
| `w4` | Width: optic canal (lateral to medial) |
| `h1` | Height: optic canal (inferior to superior) |
| `n1` | Width: optic nerve (T2 only, lateral to medial) |

---

## Troubleshooting

### "Not running in 3D Slicer" error
`landmark_transform.py` and `globe_contourmap.py` must be run inside 3D Slicer's Python Interactor. Use `exec(open(...).read())` to load the script, then call `main([...])`.

### ANTs not found
Ensure ANTs binaries are on your `PATH` or pass the full path to the ANTs `bin/` directory when running `run_template_building.sh`. For `template_transform.py`, nipype must be installed and configured to find ANTs.

### No files found
Check that your MRI filenames match the expected patterns (e.g., `Denoised_*_T1w.nii`). The filename prefix `Denoised_` and the subject/session structure (`sub-{id}_ses-{id}`) are expected by default.

### Windows / WSL
The pipeline includes cross-platform path handling. On Windows, you may need to run ANTs commands through WSL. The scripts will attempt to convert paths automatically.
