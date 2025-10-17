#!/bin/bash
set -e  # Exit on error

# Usage: Run this script from your input directory:
#   cd /path/to/your/images
#   bash run_template_building.sh /path/to/ANTs/bin/

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/ANTs/bin/"
  exit 1
fi

export ANTSPATH="$1"

# Set number of threads for ITK
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=9

# Set paths
inputPath=${PWD}/
outputPath=${PWD}/Template/

# Ensure output directory exists
mkdir -p "${outputPath}"

# Optional: Check if ANTs script exists
if [ ! -x "${ANTSPATH}/antsMultivariateTemplateConstruction2.sh" ]; then
  echo "Error: antsMultivariateTemplateConstruction2.sh not found or not executable in ${ANTSPATH}"
  exit 1
fi

# Single modality template construction (T1w)
"${ANTSPATH}/antsMultivariateTemplateConstruction2.sh" \
  -d 3 \
  -o "${outputPath}T_" \
  -g 0.2 \
  -b 1 \
  -c 2 \
  -j 18 \
  -i 5 \
  -k 1 \
  -q 120x100x70x40 \
  -f 8x4x2x1 \
  -s 3x2x1x0 \
  -n 1 \
  -l 1 \
  -m CC[4] \
  -t BSplineSyN[0.1,26,0] \
  -r 1 \
  "${inputPath}"/Denoised*.nii

# Uncomment below for two modality template construction (T1w and T2w)
# ${ANTSPATH}/antsMultivariateTemplateConstruction2.sh \
#   -d 3 \
#   -o ${outputPath}T_ \
#   -g 0.2 \
#   -b 1 \
#   -c 2 \
#   -j 9 \
#   -i 5 \
#   -k 2 \
#   -w 0.5x1 \
#   -q 100x100x70x40 \
#   -f 8x4x2x1 \
#   -s 3x2x1x0 \
#   -n 1 \
#   -l 1 \
#   -m CC[2] \
#   -t BSplineSyN[0.1,26,0] \
#   -r 1 \
#   ${inputPath}/Denoised*.nii