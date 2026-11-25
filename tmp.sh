scp getang@138.245.135.142:D:/users/getang/SANS/Slicertools/slicerutil_getang.py 

python3 template_transform.py --template /mnt/d/users/getang/IIH/template/High_resolution_template/T_IIH02mmT1.nii.gz --input-dir /mnt/d/users/getang/IIH/test --run --verbose

exec(open("D:/users/getang/IIH/landmark_transform.py").read())
main([
    '--template-dir', 'D:/users/getang/IIH/template/High_resolution_template',
    '--template-pattern', 'T_{cohort}.nii.gz',
    '--input-dir', 'D:/users/getang/IIH/test',
    '--input-pattern', 'Denoised_*_T1w.nii',
    '--fids-dir', 'D:/users/getang/IIH/data/Segmentations',
    '--fids-pattern', 'fids_{cohort}.fcsv',
    '--segs-pattern', 'Segs_{cohort}.seg.nrrd',
    '--cohort', 'IIH02mmT1',
    '--output-subdir', 'MReye',
    '--verbose',
])

# Take the header from the first file, then data lines from all files
head -1 $(ls ~/IIH_brainage/output/*.csv | head -1) > ~/IIH_brainage/output/all_subjects_brainage.csv
for f in ~/IIH_brainage/output/*.csv; do
    # Skip the combined file itself and skip the header line of each file
    [[ "$f" == *"all_subjects_brainage.csv"* ]] && continue
    tail -n +2 "$f"
done >> ~/IIH_brainage/output/all_subjects_brainage.csv


python3 fiducial_metrics_extraction.py \
  --project-path D:/users/getang/IIH/test \
  --cohort IIH02mmT1 \
  --save-results \
  --verbose


head -1 $(ls ~/IIH_brainage/output/*.csv | head -1) > ~/IIH_brainage/output/all_subjects_brainage.csv

for f in ~/IIH_brainage/output/*.csv; do
    # Skip the combined file itself and skip the header line of each file
    [[ "$f" == *"all_subjects_brainage.csv"* ]] && continue
    tail -n +2 "$f"
done >> ~/IIH_brainage/output/all_subjects_brainage.csv