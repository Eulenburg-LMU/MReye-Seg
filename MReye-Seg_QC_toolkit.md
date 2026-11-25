# MReye-Seg QC toolkit

Quantitative, reproducible QC to replace "visual inspection of registrations."
Each check returns a per-scan dict with a `flag`. Aggregate flags into one QC table,
gate on it, and report `n` excluded.

```
pip install numpy nibabel scipy scikit-image pandas matplotlib
pip install antspyx          # needed only for checks 1 and 2
```

Inputs assume your ANTs SyN outputs per subject:
`sub-XX_0GenericAffine.mat`, `sub-XX_1Warp.nii.gz`, `sub-XX_1InverseWarp.nii.gz`,
the propagated label maps, and the subject anchor image.

---

## The three conventions that will bite you

1. **RAS vs LPS** — Slicer `.fcsv` is RAS; ANTs is LPS. `parse_fcsv()` auto-converts.
   If you skip this, a correct registration looks like a ~40 mm L/R error.
2. **Point transforms invert relative to image transforms** in ANTs. The round-trip
   check cancels this, but validate direction once on a landmark with known correspondence.
3. **Jacobian on the composed field** (`affine ∘ warp`), never the warp alone, or you
   miss the affine scaling component of the volume change.

---

## 1. Registration QC (the missing gate)

```python
import mreye_qc as q

roi = q.make_roi(["sub-01_globe.nii.gz","sub-01_ON.nii.gz","sub-01_ONS.nii.gz"],
                 subject_ref="sub-01_T1w.nii.gz", out_path="sub-01_roi.nii.gz", dilate_mm=3)

jac = q.jacobian_qc("sub-01_T1w.nii.gz", "sub-01_0GenericAffine.mat",
                    "sub-01_1Warp.nii.gz", roi)
rt  = q.landmark_roundtrip_qc("sub-01_landmarks.fcsv", "sub-01_0GenericAffine.mat",
                              "sub-01_1Warp.nii.gz", "sub-01_1InverseWarp.nii.gz")
bd  = q.boundary_dice_qc("sub-01_T2w.nii.gz", "sub-01_globe.nii.gz",
                         roi_box=(x0,x1,y0,y1,z0,z1))   # box around both orbits
```

| metric | meaning | suggested flag |
|---|---|---|
| `pct_folded` | % ROI voxels with det J ≤ 0 (non-diffeomorphic) | > 0 |
| `pct_out_of_range` | % outside det J ∈ [0.5, 2.0] | > 5% |
| `logJ_std` | scalar "how much deformation" | report; compare across cohorts |
| round-trip `max_mm` | forward→inverse landmark error | > 1.0 mm |
| `boundary_dice` | propagated globe vs independent threshold seg | < 0.80 |

**ANTs CLI equivalent** (if you prefer not to use antspyx):
```bash
# compose affine + warp into one field on the subject grid, then its Jacobian
antsApplyTransforms -d 3 -r sub-01_T1w.nii.gz \
  -t sub-01_1Warp.nii.gz -t sub-01_0GenericAffine.mat \
  -o [sub-01_comptx.nii.gz,1]
CreateJacobianDeterminantImage 3 sub-01_comptx.nii.gz sub-01_jac.nii.gz 0 0
# round-trip points (LPS csv with header x,y,z,t):
antsApplyTransformsToPoints -d 3 -i pts_lps.csv -o fwd.csv \
  -t [sub-01_0GenericAffine.mat,1] -t sub-01_1Warp.nii.gz
antsApplyTransformsToPoints -d 3 -i fwd.csv -o back.csv \
  -t sub-01_1InverseWarp.nii.gz -t sub-01_0GenericAffine.mat
```

---

## 2. Accuracy QC (one-time, on a manual subset)

```python
acc = q.accuracy_metrics("sub-01_globe_pred.nii.gz", "sub-01_globe_manual.nii.gz")
# -> dice, assd_mm, hd95_mm, hd_mm, rel_vol_err_pct
```

**Protocol to satisfy a methods reviewer:**
- **Sample:** random, stratified across cohorts and (if any) across scanners/sequences.
  ~10–15 subjects per structure per cohort is a defensible minimum.
- **Blind:** the rater segments natively in subject space with **no view** of the
  propagated label (different session, shuffled filenames).
- **Reference reliability:** have a second rater (or the same rater twice) do a subset,
  so you can report the *manual* inter/intra-rater Dice as the ceiling — the pipeline
  cannot be expected to beat the reference's own agreement.
- **Report:** median [IQR] Dice, ASSD, HD95, and relative volume error per structure per
  cohort; put the manual reliability alongside. Test-retest ICC (which you already have)
  measures precision; **this** measures accuracy. Report both.

---

## 3. Landmark QC

```python
flags = q.plausibility_flags({"globe_width": 24.0, "ONSH_width": 12.0})   # fill PLAUSIBLE_MM
disp  = q.landmark_displacement("rater1.fcsv", "rater2.fcsv")            # mm per landmark
```
Fill `PLAUSIBLE_MM` from your normative Table 1 (mean ± k·SD). Report per-landmark
intra/inter-rater displacement and, ideally, how much each propagates into the derived
metric (re-run the metric with each rater's landmarks).

---

## 4. Gating + sensitivity (turn "we kept the outliers" into evidence)

```python
gated = q.image_quality_gate(scan_df, cols=("EFC","SNR","CNR"), k=2.0)  # per-cohort z-gate
sens  = q.sensitivity_report(my_anova_or_icc_fn, gated)                 # with vs without flagged
```
Set explicit thresholds and **either** exclude failures and report `n`, **or** keep them
and show via `sensitivity_report` that effect sizes / ICCs are unchanged. Note: higher EFC
= more ghosting/blur = worse; lower SNR/CNR = worse.

---

## 5. Longitudinal QC (you have repeated scans at sites 1 and 3)

```python
lc = q.longitudinal_flags(metric_long_df, subject_col="subject", metric_col="value",
                          ref_within_sd=0.5, k=2.5)   # ref = test-retest SD from the 30-min rescan
```
Robust (median-centred) individuals control chart: flags any timepoint outside the
subject's own band. Set `ref_within_sd` from your paired site1 rescan so the band is the
true short-term noise floor, not inflated by real longitudinal change.

---

## 6. Better SNR (use the paired site1 rescan)

```python
# coregister rescan to scan1 first (rigid), then:
snr = q.difference_method_snr("scan1.nii.gz", "scan2_to_scan1.nii.gz", "sub-01_roi.nii.gz")
```
NEMA difference method: noise = SD(scan1 − scan2)/√2 inside the ROI. Avoids the air-mask
bias under parallel imaging / multichannel coils.

---

## Suggested aggregate QC table (one row per scan)

`subject | cohort | timepoint | pct_folded | pct_out_of_range | roundtrip_max_mm |
boundary_dice | EFC | SNR_diff | CNR | qc_fail`

Gate `qc_fail = folding OR out_of_range>5% OR roundtrip>1mm OR boundary_dice<0.8 OR
image-quality flag`. Report counts per cohort, then run the analysis on the passing set
with a sensitivity check on the full set.