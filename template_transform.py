# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2025 GeTang
# Author: Ge Tang 

#%%
from __future__ import annotations
import sys
import os
from pathlib import Path
import platform
import subprocess
import logging
import argparse
import shlex
import time
import shutil
import re

#%% Optional imports; fail fast with friendly messages.
try:
    import SimpleITK as sitk
except Exception as e:
    sitk = None
    logging.getLogger(__name__).warning("SimpleITK not available: %s", e)

try:
    # nipype may not be installed in all environments; keep lazy import pattern
    from nipype.interfaces import ants
except Exception as e:
    ants = None
    logging.getLogger(__name__).warning("nipype.ants not available: %s", e)

# Helper: locate files (fallback to glob if file_search_tool not present)
try:
    import file_search_tool as fs
except Exception:
    import glob
    def locateFilesDf(pattern, root, level=3, tailOnly=False, sorted=False):
        # very small fallback that returns a simple list of paths
        p = Path(root)
        files = list(p.rglob(pattern))
        return files
    fs = type("fs", (), {"locateFilesDf": locateFilesDf})

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Platform helpers
_IS_WINDOWS = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"
_IS_LINUX = platform.system() == "Linux"

def run_command(cmd, check=True, shell=False, env=None):
    """
    Cross-platform subprocess runner.
    If running on Windows and the command looks like a Linux/WSL command,
    the caller can pass shell=True or pass a list starting with 'wsl'.
    """
    log.debug("Running command: %s", cmd)
    if isinstance(cmd, (list, tuple)):
        proc = subprocess.run(cmd, check=check, capture_output=True, text=True, env=env, shell=shell)
    else:
        # cmd is string
        proc = subprocess.run(cmd, check=check, capture_output=True, text=True, env=env, shell=shell)
    if proc.returncode != 0:
        log.error("Command failed (%s): %s", proc.returncode, proc.stderr)
    else:
        log.debug("Command stdout: %s", proc.stdout)
    return proc

# Path conversion helpers (kept for compatibility with older code that used WSL on Windows)
def convert_cmd_to_linux(cmd: str) -> str:
    # Convert simple Windows-style D: paths to WSL /mnt/d equivalents
    if _IS_WINDOWS:
        return cmd.replace('\\', '/').replace(':', '').replace('D/', '/mnt/d/')
    return cmd.replace('\\', '/')

def convert_linux_to_win(filename: str) -> str:
    if _IS_WINDOWS:
        return filename.replace('/mnt/d', 'D:').replace('/', '\\')
    return filename

# Keep regAnts and n4normalize structure but make imports optional and safer
def regAnts(ffFixed, ffMoving, output_prefix='output_',
            transforms=['Affine', 'SyN'],
            transform_parameters=[(2.0,), (0.25, 3.0, 0.0)],
            number_of_iterations=[[1500, 200], [70, 90, 20]],
            metric=['Mattes', 'CC'],
            metric_weight=[1, 1],
            num_threads=4,
            radius_or_number_of_bins=[32, 2],
            sampling_strategy=['Random', None],
            sampling_percentage=[0.05, None],
            convergence_threshold=[1.e-8, 1.e-9],
            convergence_window_size=[20, 20],
            smoothing_sigmas=[[2, 1], [3, 2, 1]],
            sigma_units=['vox', 'vox'],
            composite_trf=False,
            shrink_factors=[[4, 2], [4, 3, 2]],
            use_histogram_matching=[True, True],
            initial_geometric_Align=False,
            verbose=True,
            output_warped_image = 'output_warped_image.nii.gz'):
    if ants is None:
        raise RuntimeError("nipype.interfaces.ants is required for registration (install nipype).")
    reg = ants.Registration()
    reg.inputs.fixed_image = str(ffFixed)
    reg.inputs.moving_image = str(ffMoving)
    reg.inputs.output_transform_prefix = output_prefix
    reg.inputs.transforms = transforms
    reg.inputs.transform_parameters = transform_parameters
    reg.inputs.number_of_iterations = number_of_iterations
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = composite_trf
    reg.inputs.metric = metric
    reg.inputs.metric_weight = metric_weight
    reg.inputs.num_threads = num_threads
    reg.inputs.radius_or_number_of_bins = radius_or_number_of_bins
    reg.inputs.sampling_strategy = sampling_strategy
    reg.inputs.sampling_percentage = sampling_percentage
    reg.inputs.convergence_threshold = convergence_threshold
    reg.inputs.convergence_window_size = convergence_window_size
    reg.inputs.smoothing_sigmas = smoothing_sigmas
    reg.inputs.sigma_units = sigma_units
    reg.inputs.shrink_factors = shrink_factors
    reg.inputs.use_histogram_matching = use_histogram_matching
    if initial_geometric_Align:
        reg.inputs.initial_moving_transform_com = 0
    if verbose:
        reg.inputs.verbose = True
    log.info("Prepared antsRegistration command.")
    return reg

def n4normalize(ffIN, ffOUT):
    if ants is None:
        raise RuntimeError("nipype.interfaces.ants is required for N4BiasFieldCorrection.")
    n4 = ants.N4BiasFieldCorrection(dimension=3)
    n4.inputs.input_image = str(ffIN)
    n4.inputs.output_image = str(ffOUT)
    log.info("Prepared N4BiasFieldCorrection command.")
    return n4

def run_n4normalize(ffIN, ffOUT, runcmd=True, skipIfExists=True):
    n4 = n4normalize(ffIN, ffOUT)
    if runcmd:
        if skipIfExists and Path(ffOUT).exists():
            log.info("N4 output exists; skipping: %s", ffOUT)
        else:
            # run via subprocess
            cmd = n4.cmdline if hasattr(n4, "cmdline") else None
            if cmd is None:
                raise RuntimeError("Unable to get cmdline from N4 interface.")
            run_command(cmd, shell=True)
    return n4

def splitext(fn: str):
    if fn.endswith('.nii.gz'):
        root = fn[:-7]
        ext = '.nii.gz'
        return (root, ext)
    else:
        return os.path.splitext(fn)

def osnj(*args):
    return str(Path().joinpath(*args).as_posix())

def run_reg(ffFIX, ffMOV,
            output_prefix='out_',
            output_warped_image='deformed.nii',
            pnOUT=None, ffAFF=None, ffDEF=None, ffAFFinv=None,
            ffDEFinv=None, DEFVOL=None,
            skipIfExists=True, runcmd=True):
    """
    A more portable run_reg that uses pathlib.Path and returns the registration object.
    This implementation avoids Windows-specific path munging unless required.
    """
    ffFIX = Path(ffFIX)
    ffMOV = Path(ffMOV)
    if pnOUT is None:
        pnOUT = ffFIX.parent
    else:
        pnOUT = Path(pnOUT)
    fnFIX = ffFIX.name
    fnMOV = ffMOV.name

    # create output directory
    pnOUT.mkdir(parents=True, exist_ok=True)

    which_sub = re.search(r'sub-(.+?)_ses', fnFIX)
    if which_sub:
        sesnum = re.search(r'ses-(.+?)_', fnMOV)
        trfPrefix = pnOUT / f'trf-Temp_to_sub-{which_sub.group(1)}_ses-{sesnum.group(1) if sesnum else "unknown"}'
    else:
        # fallback
        m = re.search(r'sub-(.+?)_ses', fnMOV)
        subid = m.group(1) if m else 'unknown'
        sesnum = re.search(r'ses-(.+?)_', fnMOV)
        trfPrefix = pnOUT / f'trf-Temp_to_sub-{subid}_ses-{sesnum.group(1) if sesnum else "unknown"}'

    if ffAFF is None:
        ffAFF = trfPrefix.with_name(trfPrefix.name + '_AFF.mat')
    if ffDEF is None:
        ffDEF = trfPrefix.with_name(trfPrefix.name + '_DEF.nii.gz')
    if ffAFFinv is None:
        ffAFFinv = trfPrefix.with_name(trfPrefix.name + '_AFFinv.mat')
    if ffDEFinv is None:
        ffDEFinv = trfPrefix.with_name(trfPrefix.name + '_DEFinv.nii.gz')
    if DEFVOL is None:
        DEFVOL = trfPrefix.with_name(trfPrefix.name + '_DEFVOL.nii.gz')

    reg = regAnts(ffFIX, ffMOV,
                  output_prefix=str(output_prefix),
                  output_warped_image=str(output_warped_image),
                  transform_parameters=[(0.5,), (0.25, 3.0, 0.0)],
                  smoothing_sigmas = [[2,1], [4,2,1,0]],
                  shrink_factors = [[4,2], [8,4,2,1]],
                  number_of_iterations=[[500, 250], [100, 70, 50, 40]],
                  composite_trf=True,
                  convergence_threshold=[1.e-7, 1.e-8],
                  verbose=True,
                  initial_geometric_Align=True)

    if runcmd:
        # prepare command. On Windows one may need to run via WSL; the user can pass a command prefix
        cmd = reg.cmdline
        if _IS_WINDOWS and not cmd.startswith("wsl"):
            # if ANTs available natively on Windows this is fine; otherwise user should install WSL or adapt
            log.info("On Windows: ensure ANTs are available or run in WSL.")
        run_command(cmd, shell=True)
        # After successful run, create inverse transform if SimpleITK available
        try:
            outpfx = reg.inputs.output_transform_prefix
            ffoutAFF = outpfx + '0GenericAffine.mat'
            ffoutDEF = outpfx + '1Warp.nii.gz'
            ffDEFinv = outpfx + '1InverseWarp.nii.gz'
            # move files if necessary, but be conservative:
            if Path(ffoutAFF).exists():
                shutil.move(ffoutAFF, str(ffAFF))
                if sitk is not None:
                    trf = sitk.ReadTransform(str(ffAFF))
                    trfinv = trf.GetInverse()
                    sitk.WriteTransform(trfinv, str(ffAFFinv))
            if Path(ffoutDEF).exists():
                shutil.move(ffoutDEF, str(ffDEF))
            if Path(ffDEFinv).exists():
                shutil.move(ffDEFinv, str(ffDEFinv))
        except Exception as e:
            log.warning("Post-processing registration results failed: %s", e)

    return reg

def main(argv=None):
    p = argparse.ArgumentParser(description="Template registration helpers (open-access friendly)")
    p.add_argument("--template", "-t", type=Path, required=True, help="Template volume file (nii.gz)")
    p.add_argument("--input-dir", "-i", type=Path, required=True, help="Directory with subject files (will glob Denoised*.nii by default)")
    p.add_argument("--output-root", "-o", type=Path, default=None, help="Root output directory (defaults to each subject folder)")
    p.add_argument("--run", action="store_true", help="Actually run commands (otherwise dry-run)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    template = args.template.resolve()
    if not template.exists():
        log.error("Template not found: %s", template)
        return 2

    files = list(args.input_dir.rglob("Denoised*T1w.nii*"))
    if not files:
        log.warning("No files found in %s matching pattern", args.input_dir)
        return 0

    for i, ff in enumerate(files):
        log.info("Processing %d/%d: %s", i + 1, len(files), ff.name)
        subject_dir = ff.parent
        proj_path = (args.output_root.resolve() if args.output_root else subject_dir / "SANS")
        proj_path.mkdir(parents=True, exist_ok=True)
        prefixOUT = str(proj_path / "out_")
        ffOUT = proj_path / "out_volATLDeformed.nii.gz"
        try:
            reg = run_reg(ff, template, output_prefix=prefixOUT, output_warped_image=str(ffOUT), pnOUT=proj_path, runcmd=args.run)
            log.info("Prepared registration for %s", ff.name)
        except Exception as e:
            log.error("Failed registration for %s: %s", ff.name, e)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
