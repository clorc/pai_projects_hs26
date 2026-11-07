# Task 2

Run these PowerShell commands from the directory containing this README. The checker uses Python 3.8.5, PyTorch 2.0.1 on CPU, and the supplied MAP checkpoint.

### Build once

```powershell
docker build -t task2 .
```

### Run after editing the solution

Mount the current source files into the image. This skips the build entirely for Python edits. Use a new output directory for each run so the checker can write its plots and submission file.

```powershell
$out = New-Item -ItemType Directory -Path ("results-" + [guid]::NewGuid().ToString("N"))
docker run --rm -v "${PWD}/solution.py:/code/solution.py:ro" -v "${PWD}/util.py:/code/util.py:ro" -v "$($out.FullName):/results" task2
```

The checker prints its public score and saves `results_check.byte` in that directory. Rebuild if you edit the Dockerfile, environment, or bundled data. A plain run of `task2` uses the source copied at build time.

### Project layout

- `solution.py` contains the active training, SWAG inference, calibration, and evaluation code.
- `util.py` contains the handout's scoring and plotting helpers.
- `archive/` holds earlier solutions, experiment plots, and checker runs. The verified **0.840** submission is in `archive/checker-results/checker-results-7df007a4/results_check.byte`.
