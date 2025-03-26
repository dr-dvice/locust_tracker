# Locust Tracker

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/github/license/dr-dvice/locust_tracker)
![Issues](https://img.shields.io/github/issues/dr-dvice/locust_tracker)
![Last Commit](https://img.shields.io/github/last-commit/dr-dvice/locust_tracker)

## 📌 Description
Locust Tracker is a Python-based tool for analyzing coordinates from DeepLabCut-analyzed videos. This is based on the behavioral arena setup as described in this 1993 paper [here](https://www.jstor.org/stable/49916).  Locust Tracker calculates detailed movement and positional metrics to analyze behavior in various experimental conditions.
![Arena Boundary Coordinates](labelled_arena.jpg)
The coordinates of the wall boundaries, as shown in this picture, can be edited in the code to suit your setup.

## 🚀 Features
- Downstream coordinate data processing from DeepLabCut
- In-depth analysis of movement, positional, and rotational statistics
- Can quickly analyze hundreds of videos
- Command line options allow in-depth control of thresholds and arena positions
- Easy to parse output in .csv format

## 🛠 Installation
Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed. Then follow these steps to set up the environment:

```bash
# Clone the repository
git clone https://github.com/dr-dvice/locust_tracker.git
cd locust_tracker

# Create a conda environment with the conda-forge and bioconda channels active
conda create -n locust_tracker -c conda-forge -c bioconda python=3.11

# Activate the environment
conda activate locust_tracker

# Install dependencies
conda install --file locustrequirements.txt
```

## 📦 Usage
To analyze videos using Locust Tracker:

```bash
python locust_tracker.py data_directory results_folder [options]
```

**Positional Arguments:**
- `data_directory`: Path to the directory containing `.h5` files with tracking coordinates.
- `results_folder`: Path where result files will be saved.

**Optional Arguments:**
- `-s, --stimulus`: Specify the side of the arena where the stimulus cage is located. Options: `left`, `right`, `l`, or `r` (default: `right`).
- `-r, --regex`: Regex pattern to extract trial numbers from video filenames (default: `Trial\s+(\d+)`).
- `-l, --length, --videolength`: Expected trial video length in seconds. Videos shorter than this will be discarded (default: `600`).
- `-p, --plot`: Plot movement for a specific trial from the data directory without analyzing, to determine movement thresholds.
- `-m, --movethresh`: Threshold for detected movement in cm per frame (default: `0.025`).
- `-a, --anglethresh`: Threshold for angular changes in degrees per frame (default: `1`).
- `-f, --fps`: Frames per second used for analysis (default: `2`).
- `-h5, --saveh5`: Save updated `.h5` data with calculated movement, distances, and angles.
- `-d, --debug`: Enable debug mode with additional outputs and visualizations.

## 🧪 Examples
Analyze data with default options:
```bash
python locust_tracker.py ./data ./results
```

Specify a left-side stimulus and a custom video length:
```bash
python locust_tracker.py ./data ./results -s left -l 300
```

Plot movement for a specific trial to determine thresholds:
```bash
python locust_tracker.py ./data ./results -p Trial_01.h5
```

## 📜 License
This project is licensed under the MIT License. See `LICENSE` for details.

## 📧 Contact
For questions or feedback, reach out via GitHub issues or email.

---
Created by David Bellini, PhD Candidate at Baylor College of Medicine.



