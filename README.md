# Medical OCT Volume Viewer

Tool for visualizing and analyzing 512×512×512 uint8 OCT
volumetric data.

![Interface](screenshot.png)

## Features

### Core Visualization

-   3D volume navigation with slice-by-slice browsing
-   Maximum Intensity Projection (MIP) with adjustable depth
-   Dual view for comparing original slice and MIP

### Analysis Tools

-   Draggable circular ROI markers
-   Real-time metrics: max, median, Q75, variance, size
-   Histogram visualization for intensity distribution
-   3D region analysis across Z‑slices

### User Experience

-   Automatic dark/light theme detection
-   Multiple color palettes: gray, viridis, plasma, magma, inferno, jet
-   LRU caching and optional RAM loading
-   Responsive UI with realtime updates

## Quick Start

1.  Download `oct_viewer.exe` from the latest release.
2.  Run and open a `.dat` volume file.
3.  Use the Z‑slider to navigate slices.
4.  Adjust MIP depth.
5.  Drag ROI circles to measure regions.

## Developer Guide

### Architecture

-   MVC separation: data, visualization, control
-   Modular components (ImageDataManager, Analytics, Renderer)
-   Full type annotation
-   Structured logging

### Requirements

-   Python 3.8+
-   NumPy
-   PySide2
-   Matplotlib
-   darkdetect

### Installation

#### Using uv

``` bash
uv sync
```

#### Using pip

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Application

``` bash
uv run main.py
# or
python main.py
```

### Data Analysis Workflow

1.  Load a `.dat` volume.
2.  Navigate slices.
3.  Configure MIP depth.
4.  Place measurement circles.
5.  Review metrics and histograms.

### Generating Test Data

``` bash
uv run generator.py
```

### Rebuilding UI from Qt Designer

``` bash
uv run pyside2-uic ui/design.ui -o design_ui.py
```

## Technical Notes

-   Raw volume size: 512×512×512 uint8 (134,217,728 bytes)
-   Optional memmap mode for large files
-   LRU caching for pixmaps, MIP slices, and masks
-   Windows 7+ support via Python 3.8 + PySide2

## Architecture Overview

-   **ImageDataManager**: loading and memory management
-   **Analytics**: statistical processing
-   **CircleManager**: ROI tracking and interaction
-   **PixmapCache**: optimized rendering
-   **MetricsTable**: structured metric reporting

## Code Quality

-   ruff + pyright configuration
-   Full type hints
-   Structured logging
-   Robust error handling

## Building Executables

``` bash
uv run pyinstaller --onefile --windowed main.py
```

Built for scientific research and medical image analysis.
