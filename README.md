# OCT Image Viewer

A simple viewer for `.dat` images from an OCT scanner, built with Python and PySide6.

## Requirements

- Python 3.10+  
- [PySide6](https://pypi.org/project/PySide6/)
- [uv](https://docs.astral.sh/uv/) - python package manager
- NumPy

## Setup and use

1. Sync dependencies using uv:

```bash
uv sync
```

2. Run:

```bash
uv run main.py
```

3. Use:

Use menu bar to open .dat file.

## Examples generation:

You can generate an example image using numpy, like:

```python
import numpy as np

data = np.linspace(0, 255, 512*512, dtype=np.uint8).reshape((512, 512))
data.tofile("images/example.dat")
```

...and running it:

```bash
uv run generator.py
```

...or you can use my built-in generator:

```bash
uv run dat_generator.py
```

## Notes
- Make sure .dat files are 512×512 and uint8.
- The viewer uses QGraphicsView for scalable display.
