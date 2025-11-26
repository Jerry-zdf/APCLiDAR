# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install package in development mode
pip install -e .
```

## Basic Usage

### Python API

```python
from bathymetry import BathymetryWorkflow
import laspy

# Load LAS file
las = laspy.read("your_strip.las")

# Initialize workflow
workflow = BathymetryWorkflow()

# Process
results = workflow.process_strip(
    las.x, las.y, las.z, las.intensity
)

# Get gridded depths
D_grid = results['stage6_gridding']['D_grid']
```

### Command Line

```bash
# Basic processing
python -m bathymetry.cli input.las -o output.asc

# With custom parameters
python -m bathymetry.cli input.las \
    --tile-size 256.0 \
    --grid-resolution 0.5 \
    --fdr-alpha 0.05 \
    -o output.asc

# With reference for accuracy
python -m bathymetry.cli input.las \
    --reference reference.las \
    -o output.asc
```

## Example Parameters

### High Resolution Processing
```python
workflow = BathymetryWorkflow(
    dx=0.5, dy=0.5,           # 0.5m voxels
    grid_resolution=0.5,      # 0.5m grid
    tile_size=256.0,          # Larger tiles
    fdr_alpha=0.05            # Stricter gating
)
```

### Fast Processing (Lower Resolution)
```python
workflow = BathymetryWorkflow(
    dx=2.0, dy=2.0,           # 2m voxels
    grid_resolution=2.0,       # 2m grid
    tile_size=512.0,           # Very large tiles
    fdr_alpha=0.15            # More lenient gating
)
```

### Deep Water (Emphasize Vertical Resolution)
```python
workflow = BathymetryWorkflow(
    a=0.3,                     # More weight on vertical
    beta_bottom=1.5,           # Larger bottom clusters
    minPts_bottom=20           # More points per cluster
)
```

### Shallow Water (Emphasize Planar Resolution)
```python
workflow = BathymetryWorkflow(
    a=0.7,                     # More weight on planar
    beta_surface=0.5,          # Smaller surface clusters
    minPts_surface=8           # Fewer points per cluster
)
```

## Output Format

The workflow returns a dictionary with:

- `stage1_preprocessing`: Preprocessed points, water surface elevation
- `stage2_voxelization`: FDR-gated significant photons
- `stage3_clustering`: Cluster labels for each layer
- `stage4_trajectory`: Surface and bottom tracks
- `stage5_refraction`: Refraction-corrected bottom points
- `stage6_gridding`: Gridded depths (X_grid, Y_grid, D_grid)
- `accuracy_metrics`: Accuracy statistics (if reference provided)

## Saving Results

### ASCII Grid
```python
# Save as ASCII grid
X_grid = results['stage6_gridding']['X_grid']
Y_grid = results['stage6_gridding']['Y_grid']
D_grid = results['stage6_gridding']['D_grid']

# Use provided CLI or implement custom save function
```

### NumPy Array
```python
np.savez("bathymetry.npz", 
    X=X_grid, Y=Y_grid, D=D_grid
)
```

### GeoTIFF (requires rasterio)
```python
import rasterio
from rasterio.transform import from_bounds

X_grid = results['stage6_gridding']['X_grid']
Y_grid = results['stage6_gridding']['Y_grid']
D_grid = results['stage6_gridding']['D_grid']

transform = from_bounds(
    X_grid.min(), Y_grid.min(),
    X_grid.max(), Y_grid.max(),
    D_grid.shape[1], D_grid.shape[0]
)

with rasterio.open("bathymetry.tif", "w",
    driver="GTiff",
    height=D_grid.shape[0],
    width=D_grid.shape[1],
    count=1,
    dtype=D_grid.dtype,
    crs="EPSG:4326",  # Adjust to your CRS
    transform=transform,
    nodata=-9999
) as dst:
    dst.write(D_grid, 1)
```

## Troubleshooting

### Memory Issues
- Reduce `tile_size` to process smaller chunks
- Increase `dx`, `dy` to use larger voxels
- Process strips separately and mosaic results

### No Bottom Detected
- Check `fdr_alpha` - may be too strict
- Adjust `minPts_bottom` - may be too high
- Verify input data has sufficient bottom returns

### Poor Accuracy
- Verify water surface estimation is correct
- Check refraction correction parameters
- Ensure reference data is properly aligned

### Slow Processing
- Reduce `n_slices` for trajectory stage
- Use larger voxels (`dx`, `dy`)
- Disable anisotropic gridding if not needed

