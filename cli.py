import argparse
import sys
import numpy as np
import laspy
from pathlib import Path

from .workflow import BathymetryWorkflow


def load_las_file(filepath: str) -> dict:
    try:
        las = laspy.read(filepath)
        return {
            'x': np.array(las.x),
            'y': np.array(las.y),
            'z': np.array(las.z),
            'intensity': np.array(las.intensity) if hasattr(las, 'intensity') else None
        }
    except Exception as e:
        print(f"Error loading LAS file: {e}", file=sys.stderr)
        sys.exit(1)


def save_grid_ascii(X_grid, Y_grid, D_grid, output_file: str):
    with open(output_file, 'w') as f:
        f.write(f"ncols {X_grid.shape[1]}\n")
        f.write(f"nrows {X_grid.shape[0]}\n")
        f.write(f"xllcorner {X_grid.min()}\n")
        f.write(f"yllcorner {Y_grid.min()}\n")
        f.write(f"cellsize {X_grid[0, 1] - X_grid[0, 0]}\n")
        f.write(f"NODATA_value -9999\n")
        for row in reversed(D_grid):
            f.write(" ".join([
                f"{val:.3f}" if np.isfinite(val) else "-9999"
                for val in row
            ]) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process bathymetry LAS files through six-stage workflow"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input LAS file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="bathymetry_grid.asc",
        help="Output grid file (default: bathymetry_grid.asc)"
    )
    
    parser.add_argument(
        "--tile-size",
        type=float,
        default=128.0,
        help="Processing tile size in meters (default: 128.0)"
    )
    
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=1.0,
        help="Grid cell size in meters (default: 1.0)"
    )
    
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.10,
        help="FDR control level (default: 0.10)"
    )
    
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference LAS file for accuracy assessment (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    
    args = parser.parse_args()
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    data = load_las_file(args.input_file)
    reference_depths = None
    if args.reference:
        ref_data = load_las_file(args.reference)
        reference_depths = ref_data['z']
    workflow = BathymetryWorkflow(
        tile_size=args.tile_size,
        grid_resolution=args.grid_resolution,
        fdr_alpha=args.fdr_alpha
    )
    results = workflow.process_strip(
        data['x'],
        data['y'],
        data['z'],
        data['intensity'],
        reference_depths
    )
    if len(results['stage6_gridding']['X_grid']) > 0:
        save_grid_ascii(
            results['stage6_gridding']['X_grid'],
            results['stage6_gridding']['Y_grid'],
            results['stage6_gridding']['D_grid'],
            args.output
        )
        if args.verbose:
            D_grid = results['stage6_gridding']['D_grid']
            print(f"Grid size: {D_grid.shape[0]} x {D_grid.shape[1]} cells")
            print(f"Valid cells: {np.sum(np.isfinite(D_grid))}")
            print(f"Depth range: {np.nanmin(D_grid):.2f} to {np.nanmax(D_grid):.2f} m")
            if results['accuracy_metrics']:
                metrics = results['accuracy_metrics']
                print(f"RMSE: {metrics['rmse']:.3f} m, R²: {metrics['r2']:.3f}")
    else:
        print("Warning: No valid grid data to save", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

