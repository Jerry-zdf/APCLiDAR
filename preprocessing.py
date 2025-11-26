import numpy as np
from scipy import stats
from typing import Tuple, Optional
import warnings


class Preprocessor:
    def __init__(
        self,
        tile_size: float = 128.0,
        overlap: float = 0.5,
        zscore_threshold: float = 3.0
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.zscore_threshold = zscore_threshold
    
    def partition_tiles(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> list:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        step = self.tile_size * (1 - self.overlap)
        tiles = []
        
        x_starts = np.arange(x_min, x_max, step)
        y_starts = np.arange(y_min, y_max, step)
        
        for x_start in x_starts:
            for y_start in y_starts:
                x_end = x_start + self.tile_size
                y_end = y_start + self.tile_size
                mask = (
                    (x >= x_start) & (x < x_end) &
                    (y >= y_start) & (y < y_end)
                )
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    tiles.append({
                        'x_min': x_start,
                        'x_max': x_end,
                        'y_min': y_start,
                        'y_max': y_end,
                        'indices': indices
                    })
        
        return tiles
    
    def remove_outliers(
        self,
        z: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        if method == 'zscore':
            z_median = np.median(z)
            mad = np.median(np.abs(z - z_median))
            if mad == 0:
                return np.ones(len(z), dtype=bool)
            z_scores = 0.6745 * (z - z_median) / mad
            return np.abs(z_scores) < self.zscore_threshold
        elif method == 'iqr':
            q1, q3 = np.percentile(z, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (z >= lower) & (z <= upper)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate_water_surface(
        self,
        z: np.ndarray,
        adaptive_bin: bool = True
    ) -> Tuple[float, float]:
        n = len(z)
        if n < 10:
            return np.median(z), np.std(z) * 0.1
        
        if adaptive_bin:
            iqr = np.percentile(z, 75) - np.percentile(z, 25)
            if iqr == 0:
                iqr = np.std(z)
            h_z = 2 * iqr / (n ** (1/3))
            h_z = max(h_z, 0.01)
        else:
            h_z = np.std(z) * 0.1
        z_min, z_max = z.min(), z.max()
        bins = np.arange(z_min, z_max + h_z, h_z)
        counts, bin_edges = np.histogram(z, bins=bins)
        modal_idx = np.argmax(counts)
        z_surf = (bin_edges[modal_idx] + bin_edges[modal_idx + 1]) / 2
        
        return z_surf, h_z
    
    def compute_depths(
        self,
        z: np.ndarray,
        z_surf: float
    ) -> np.ndarray:
        return z_surf - z
    
    def process_tile(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        intensity: Optional[np.ndarray] = None
    ) -> dict:
        valid_mask = self.remove_outliers(z)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        
        if intensity is not None:
            intensity_valid = intensity[valid_mask]
        else:
            intensity_valid = None
        
        if len(z_valid) == 0:
            warnings.warn("No valid points after outlier removal")
            return {
                'x': x_valid,
                'y': y_valid,
                'z': z_valid,
                'd': np.array([]),
                'z_surf': np.nan,
                'h_z': np.nan,
                'intensity': intensity_valid,
                'valid_mask': valid_mask
            }
        z_surf, h_z = self.estimate_water_surface(z_valid)
        d = self.compute_depths(z_valid, z_surf)
        
        return {
            'x': x_valid,
            'y': y_valid,
            'z': z_valid,
            'd': d,
            'z_surf': z_surf,
            'h_z': h_z,
            'intensity': intensity_valid,
            'valid_mask': valid_mask
        }

