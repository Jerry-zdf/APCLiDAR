import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional
import warnings


class Voxelizer:
    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: Optional[float] = None
    ):
        self.dx = dx
        self.dy = dy
        self.dz = dz
    
    def voxelize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        dz: float
    ) -> Tuple[np.ndarray, dict]:
        if len(x) == 0:
            return np.array([]), {}
        x_min, y_min, z_min = x.min(), y.min(), z.min()
        i = ((x - x_min) / self.dx).astype(int)
        j = ((y - y_min) / self.dy).astype(int)
        k = ((z - z_min) / dz).astype(int)
        voxel_keys = np.column_stack([i, j, k])
        unique_voxels, counts = np.unique(voxel_keys, axis=0, return_counts=True)
        voxel_centers_x = x_min + (unique_voxels[:, 0] + 0.5) * self.dx
        voxel_centers_y = y_min + (unique_voxels[:, 1] + 0.5) * self.dy
        voxel_centers_z = z_min + (unique_voxels[:, 2] + 0.5) * dz
        
        voxel_info = {
            'keys': unique_voxels,
            'counts': counts,
            'centers_x': voxel_centers_x,
            'centers_y': voxel_centers_y,
            'centers_z': voxel_centers_z,
            'x_min': x_min,
            'y_min': y_min,
            'z_min': z_min,
            'dx': self.dx,
            'dy': self.dy,
            'dz': dz
        }
        
        return voxel_keys, voxel_info


class BackgroundEstimator:
    def __init__(
        self,
        polynomial_order: int = 2,
        kernel_bandwidth: float = 10.0,
        mad_threshold: float = 3.0
    ):
        self.polynomial_order = polynomial_order
        self.kernel_bandwidth = kernel_bandwidth
        self.mad_threshold = mad_threshold
    
    def mask_surface_bottom(
        self,
        z: np.ndarray,
        counts: np.ndarray
    ) -> np.ndarray:
        z_unique, inverse = np.unique(z, return_inverse=True)
        hist_counts = np.bincount(inverse, weights=counts)
        median_count = np.median(hist_counts)
        mad = np.median(np.abs(hist_counts - median_count))
        if mad == 0:
            return np.ones(len(z), dtype=bool)
        z_scores = 0.6745 * np.abs(hist_counts - median_count) / mad
        peak_mask = z_scores > self.mad_threshold
        peak_depths = z_unique[peak_mask]
        if len(peak_depths) == 0:
            return np.ones(len(z), dtype=bool)
        background_mask = np.ones(len(z), dtype=bool)
        for peak_z in peak_depths:
            depth_dist = np.abs(z - peak_z)
            background_mask &= (depth_dist > 1.0)
        
        return background_mask
    
    def estimate_background(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        counts: np.ndarray,
        voxel_volume: float
    ) -> np.ndarray:
        if len(x) == 0:
            return np.array([])
        background_mask = self.mask_surface_bottom(z, counts)
        if np.sum(background_mask) < 10:
            background_mask = np.ones(len(x), dtype=bool)
        x_bg = x[background_mask]
        y_bg = y[background_mask]
        z_bg = z[background_mask]
        counts_bg = counts[background_mask]
        if len(x_bg) > 0:
            features = np.column_stack([x_bg, y_bg, z_bg])
            poly = PolynomialFeatures(degree=self.polynomial_order)
            X_poly = poly.fit_transform(features)
            rates_bg = counts_bg / voxel_volume
            reg = LinearRegression()
            reg.fit(X_poly, rates_bg)
            features_all = np.column_stack([x, y, z])
            X_poly_all = poly.transform(features_all)
            lambda_bg = reg.predict(X_poly_all)
            lambda_bg = np.maximum(lambda_bg, 0.0)
        else:
            lambda_bg = np.full(len(x), np.mean(counts) / voxel_volume)
        if self.kernel_bandwidth > 0:
            lambda_bg = self._kernel_smooth_xy(x, y, lambda_bg)
        
        return lambda_bg
    
    def _kernel_smooth_xy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray
    ) -> np.ndarray:
        if len(x) < 2:
            return values
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        grid_res = self.kernel_bandwidth / 2
        x_grid = np.arange(x_min, x_max + grid_res, grid_res)
        y_grid = np.arange(y_min, y_max + grid_res, grid_res)
        from scipy.interpolate import griddata
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid_values = griddata(
            (x, y), values, (grid_x, grid_y),
            method='linear', fill_value=np.nan
        )
        sigma = self.kernel_bandwidth / grid_res
        grid_values_smooth = gaussian_filter(
            grid_values, sigma=sigma, mode='constant', cval=np.nanmean(grid_values)
        )
        smoothed = griddata(
            (grid_x.ravel(), grid_y.ravel()), grid_values_smooth.ravel(),
            (x, y), method='linear', fill_value=np.nanmean(values)
        )
        
        return smoothed


class FDRGating:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
    
    def compute_poisson_pvalues(
        self,
        counts: np.ndarray,
        mu: np.ndarray
    ) -> np.ndarray:
        p_values = np.zeros(len(counts))
        for i in range(len(counts)):
            k = counts[i]
            mu_i = mu[i]
            if mu_i <= 0:
                p_values[i] = 0.0
            elif k == 0:
                p_values[i] = 1.0
            else:
                p_values[i] = 1.0 - stats.poisson.cdf(k - 1, mu_i)
        
        return p_values
    
    def benjamini_hochberg(
        self,
        p_values: np.ndarray
    ) -> np.ndarray:
        if len(p_values) == 0:
            return np.array([], dtype=bool)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        m = len(p_values)
        bh_threshold = (np.arange(1, m + 1) / m) * self.alpha
        significant = sorted_p <= bh_threshold
        if np.any(significant):
            k_max = np.where(significant)[0][-1]
            significant_mask = np.zeros(m, dtype=bool)
            significant_mask[sorted_indices[:k_max + 1]] = True
        else:
            significant_mask = np.zeros(m, dtype=bool)
        
        return significant_mask
    
    def gate_photons(
        self,
        voxel_info: dict,
        lambda_bg: np.ndarray
    ) -> np.ndarray:
        counts = voxel_info['counts']
        voxel_volume = (
            voxel_info['dx'] * voxel_info['dy'] * voxel_info['dz']
        )
        mu = lambda_bg * voxel_volume
        p_values = self.compute_poisson_pvalues(counts, mu)
        significant_mask = self.benjamini_hochberg(p_values)
        
        return significant_mask

