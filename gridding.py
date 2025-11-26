import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from typing import Tuple, Optional, Dict
import warnings


class Gridder:
    def __init__(
        self,
        grid_resolution: float = 1.0,
        search_radius: float = 1.5,
        anisotropic: bool = True
    ):
        self.grid_resolution = grid_resolution
        self.search_radius = search_radius
        self.anisotropic = anisotropic
    
    def create_grid(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_min -= self.grid_resolution
        x_max += self.grid_resolution
        y_min -= self.grid_resolution
        y_max += self.grid_resolution
        
        x_grid = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_grid = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_coords = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        return X_grid, Y_grid, grid_coords
    
    def compute_robust_median(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        grid_coords: np.ndarray,
        search_radius: Optional[float] = None
    ) -> np.ndarray:
        if len(x) == 0 or len(grid_coords) == 0:
            return np.full(len(grid_coords), np.nan)
        if search_radius is None:
            search_radius = self.search_radius
        grid_depths = np.full(len(grid_coords), np.nan)
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([x, y]))
        for i, grid_point in enumerate(grid_coords):
            indices = tree.query_ball_point(grid_point, search_radius)
            if len(indices) > 0:
                z_local = z[indices]
                grid_depths[i] = np.median(z_local)
        
        return grid_depths
    
    def compute_anisotropic_median(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        grid_coords: np.ndarray,
        search_radius: Optional[float] = None
    ) -> np.ndarray:
        if len(x) == 0 or len(grid_coords) == 0:
            return np.full(len(grid_coords), np.nan)
        if search_radius is None:
            search_radius = self.search_radius
        grid_depths = np.full(len(grid_coords), np.nan)
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([x, y]))
        for i, grid_point in enumerate(grid_coords):
            indices = tree.query_ball_point(grid_point, search_radius * 1.5)
            if len(indices) < 3:
                indices = tree.query_ball_point(grid_point, search_radius)
                if len(indices) > 0:
                    grid_depths[i] = np.median(z[indices])
                continue
            x_local = x[indices]
            y_local = y[indices]
            z_local = z[indices]
            if len(x_local) >= 3:
                A = np.column_stack([x_local, y_local, np.ones(len(x_local))])
                coeffs = np.linalg.lstsq(A, z_local, rcond=None)[0]
                grad = np.array([coeffs[0], coeffs[1]])
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-6:
                    isobath_dir = np.array([-grad[1], grad[0]]) / grad_norm
                    points_local = np.column_stack([x_local, y_local])
                    center = grid_point
                    vecs = points_local - center
                    along_isobath = np.abs(np.dot(vecs, isobath_dir))
                    across_isobath = np.abs(np.dot(vecs, grad / grad_norm))
                    weighted_dist = np.sqrt(
                        (along_isobath / (search_radius * 2))**2 +
                        (across_isobath / search_radius)**2
                    )
                    valid = weighted_dist <= 1.0
                    if np.sum(valid) > 0:
                        grid_depths[i] = np.median(z_local[valid])
                    else:
                        grid_depths[i] = np.median(z_local)
                else:
                    grid_depths[i] = np.median(z_local)
            else:
                grid_depths[i] = np.median(z_local)
        
        return grid_depths
    
    def grid_depths(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])
        
        X_grid, Y_grid, grid_coords = self.create_grid(x, y)
        
        if self.anisotropic:
            grid_depths = self.compute_anisotropic_median(x, y, d, grid_coords)
        else:
            grid_depths = self.compute_robust_median(x, y, d, grid_coords)
        
        D_grid = grid_depths.reshape(X_grid.shape)
        
        return X_grid, Y_grid, D_grid


class AccuracyMetrics:
    def __init__(self):
        pass
    
    def compute_metrics(
        self,
        d_pred: np.ndarray,
        d_true: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        if mask is not None:
            d_pred = d_pred[mask]
            d_true = d_true[mask]
        valid = np.isfinite(d_pred) & np.isfinite(d_true)
        d_pred = d_pred[valid]
        d_true = d_true[valid]
        if len(d_pred) == 0:
            return {
                'bias': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'nmad': np.nan,
                'p68': np.nan,
                'p95': np.nan,
                'n_valid': 0
            }
        residuals = d_pred - d_true
        bias = np.mean(residuals)
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((d_true - np.mean(d_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        nmad = 1.4826 * mad
        p68 = np.percentile(np.abs(residuals), 68)
        p95 = np.percentile(np.abs(residuals), 95)
        
        return {
            'bias': bias,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'nmad': nmad,
            'p68': p68,
            'p95': p95,
            'n_valid': len(d_pred)
        }
    
    def estimate_mean_water_surface(
        self,
        surface_tracks: Dict,
        time_window: Optional[float] = None
    ) -> float:
        z_surface_all = []
        for track_data in surface_tracks.values():
            z_track = track_data.get('z', np.array([]))
            if len(z_track) > 0:
                z_surface_all.extend(z_track)
        if len(z_surface_all) == 0:
            return 0.0
        z_surface_all = np.array(z_surface_all)
        z_mean = np.median(z_surface_all)
        
        return z_mean

