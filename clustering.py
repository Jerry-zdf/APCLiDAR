import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict
import warnings


class LayerWiseClustering:
    def __init__(
        self,
        beta_surface: float = 0.6,
        beta_water: float = 0.9,
        beta_bottom: float = 1.2,
        a: float = 0.5,
        minPts_surface: int = 10,
        minPts_water: int = 16,
        minPts_bottom: int = 12
    ):
        self.beta_surface = beta_surface
        self.beta_water = beta_water
        self.beta_bottom = beta_bottom
        self.a = a
        self.minPts_surface = minPts_surface
        self.minPts_water = minPts_water
        self.minPts_bottom = minPts_bottom
    
    def estimate_areal_density(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0
    ) -> np.ndarray:
        if len(x) == 0:
            return np.array([])
        points = np.column_stack([x, y])
        n_neighbors = min(10, len(x) - 1)
        if n_neighbors < 1:
            return np.full(len(x), 1.0)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        k_dist = distances[:, -1]
        area = np.pi * k_dist ** 2
        area = np.maximum(area, dx * dy)
        rho_xy = n_neighbors / area
        return rho_xy
    
    def compute_adaptive_radius(
        self,
        h_z: float,
        rho_xy: np.ndarray,
        beta: float
    ) -> np.ndarray:
        delta_xy = np.sqrt(1.0 / np.maximum(rho_xy, 0.01))
        r = beta * (h_z ** self.a) * (delta_xy ** (1 - self.a))
        return r
    
    def identify_surface_seeds(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        d: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        if len(x) == 0:
            return np.array([], dtype=bool)
        surface_candidates = d < np.percentile(d, 10)
        if np.sum(surface_candidates) < 3:
            return np.zeros(len(x), dtype=bool)
        x_surf = x[surface_candidates]
        y_surf = y[surface_candidates]
        z_surf = z[surface_candidates]
        best_inliers = np.zeros(len(x_surf), dtype=bool)
        best_score = 0
        for _ in range(min(n_samples, len(x_surf) // 3)):
            idx = np.random.choice(len(x_surf), 3, replace=False)
            p1 = np.array([x_surf[idx[0]], y_surf[idx[0]], z_surf[idx[0]]])
            p2 = np.array([x_surf[idx[1]], y_surf[idx[1]], z_surf[idx[1]]])
            p3 = np.array([x_surf[idx[2]], y_surf[idx[2]], z_surf[idx[2]]])
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            points = np.column_stack([x_surf, y_surf, z_surf])
            distances = np.abs(np.dot(points - p1, normal))
            threshold = 0.1
            inliers = distances < threshold
            score = np.sum(inliers)
            if score > best_score:
                best_score = score
                best_inliers = inliers
        surface_mask = np.zeros(len(x), dtype=bool)
        surface_mask[np.where(surface_candidates)[0][best_inliers]] = True
        
        return surface_mask
    
    def identify_bottom_seeds(
        self,
        z: np.ndarray,
        d: np.ndarray,
        window_size: float = 2.0
    ) -> np.ndarray:
        if len(z) == 0:
            return np.array([], dtype=bool)
        subsurface = d > np.percentile(d[d > 0], 20)
        if np.sum(subsurface) < 10:
            return np.zeros(len(z), dtype=bool)
        z_subsurface = z[subsurface]
        d_subsurface = d[subsurface]
        z_sorted_idx = np.argsort(z_subsurface)
        z_sorted = z_subsurface[z_sorted_idx]
        d_sorted = d_subsurface[z_sorted_idx]
        window = int(len(z_sorted) * 0.1)
        if window < 3:
            d_detrended = d_sorted
        else:
            d_trend = np.convolve(
                d_sorted, np.ones(window) / window, mode='same'
            )
            d_detrended = d_sorted - d_trend
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(d_detrended, height=np.percentile(d_detrended, 75))
        bottom_mask = np.zeros(len(z), dtype=bool)
        subsurface_indices = np.where(subsurface)[0]
        if len(peaks) > 0:
            peak_indices = subsurface_indices[z_sorted_idx[peaks]]
            bottom_mask[peak_indices] = True
        
        return bottom_mask
    
    def cluster_layer(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        radius: np.ndarray,
        minPts: int,
        layer_name: str = "unknown"
    ) -> Tuple[np.ndarray, Dict]:
        if len(x) == 0:
            return np.array([]), {}
        points = np.column_stack([x, y, z])
        eps = np.median(radius)
        if eps <= 0:
            eps = 0.1
        if len(x) > 0:
            rho_xy = self.estimate_areal_density(x, y)
            density_factor = np.median(rho_xy) / 100.0
            minPts_adjusted = int(minPts * (1 + density_factor))
        else:
            minPts_adjusted = minPts
        clustering = DBSCAN(eps=eps, min_samples=minPts_adjusted)
        labels = clustering.fit_predict(points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        cluster_info = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'minPts': minPts_adjusted
        }
        
        return labels, cluster_info
    
    def process_layers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        d: np.ndarray,
        h_z: float
    ) -> Dict:
        results = {}
        rho_xy = self.estimate_areal_density(x, y)
        surface_mask = self.identify_surface_seeds(x, y, z, d)
        if np.sum(surface_mask) > 0:
            x_surf = x[surface_mask]
            y_surf = y[surface_mask]
            z_surf = z[surface_mask]
            rho_surf = rho_xy[surface_mask]
            radius_surf = self.compute_adaptive_radius(
                h_z, rho_surf, self.beta_surface
            )
            labels_surf, info_surf = self.cluster_layer(
                x_surf, y_surf, z_surf, radius_surf,
                self.minPts_surface, "surface"
            )
            results['surface'] = {
                'mask': surface_mask,
                'labels': labels_surf,
                'info': info_surf
            }
        else:
            results['surface'] = {
                'mask': surface_mask,
                'labels': np.array([]),
                'info': {}
            }
        bottom_mask = self.identify_bottom_seeds(z, d)
        if np.sum(bottom_mask) > 0:
            x_bottom = x[bottom_mask]
            y_bottom = y[bottom_mask]
            z_bottom = z[bottom_mask]
            rho_bottom = rho_xy[bottom_mask]
            radius_bottom = self.compute_adaptive_radius(
                h_z, rho_bottom, self.beta_bottom
            )
            labels_bottom, info_bottom = self.cluster_layer(
                x_bottom, y_bottom, z_bottom, radius_bottom,
                self.minPts_bottom, "bottom"
            )
            results['bottom'] = {
                'mask': bottom_mask,
                'labels': labels_bottom,
                'info': info_bottom
            }
        else:
            results['bottom'] = {
                'mask': bottom_mask,
                'labels': np.array([]),
                'info': {}
            }
        water_mask = ~(surface_mask | bottom_mask)
        if np.sum(water_mask) > 0:
            x_water = x[water_mask]
            y_water = y[water_mask]
            z_water = z[water_mask]
            rho_water = rho_xy[water_mask]
            radius_water = self.compute_adaptive_radius(
                h_z, rho_water, self.beta_water
            )
            labels_water, info_water = self.cluster_layer(
                x_water, y_water, z_water, radius_water,
                self.minPts_water, "water"
            )
            results['water'] = {
                'mask': water_mask,
                'labels': labels_water,
                'info': info_water
            }
        else:
            results['water'] = {
                'mask': water_mask,
                'labels': np.array([]),
                'info': {}
            }
        
        return results

