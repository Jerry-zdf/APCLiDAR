import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional
import warnings


class TrajectoryConsistency:
    def __init__(
        self,
        n_slices: int = 5,
        L_min: int = 3,
        w_d: float = 1.0,
        w_z: float = 2.0,
        w_kappa: float = 1.0,
        w_I: float = 0.5,
        w_rho: float = 0.1
    ):
        self.n_slices = n_slices
        self.L_min = L_min
        self.w_d = w_d
        self.w_z = w_z
        self.w_kappa = w_kappa
        self.w_I = w_I
        self.w_rho = w_rho
    
    def infer_flight_direction(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        if len(x) < 2:
            return np.array([1.0, 0.0])
        points = np.column_stack([x, y])
        pca = PCA(n_components=2)
        pca.fit(points)
        u = pca.components_[0]
        return u / np.linalg.norm(u)
    
    def partition_slices(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray
    ) -> List[Dict]:
        points = np.column_stack([x, y])
        projections = np.dot(points, u)
        proj_min, proj_max = projections.min(), projections.max()
        slice_width = (proj_max - proj_min) / self.n_slices
        slices = []
        for i in range(self.n_slices):
            proj_start = proj_min + i * slice_width
            proj_end = proj_min + (i + 1) * slice_width
            mask = (projections >= proj_start) & (projections < proj_end)
            if i == self.n_slices - 1:
                mask |= (projections == proj_end)
            
            indices = np.where(mask)[0]
            slices.append({
                'indices': indices,
                'projection_start': proj_start,
                'projection_end': proj_end
            })
        
        return slices
    
    def compute_cluster_centroids(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        labels: np.ndarray,
        intensity: Optional[np.ndarray] = None
    ) -> Dict:
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) == 0:
                continue
            
            x_cluster = x[mask]
            y_cluster = y[mask]
            z_cluster = z[mask]
            
            centroid = {
                'x': np.median(x_cluster),
                'y': np.median(y_cluster),
                'z': np.median(z_cluster),
                'n_points': np.sum(mask),
                'density': np.sum(mask) / (
                    (x_cluster.max() - x_cluster.min() + 1e-6) *
                    (y_cluster.max() - y_cluster.min() + 1e-6) *
                    (z_cluster.max() - z_cluster.min() + 1e-6)
                )
            }
            
            if intensity is not None:
                centroid['intensity'] = np.median(intensity[mask])
            else:
                centroid['intensity'] = 0.0
            
            centroids[label] = centroid
        
        return centroids
    
    def compute_matching_cost(
        self,
        c1: Dict,
        c2: Dict,
        v_z: float,
        delta_s: float = 1.0
    ) -> float:
        d_xy = np.sqrt(
            (c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2
        )
        delta_z = c2['z'] - c1['z']
        z_penalty = abs(delta_z - v_z * delta_s)
        kappa = 0.0
        delta_I = abs(c2['intensity'] - c1['intensity'])
        rho_reward = c1['density'] + c2['density']
        
        cost = (
            self.w_d * d_xy +
            self.w_z * z_penalty +
            self.w_kappa * kappa +
            self.w_I * delta_I -
            self.w_rho * rho_reward
        )
        
        return cost
    
    def match_slices(
        self,
        centroids1: Dict,
        centroids2: Dict,
        v_z: float = 0.0
    ) -> List[Tuple[int, int]]:
        if len(centroids1) == 0 or len(centroids2) == 0:
            return []
        labels1 = list(centroids1.keys())
        labels2 = list(centroids2.keys())
        cost_matrix = np.zeros((len(labels1), len(labels2)))
        for i, l1 in enumerate(labels1):
            for j, l2 in enumerate(labels2):
                cost_matrix[i, j] = self.compute_matching_cost(
                    centroids1[l1], centroids2[l2], v_z
                )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        cost_threshold = np.percentile(cost_matrix, 25)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < cost_threshold:
                matches.append((labels1[r], labels2[c]))
        
        return matches
    
    def build_tracks(
        self,
        slice_centroids: List[Dict],
        matches: List[List[Tuple[int, int]]]
    ) -> List[List[int]]:
        tracks = []
        graph = {}
        for slice_idx, match_list in enumerate(matches):
            for label1, label2 in match_list:
                key = (slice_idx, label1)
                if key not in graph:
                    graph[key] = []
                graph[key].append((slice_idx + 1, label2))
        all_nodes = set(graph.keys())
        all_targets = set()
        for targets in graph.values():
            all_targets.update(targets)
        starts = all_nodes - all_targets
        for start in starts:
            track = [start]
            current = start
            while current in graph and len(graph[current]) > 0:
                next_node = graph[current][0]
                track.append(next_node)
                current = next_node
            
            if len(track) >= self.L_min:
                tracks.append(track)
        
        return tracks
    
    def smooth_surface_tracks(
        self,
        tracks: List[List[int]],
        slice_centroids: List[Dict],
        slope_bound: float = 0.5,
        curvature_bound: float = 0.1
    ) -> Dict:
        smoothed_tracks = {}
        for track_idx, track in enumerate(tracks):
            if len(track) < 2:
                continue
            x_track = []
            y_track = []
            z_track = []
            for slice_idx, label in track:
                if slice_idx < len(slice_centroids) and label in slice_centroids[slice_idx]:
                    c = slice_centroids[slice_idx][label]
                    x_track.append(c['x'])
                    y_track.append(c['y'])
                    z_track.append(c['z'])
            if len(x_track) < 2:
                continue
            x_track = np.array(x_track)
            y_track = np.array(y_track)
            z_track = np.array(z_track)
            z_smooth = z_track.copy()
            for i in range(1, len(z_track) - 1):
                dz_ds = (z_track[i+1] - z_track[i-1]) / 2.0
                if abs(dz_ds) > slope_bound:
                    z_smooth[i] = (z_track[i-1] + z_track[i+1]) / 2.0
                d2z_ds2 = z_track[i+1] - 2*z_track[i] + z_track[i-1]
                if abs(d2z_ds2) > curvature_bound:
                    z_smooth[i] = (z_track[i-1] + z_track[i+1]) / 2.0
            
            smoothed_tracks[track_idx] = {
                'x': x_track,
                'y': y_track,
                'z': z_smooth,
                'original_z': z_track
            }
        
        return smoothed_tracks
    
    def regularize_bottom_tracks(
        self,
        tracks: List[List[int]],
        slice_centroids: List[Dict],
        surface_elevation: Optional[float] = None
    ) -> Dict:
        regularized_tracks = {}
        for track_idx, track in enumerate(tracks):
            if len(track) < 3:
                continue
            x_track = []
            y_track = []
            z_track = []
            for slice_idx, label in track:
                if slice_idx < len(slice_centroids) and label in slice_centroids[slice_idx]:
                    c = slice_centroids[slice_idx][label]
                    x_track.append(c['x'])
                    y_track.append(c['y'])
                    z_track.append(c['z'])
            if len(x_track) < 3:
                continue
            x_track = np.array(x_track)
            y_track = np.array(y_track)
            z_track = np.array(z_track)
            z_reg = z_track.copy()
            window = min(3, len(z_track) // 2)
            for i in range(len(z_track)):
                start = max(0, i - window)
                end = min(len(z_track), i + window + 1)
                x_local = x_track[start:end]
                y_local = y_track[start:end]
                z_local = z_track[start:end]
                if len(x_local) >= 3:
                    A = np.column_stack([x_local, y_local, np.ones(len(x_local))])
                    coeffs = np.linalg.lstsq(A, z_local, rcond=None)[0]
                    z_plane = coeffs[0] * x_track[i] + coeffs[1] * y_track[i] + coeffs[2]
                    z_reg[i] = 0.7 * z_track[i] + 0.3 * z_plane
                    if i > 0 and z_reg[i] > z_reg[i-1] + 0.1:
                        z_reg[i] = z_reg[i-1] - 0.01
                    if surface_elevation is not None and z_reg[i] > surface_elevation:
                        z_reg[i] = surface_elevation - 0.1
            
            regularized_tracks[track_idx] = {
                'x': x_track,
                'y': y_track,
                'z': z_reg,
                'original_z': z_track
            }
        
        return regularized_tracks

