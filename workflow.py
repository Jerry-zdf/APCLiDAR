import numpy as np
from typing import Dict, Optional, Tuple
import warnings

from .preprocessing import Preprocessor
from .voxelization import Voxelizer, BackgroundEstimator, FDRGating
from .clustering import LayerWiseClustering
from .trajectory import TrajectoryConsistency
from .refraction import RefractionCorrector
from .gridding import Gridder, AccuracyMetrics


class BathymetryWorkflow:
    def __init__(
        self,
        tile_size: float = 128.0,
        overlap: float = 0.5,
        zscore_threshold: float = 3.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: Optional[float] = None,
        polynomial_order: int = 2,
        kernel_bandwidth: float = 10.0,
        fdr_alpha: float = 0.10,
        beta_surface: float = 0.6,
        beta_water: float = 0.9,
        beta_bottom: float = 1.2,
        a: float = 0.5,
        minPts_surface: int = 10,
        minPts_water: int = 16,
        minPts_bottom: int = 12,
        n_slices: int = 5,
        L_min: int = 3,
        w_d: float = 1.0,
        w_z: float = 2.0,
        w_kappa: float = 1.0,
        w_I: float = 0.5,
        w_rho: float = 0.1,
        n_air: float = 1.0003,
        n_water: float = 1.33,
        wave_uncertainty_threshold: float = 0.05,
        grid_resolution: float = 1.0,
        search_radius: float = 1.5,
        anisotropic: bool = True
    ):
        self.preprocessor = Preprocessor(
            tile_size=tile_size,
            overlap=overlap,
            zscore_threshold=zscore_threshold
        )
        
        self.voxelizer = Voxelizer(dx=dx, dy=dy, dz=dz)
        self.background_estimator = BackgroundEstimator(
            polynomial_order=polynomial_order,
            kernel_bandwidth=kernel_bandwidth
        )
        self.fdr_gating = FDRGating(alpha=fdr_alpha)
        
        self.clustering = LayerWiseClustering(
            beta_surface=beta_surface,
            beta_water=beta_water,
            beta_bottom=beta_bottom,
            a=a,
            minPts_surface=minPts_surface,
            minPts_water=minPts_water,
            minPts_bottom=minPts_bottom
        )
        
        self.trajectory = TrajectoryConsistency(
            n_slices=n_slices,
            L_min=L_min,
            w_d=w_d,
            w_z=w_z,
            w_kappa=w_kappa,
            w_I=w_I,
            w_rho=w_rho
        )
        
        self.refraction = RefractionCorrector(
            n_air=n_air,
            n_water=n_water,
            wave_uncertainty_threshold=wave_uncertainty_threshold
        )
        
        self.gridder = Gridder(
            grid_resolution=grid_resolution,
            search_radius=search_radius,
            anisotropic=anisotropic
        )
        
        self.metrics = AccuracyMetrics()
    
    def process_strip(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        reference_depths: Optional[np.ndarray] = None
    ) -> Dict:
        results = {
            'stage1_preprocessing': {},
            'stage2_voxelization': {},
            'stage3_clustering': {},
            'stage4_trajectory': {},
            'stage5_refraction': {},
            'stage6_gridding': {},
            'accuracy_metrics': {}
        }
        tiles = self.preprocessor.partition_tiles(x, y)
        
        processed_tiles = []
        for tile in tiles:
            tile_data = self.preprocessor.process_tile(
                x[tile['indices']],
                y[tile['indices']],
                z[tile['indices']],
                intensity[tile['indices']] if intensity is not None else None
            )
            processed_tiles.append(tile_data)
        x_processed = np.concatenate([t['x'] for t in processed_tiles])
        y_processed = np.concatenate([t['y'] for t in processed_tiles])
        z_processed = np.concatenate([t['z'] for t in processed_tiles])
        d_processed = np.concatenate([t['d'] for t in processed_tiles])
        h_z = np.median([t['h_z'] for t in processed_tiles])
        z_surf_mean = np.median([t['z_surf'] for t in processed_tiles])
        
        intensity_processed = None
        if intensity is not None:
            intensity_processed = np.concatenate([
                t['intensity'] for t in processed_tiles
                if t['intensity'] is not None
            ])
        
        results['stage1_preprocessing'] = {
            'x': x_processed,
            'y': y_processed,
            'z': z_processed,
            'd': d_processed,
            'z_surf_mean': z_surf_mean,
            'h_z': h_z,
            'n_tiles': len(tiles)
        }
        dz = h_z if self.voxelizer.dz is None else self.voxelizer.dz
        
        voxel_keys, voxel_info = self.voxelizer.voxelize(
            x_processed, y_processed,             z_processed, dz
        )
        lambda_bg = self.background_estimator.estimate_background(
            voxel_info['centers_x'],
            voxel_info['centers_y'],
            voxel_info['centers_z'],
            voxel_info['counts'],
            voxel_info['dx'] * voxel_info['dy'] * voxel_info['dz']
        )
        significant_mask = self.fdr_gating.gate_photons(voxel_info, lambda_bg)
        significant_voxels = voxel_info['keys'][significant_mask]
        point_mask = np.zeros(len(x_processed), dtype=bool)
        significant_set = set(tuple(vk) for vk in significant_voxels)
        
        for i, voxel_key in enumerate(voxel_keys):
            if tuple(voxel_key) in significant_set:
                point_mask[i] = True
        
        x_star = x_processed[point_mask]
        y_star = y_processed[point_mask]
        z_star = z_processed[point_mask]
        d_star = d_processed[point_mask]
        
        if intensity_processed is not None:
            intensity_star = intensity_processed[point_mask]
        else:
            intensity_star = None
        
        results['stage2_voxelization'] = {
            'x': x_star,
            'y': y_star,
            'z': z_star,
            'd': d_star,
            'intensity': intensity_star,
            'n_significant': len(x_star)
        }
        clustering_results = self.clustering.process_layers(
            x_star, y_star, z_star, d_star, h_z
        )
        
        results['stage3_clustering'] = clustering_results
        u = self.trajectory.infer_flight_direction(x_star, y_star)
        slices = self.trajectory.partition_slices(x_star, y_star, u)
        surface_tracks_data = {}
        bottom_tracks_data = {}
        if 'surface' in clustering_results:
            surface_mask = clustering_results['surface']['mask']
            surface_labels = clustering_results['surface']['labels']
            
            if np.sum(surface_mask) > 0:
                x_surf = x_star[surface_mask]
                y_surf = y_star[surface_mask]
                z_surf = z_star[surface_mask]
                slice_centroids = []
                for slice_data in slices:
                    slice_mask = np.isin(
                        np.arange(len(x_surf)),
                        np.where(surface_mask)[0][slice_data['indices']]
                    )
                    if np.sum(slice_mask) > 0:
                        labels_slice = surface_labels[slice_mask]
                        centroids = self.trajectory.compute_cluster_centroids(
                            x_surf[slice_mask],
                            y_surf[slice_mask],
                            z_surf[slice_mask],
                            labels_slice
                        )
                        slice_centroids.append(centroids)
                    else:
                        slice_centroids.append({})
                matches = []
                v_z = 0.0
                for i in range(len(slice_centroids) - 1):
                    match = self.trajectory.match_slices(
                        slice_centroids[i],
                        slice_centroids[i + 1],
                        v_z
                    )
                    matches.append(match)
                tracks = self.trajectory.build_tracks(slice_centroids, matches)
                smoothed = self.trajectory.smooth_surface_tracks(
                    tracks, slice_centroids
                )
                surface_tracks_data = smoothed
        if 'bottom' in clustering_results:
            bottom_mask = clustering_results['bottom']['mask']
            bottom_labels = clustering_results['bottom']['labels']
            
            if np.sum(bottom_mask) > 0:
                x_bottom = x_star[bottom_mask]
                y_bottom = y_star[bottom_mask]
                z_bottom = z_star[bottom_mask]
                slice_centroids = []
                for slice_data in slices:
                    slice_mask = np.isin(
                        np.arange(len(x_bottom)),
                        np.where(bottom_mask)[0][slice_data['indices']]
                    )
                    if np.sum(slice_mask) > 0:
                        labels_slice = bottom_labels[slice_mask]
                        centroids = self.trajectory.compute_cluster_centroids(
                            x_bottom[slice_mask],
                            y_bottom[slice_mask],
                            z_bottom[slice_mask],
                            labels_slice
                        )
                        slice_centroids.append(centroids)
                    else:
                        slice_centroids.append({})
                
                matches = []
                for i in range(len(slice_centroids) - 1):
                    match = self.trajectory.match_slices(
                        slice_centroids[i],
                        slice_centroids[i + 1],
                        0.0
                    )
                    matches.append(match)
                
                tracks = self.trajectory.build_tracks(slice_centroids, matches)
                regularized = self.trajectory.regularize_bottom_tracks(
                    tracks, slice_centroids, z_surf_mean
                )
                bottom_tracks_data = regularized
        
        results['stage4_trajectory'] = {
            'surface_tracks': surface_tracks_data,
            'bottom_tracks': bottom_tracks_data,
            'flight_direction': u
        }
        surface_data = None
        if len(surface_tracks_data) > 0:
            x_surf_all = []
            y_surf_all = []
            z_surf_all = []
            for track_data in surface_tracks_data.values():
                x_surf_all.extend(track_data['x'])
                y_surf_all.extend(track_data['y'])
                z_surf_all.extend(track_data['z'])
            
            if len(x_surf_all) > 0:
                surface_data = {
                    'x': np.array(x_surf_all),
                    'y': np.array(y_surf_all),
                    'z': np.array(z_surf_all)
                }
        if len(bottom_tracks_data) > 0:
            x_bottom_all = []
            y_bottom_all = []
            z_bottom_all = []
            for track_data in bottom_tracks_data.values():
                x_bottom_all.extend(track_data['x'])
                y_bottom_all.extend(track_data['y'])
                z_bottom_all.extend(track_data['z'])
            
            if len(x_bottom_all) > 0:
                x_bottom_corr = np.array(x_bottom_all)
                y_bottom_corr = np.array(y_bottom_all)
                z_bottom_corr = np.array(z_bottom_all)
                d_corrected, xy_corrected = self.refraction.correct_refraction(
                    x_bottom_corr,
                    y_bottom_corr,
                    z_bottom_corr,
                    z_surf_mean,
                    surface_data
                )
                z_corrected = z_surf_mean - d_corrected
                
                results['stage5_refraction'] = {
                    'x': xy_corrected[:, 0],
                    'y': xy_corrected[:, 1],
                    'z': z_corrected,
                    'd': d_corrected
                }
            else:
                bottom_mask = clustering_results['bottom']['mask']
                results['stage5_refraction'] = {
                    'x': x_star[bottom_mask],
                    'y': y_star[bottom_mask],
                    'z': z_star[bottom_mask],
                    'd': d_star[bottom_mask]
                }
        else:
            if 'bottom' in clustering_results:
                bottom_mask = clustering_results['bottom']['mask']
                x_bottom = x_star[bottom_mask]
                y_bottom = y_star[bottom_mask]
                z_bottom = z_star[bottom_mask]
                d_bottom = d_star[bottom_mask]
                
                d_corrected, xy_corrected = self.refraction.correct_refraction(
                    x_bottom, y_bottom, z_bottom, z_surf_mean, surface_data
                )
                z_corrected = z_surf_mean - d_corrected
                
                results['stage5_refraction'] = {
                    'x': xy_corrected[:, 0],
                    'y': xy_corrected[:, 1],
                    'z': z_corrected,
                    'd': d_corrected
                }
            else:
                results['stage5_refraction'] = {
                    'x': np.array([]),
                    'y': np.array([]),
                    'z': np.array([]),
                    'd': np.array([])
                }
        ref_data = results['stage5_refraction']
        if len(ref_data['x']) > 0:
            X_grid, Y_grid, D_grid = self.gridder.grid_depths(
                ref_data['x'],
                ref_data['y'],
                ref_data['d']
            )
            
            results['stage6_gridding'] = {
                'X_grid': X_grid,
                'Y_grid': Y_grid,
                'D_grid': D_grid
            }
            if reference_depths is not None:
                from scipy.interpolate import griddata
                D_ref_grid = griddata(
                    (x, y), reference_depths,
                    (X_grid, Y_grid),
                    method='linear'
                )
                metrics = self.metrics.compute_metrics(
                    D_grid, D_ref_grid
                )
                results['accuracy_metrics'] = metrics
        else:
            results['stage6_gridding'] = {
                'X_grid': np.array([]),
                'Y_grid': np.array([]),
                'D_grid': np.array([])
            }
            results['accuracy_metrics'] = {}
        return results

