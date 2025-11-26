import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from typing import Tuple, Optional, Dict
import warnings


class RefractionCorrector:
    def __init__(
        self,
        n_air: float = 1.0003,
        n_water: float = 1.33,
        wave_uncertainty_threshold: float = 0.05
    ):
        self.n_air = n_air
        self.n_water = n_water
        self.wave_uncertainty_threshold = wave_uncertainty_threshold
    
    def fit_wave_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: Optional[np.ndarray] = None
    ) -> Dict:
        if len(x) < 3:
            z_mean = np.mean(z) if len(z) > 0 else 0.0
            return {
                'z_surface': lambda x, y: np.full(len(x), z_mean),
                'normals': lambda x, y: np.column_stack([
                    np.zeros(len(x)),
                    np.zeros(len(x)),
                    np.ones(len(x))
                ]),
                'uncertainty': np.inf,
                'method': 'constant'
            }
        try:
            rbf = RBFInterpolator(
                np.column_stack([x, y]), z,
                kernel='thin_plate_spline', smoothing=0.1
            )
            def z_surface(x_new, y_new):
                points = np.column_stack([x_new, y_new])
                return rbf(points)
            def compute_normals(x_new, y_new):
                eps = 0.01
                z_center = z_surface(x_new, y_new)
                z_x = z_surface(x_new + eps, y_new) - z_center
                z_y = z_surface(x_new, y_new + eps) - z_center
                normals = np.column_stack([
                    -z_x / eps,
                    -z_y / eps,
                    np.ones(len(x_new))
                ])
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / np.maximum(norms, 1e-6)
                return normals
            x_test = np.linspace(x.min(), x.max(), 10)
            y_test = np.linspace(y.min(), y.max(), 10)
            X_test, Y_test = np.meshgrid(x_test, y_test)
            normals_test = compute_normals(X_test.ravel(), Y_test.ravel())
            slopes = np.arctan(np.sqrt(
                normals_test[:, 0]**2 + normals_test[:, 1]**2
            ))
            uncertainty = np.std(slopes)
            
            return {
                'z_surface': z_surface,
                'normals': compute_normals,
                'uncertainty': uncertainty,
                'method': 'rbf'
            }
        except Exception as e:
            warnings.warn(f"RBF interpolation failed: {e}, using constant surface")
            z_mean = np.mean(z)
            return {
                'z_surface': lambda x, y: np.full(len(x), z_mean),
                'normals': lambda x, y: np.column_stack([
                    np.zeros(len(x)),
                    np.zeros(len(x)),
                    np.ones(len(x))
                ]),
                'uncertainty': np.inf,
                'method': 'constant'
            }
    
    def apply_snells_law(
        self,
        ray_direction: np.ndarray,
        surface_normal: np.ndarray
    ) -> np.ndarray:
        cos_theta_air = np.dot(ray_direction, surface_normal)
        cos_theta_air = np.clip(cos_theta_air, -1.0, 1.0)
        sin_theta_air = np.sqrt(1 - cos_theta_air**2)
        sin_theta_water = (self.n_air / self.n_water) * sin_theta_air
        if sin_theta_water > 1.0:
            return ray_direction - 2 * cos_theta_air * surface_normal
        cos_theta_water = np.sqrt(1 - sin_theta_water**2)
        perp = ray_direction - cos_theta_air * surface_normal
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-6:
            perp = perp / perp_norm
        refracted = sin_theta_water * perp - cos_theta_water * surface_normal
        
        return refracted / np.linalg.norm(refracted)
    
    def correct_3d_refraction(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        z_surface: float,
        wave_model: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.array([]), np.column_stack([x, y])
        corrected_depths = np.zeros(len(x))
        corrected_xy = np.zeros((len(x), 2))
        for i in range(len(x)):
            z_surf_i = wave_model['z_surface'](
                np.array([x[i]]), np.array([y[i]])
            )[0]
            normal_i = wave_model['normals'](
                np.array([x[i]]), np.array([y[i]])
            )[0]
            ray_incident = np.array([0.0, 0.0, -1.0])
            ray_refracted = self.apply_snells_law(ray_incident, normal_i)
            d_obs = z_surf_i - z[i]
            cos_theta_air = -ray_incident[2]
            cos_theta_water = -ray_refracted[2]
            if cos_theta_water > 1e-6:
                d_corrected = d_obs * (cos_theta_air / cos_theta_water)
            else:
                d_corrected = d_obs
            corrected_depths[i] = d_corrected
            horizontal_shift = d_obs * np.tan(
                np.arccos(cos_theta_air) - np.arccos(cos_theta_water)
            )
            corrected_xy[i, 0] = x[i] + horizontal_shift * normal_i[0]
            corrected_xy[i, 1] = y[i] + horizontal_shift * normal_i[1]
        
        return corrected_depths, corrected_xy
    
    def correct_1d_refraction(
        self,
        d_obs: np.ndarray
    ) -> np.ndarray:
        d_true = d_obs * (self.n_air / self.n_water)
        return d_true
    
    def correct_refraction(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        z_surface_mean: float,
        surface_track_data: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.array([]), np.column_stack([x, y])
        if surface_track_data is not None:
            x_surf = surface_track_data.get('x', np.array([]))
            y_surf = surface_track_data.get('y', np.array([]))
            z_surf = surface_track_data.get('z', np.array([]))
            if len(x_surf) > 0:
                wave_model = self.fit_wave_surface(x_surf, y_surf, z_surf)
                if wave_model['uncertainty'] < self.wave_uncertainty_threshold:
                    return self.correct_3d_refraction(
                        x, y, z, z_surface_mean, wave_model
                    )
        d_obs = z_surface_mean - z
        d_corrected = self.correct_1d_refraction(d_obs)
        corrected_xy = np.column_stack([x, y])
        
        return d_corrected, corrected_xy

