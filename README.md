# 水深测量处理工作流

一个全面的六阶段流水线，用于将原始地理参考LAS点处理为相对于平均水面的深度网格。


## 安装

### 要求

- Python 3.8+
- 完整依赖列表请参见 `requirements.txt`

### 安装依赖

```bash
pip install -r requirements.txt
```


### 自定义参数

```python
workflow = BathymetryWorkflow(
    # 预处理
    tile_size=256.0,
    overlap=0.3,
    
    # 体素化
    dx=0.5,
    dy=0.5,
    fdr_alpha=0.05,
    
    # 聚类
    beta_surface=0.5,
    beta_water=1.0,
    beta_bottom=1.5,
    a=0.4,
    
    # 网格化
    grid_resolution=0.5,
    search_radius=2.0
)

## 关键参数

### 预处理
- `tile_size`：处理瓦片大小（米，默认：128.0）
- `overlap`：瓦片重叠分数（默认：0.5）
- `zscore_threshold`：异常值移除的Z分数阈值（默认：3.0）

### 体素化
- `dx`, `dy`：水平体素尺寸（米，默认：1.0）
- `dz`：垂直体素厚度（米，从FD规则自动计算）
- `fdr_alpha`：FDR控制水平（默认：0.10）

### 聚类
- `beta_surface`, `beta_water`, `beta_bottom`：分层特定比例因子（默认：0.6, 0.9, 1.2）
- `a`：半径计算的混合指数（默认：0.5）
- `minPts_*`：每层的最小聚类大小（默认：10, 16, 12）

### 轨迹
- `n_slices`：连续切片数量（默认：5）
- `L_min`：最小轨迹长度（切片数，默认：3）

### 折射
- `n_air`, `n_water`：折射率（默认：1.0003, 1.33）
- `wave_uncertainty_threshold`：一维回退阈值（默认：0.05 rad/m）

### 网格化
- `grid_resolution`：网格单元大小（米，默认：1.0）
- `search_radius`：聚合搜索半径（米，默认：1.5）
- `anisotropic`：使用各向异性搜索（默认：True）

## 输出

工作流返回包含每个阶段结果的字典：

```python
results = {
    'stage1_preprocessing': {...},      # 预处理点和水面
    'stage2_voxelization': {...},       # FDR门控的显著光子
    'stage3_clustering': {...},         # 分层聚类标签
    'stage4_trajectory': {...},         # 表面和底部轨迹
    'stage5_refraction': {...},         # 折射校正的底部点
    'stage6_gridding': {...},           # 网格化深度（X_grid, Y_grid, D_grid）
    'accuracy_metrics': {...}            # 精度指标（如果提供了参考数据）
}
```

## 精度指标

如果提供了参考深度，工作流计算：
- **Bias**：平均残差（米）
- **MAE**：平均绝对误差（米）
- **RMSE**：均方根误差（米）
- **R²**：决定系数
- **NMAD**：归一化中位数绝对偏差（米）
- **P68, P95**：绝对残差的第68和第95百分位数（米）

## 依赖

- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- laspy >= 2.5.0
- pandas >= 2.0.0
- scikit-image >= 0.21.0
- statsmodels >= 0.14.0
- numba >= 0.57.0
- matplotlib >= 3.7.0
- pyproj >= 3.6.0
- shapely >= 2.0.0



