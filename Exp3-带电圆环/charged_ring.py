import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 常量定义
a = 1.0  # 圆环半径
q = 1.0  # 电荷参数
C = q / (2 * np.pi)  # 积分常数

def calculate_potential_on_grid(y_coords, z_coords):
    """在 yz 平面 (x=0) 的网格上计算电势"""
    # 创建二维网格
    y_grid, z_grid = np.meshgrid(y_coords, z_coords)
    
    # 生成积分角度phi (0到2π)
    phi = np.linspace(0, 2*np.pi, 100)
    
    # 计算环上电荷元坐标
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    
    # 初始化电势矩阵
    V = np.zeros_like(y_grid)
    
    # 对每个网格点进行积分
    for i in range(y_grid.shape[0]):
        for j in range(y_grid.shape[1]):
            # 计算场点到环上各点的距离
            dx = 0 - x_ring
            dy = y_grid[i,j] - y_ring
            dz = z_grid[i,j] - 0
            R = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 处理奇异点
            R[R < 1e-10] = np.inf
            
            # 积分计算电势
            integrand = 1/R
            V[i,j] = C * simpson(integrand, phi)
    
    return V, y_grid, z_grid

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """通过数值微分计算电场"""
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]
    
    # 计算梯度
    Ey, Ez = np.gradient(-V, dy, dz)
    
    # 归一化处理
    max_E = max(np.abs(Ey).max(), np.abs(Ez).max())
    if max_E > 0:
        Ey /= max_E
        Ez /= max_E
    
    return Ey, Ez

# 可视化函数保持不变

# ... (保持原有可视化函数不变) ...

if __name__ == "__main__":
    # 测试用的坐标范围
    y = np.linspace(-2, 2, 5)
    z = np.linspace(-2, 2, 5)
    
    # 计算电势和电场
    V, y_grid, z_grid = calculate_potential_on_grid(y, z)
    Ey, Ez = calculate_electric_field_on_grid(V, y, z)
    
    # 执行可视化
    plot_potential_and_field(y, z, V, Ey, Ez, y_grid, z_grid)
