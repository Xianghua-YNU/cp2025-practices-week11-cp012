"""
带电圆环电势与电场计算及可视化程序
功能：
1. 计算带电圆环在Y-Z平面上的电势分布
2. 通过电势梯度计算电场强度分布
3. 可视化等势线和电场线分布
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_potential_on_grid(y_coords, z_coords):
    """
    计算带电圆环在Y-Z平面上的电势分布
    
    参数:
        y_coords: y轴坐标数组
        z_coords: z轴坐标数组
        
    返回:
        V: 电势矩阵(2D)
        y_grid: y坐标网格
        z_grid: z坐标网格
    """
    # 生成计算网格
    y_grid, z_grid = np.meshgrid(y_coords, z_coords)
    
    # 初始化电势矩阵
    V = np.zeros_like(y_grid)
    
    print("开始计算电势...")
    
    # 圆环参数
    radius = 1.0      # 圆环半径(m)
    charge = 1e-9     # 总电荷量(C)
    k = 8.988e9       # 库仑常数(N·m²/C²)
    
    # 数值积分参数
    num_points = 1000
    theta = np.linspace(0, 2*np.pi, num_points)
    dq = charge / num_points  # 每个点电荷的电荷量
    
    for angle in theta:
        # 圆环上的点坐标(Y-Z平面)
        y_ring = radius * np.cos(angle)
        z_ring = radius * np.sin(angle)
        
        # 计算距离
        r = np.sqrt((y_grid - y_ring)**2 + (z_grid - z_ring)**2)
        r[r < 1e-10] = np.inf  # 处理奇点
        
        # 累加电势
        V += k * dq / r
    
    print("电势计算完成.")
    return V, y_grid, z_grid  # 确保返回三个值

def calculate_electric_field(V, y_grid, z_grid):
    """
    通过电势梯度计算电场强度
    
    参数:
        V: 电势矩阵
        y_grid: y坐标网格
        z_grid: z坐标网格
        
    返回:
        Ey: y方向电场分量
        Ez: z方向电场分量
    """
    # 计算电势梯度
    Ey, Ez = np.gradient(-V)
    
    # 计算网格步长(假设均匀网格)
    dy = y_grid[0, 1] - y_grid[0, 0]
    dz = z_grid[1, 0] - z_grid[0, 0]
    
    # 归一化梯度值
    Ey /= dy
    Ez /= dz
    
    return Ey, Ez
def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    可视化电势和电场分布
    
    参数:
        y_coords, z_coords: 坐标范围
        V: 电势矩阵
        Ey, Ez: 电场分量
        y_grid, z_grid: 计算网格
    """
    print("开始绘制结果...")
    plt.figure('带电圆环电势与电场分布', figsize=(14, 6))
    
    # 1. 等势线图
    plt.subplot(1, 2, 1)
    levels = np.linspace(V.min(), V.max(), 30)
    contour = plt.contourf(y_grid/RING_RADIUS, z_grid/RING_RADIUS, V, 
                          levels=levels, cmap='viridis')
    plt.colorbar(contour, label='电势 (V)')
    plt.xlabel('y/a (a=圆环半径)')
    plt.ylabel('z/a')
    plt.title('等势线分布')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # 2. 电场线图
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    plt.streamplot(y_grid/RING_RADIUS, z_grid/RING_RADIUS, Ey, Ez,
                  color=E_magnitude, cmap='autumn',
                  linewidth=1, density=2, arrowstyle='->')
    plt.colorbar(label='电场强度 (V/m)')
    plt.xlabel('y/a')
    plt.ylabel('z/a')
    plt.title('电场线分布')
    plt.scatter([-1, 1], [0, 0], c='red', label='圆环截面')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    print("绘图完成.")

if __name__ == "__main__":
    # 参数设置
    GRID_POINTS = 40       # 每个方向的网格点数
    PLOT_RANGE = 2.5       # 绘图范围为半径的倍数
    
    # 创建计算网格
    y = np.linspace(-PLOT_RANGE*RING_RADIUS, PLOT_RANGE*RING_RADIUS, GRID_POINTS)
    z = np.linspace(-PLOT_RANGE*RING_RADIUS, PLOT_RANGE*RING_RADIUS, GRID_POINTS)
    
    # 计算电势和电场
    potential, y_mesh, z_mesh = calculate_potential_on_grid(y, z)
    E_y, E_z = calculate_electric_field(potential, y_mesh, z_mesh)
    
    # 可视化结果
    plot_potential_and_field(y, z, potential, E_y, E_z, y_mesh, z_mesh)
