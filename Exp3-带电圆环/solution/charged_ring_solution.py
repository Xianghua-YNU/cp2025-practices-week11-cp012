import numpy as np
import matplotlib.pyplot as plt


def calculate_potential_on_grid(y_coords, z_coords):
    """计算Y-Z平面上的电势分布"""
    # 生成网格坐标
    y_grid, z_grid = np.meshgrid(y_coords, z_coords)
    
    # 常量参数
    radius = 1.0     # 圆环半径(m)
    charge = 1e-9    # 电荷量(C)
    k = 8.988e9      # 库仑常数(N·m²/C²)
    
    # 初始化电势矩阵
    V = np.zeros_like(y_grid)
    
    print("开始计算电势...")
    
    # 数值积分（离散求和）
    num_points = 1000  # 积分采样点数量
    theta = np.linspace(0, 2*np.pi, num_points)
    dq = charge / num_points  # 每个点电荷的电荷量
    
    for i in range(num_points):
        # 圆环上的点坐标（在Y-Z平面，x=0）
        y_ring = radius * np.cos(theta[i])
        z_ring = radius * np.sin(theta[i])
        
        # 计算每个网格点到当前点电荷的距离
        r = np.sqrt((y_grid - y_ring)**2 + (z_grid - z_ring)**2)
        
        # 累加电势
        V += k * dq / r
    
    print("电势计算完成.")
    return V, y_grid, z_grid  # 确保返回三个变量


def calculate_electric_field(V, y_grid, z_grid):
    """通过电势梯度计算电场强度"""
    # 计算电场分量
    Ey, Ez = np.gradient(-V)
    
    # 计算梯度步长（假设网格均匀）
    dy = y_grid[0, 1] - y_grid[0, 0]
    dz = z_grid[1, 0] - z_grid[0, 0]
    
    # 调整梯度值
    Ey = Ey / dy
    Ez = Ez / dz
    
    return Ey, Ez
# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制 yz 平面上的等势线和电场线。

    参数:
        y_coords, z_coords: 定义网格的坐标范围
        V: 电势网格
        Ey, Ez: 电场分量网格
        y_grid, z_grid: 绘图用的二维网格坐标
    """
    print("开始绘图...")
    fig = plt.figure('Potential and Electric Field of Charged Ring (yz plane, x=0)', figsize=(12, 6))

    # 1. 绘制等势线图 (左侧子图)
    plt.subplot(1, 2, 1)
    contourf_plot = plt.contourf(y_grid/a, z_grid/a, V, levels=30, cmap='viridis')
    plt.colorbar(contourf_plot, label='Electric Potential V')
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Equipotential Contours')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 2. 绘制电场线图 (右侧子图)
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    stream_plot = plt.streamplot(y_grid/a, z_grid/a, Ey, Ez,
                                 color=E_magnitude,
                                 cmap='autumn',
                                 linewidth=1,
                                 density=1.5,
                                 arrowstyle='->',
                                 arrowsize=1.0)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Electric Field Lines')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.3)
    # 标记圆环在 yz 平面的截面位置 (y=±a, z=0)
    plt.plot([-1, 1], [0, 0], 'ro', markersize=5, label='Ring Cross-section')
    plt.legend()

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz 平面, x=0)
    num_points_y = 40  # y 方向点数
    num_points_z = 40  # z 方向点数
    range_factor = 2   # 计算范围是半径的多少倍
    y_range = np.linspace(-range_factor * a, range_factor * a, num_points_y)
    z_range = np.linspace(-range_factor * a, range_factor * a, num_points_z)

    # 1. 计算电势
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)

    # 2. 计算电场
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)

    # 3. 可视化
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
