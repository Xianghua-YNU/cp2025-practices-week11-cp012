import numpy as np
import matplotlib.pyplot as plt

# --- 常量定义 ---
a = 1.0  # 圆环半径 (单位: m)
C = 1.0 / (2 * np.pi)  # 电势计算常数，对应 q=1

# --- 计算函数 ---

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V(0, y, z)。
    使用 numpy 的向量化和 trapz 进行数值积分。

    参数:
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组

    返回:
        V (np.ndarray): 在 (y, z) 网格上的电势值 (z 维度优先)
        y_grid (np.ndarray): 绘图用的二维 y 网格坐标
        z_grid (np.ndarray): 绘图用的二维 z 网格坐标
    """
    print("开始计算电势...")
    
    # 1. 创建 y, z, phi 网格 (使用 np.mgrid)
    z_grid, y_grid, phi_grid = np.mgrid[z_coords.min():z_coords.max():complex(0, len(z_coords)),
                                        y_coords.min():y_coords.max():complex(0, len(y_coords)),
                                        0:2*np.pi:100j]
    
    # 2. 计算场点到圆环上各点的距离 R
    # 圆环上的点: (a*cos(phi), a*sin(phi), 0)
    # 场点: (0, y, z)
    R = np.sqrt((a * np.cos(phi_grid))**2 + (y_grid - a * np.sin(phi_grid))**2 + z_grid**2)
    
    # 3. 处理 R 可能为零或非常小的情况，避免除零错误
    R[R < 1e-10] = 1e-10
    
    # 4. 计算电势微元 dV = C / R
    dV = C / R
    
    # 5. 对 phi 进行积分 (使用 np.trapz)
    # np.trapz 默认沿最后一个轴积分
    V = np.trapz(dV, dx=phi_grid[0,0,1]-phi_grid[0,0,0], axis=-1)
    
    print("电势计算完成.")
    # 6. 返回计算得到的电势 V 和对应的 y_grid, z_grid (取一个切片)
    return V, y_grid[:,:,0], z_grid[:,:,0]

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    根据电势 V 计算 yz 平面上的电场 E = -∇V。
    使用 np.gradient 进行数值微分。

    参数:
        V (np.ndarray): 电势网格 (z 维度优先)
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组

    返回:
        Ey (np.ndarray): 电场的 y 分量
        Ez (np.ndarray): 电场的 z 分量
    """
    print("开始计算电场...")
    
    # 1. 计算 y 和 z 方向的网格间距 dy, dz
    dz = z_coords[1] - z_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    # 2. 使用 np.gradient 计算电势的负梯度
    # 注意 V 的维度顺序是 (z, y)
    # gradient 返回值顺序与 V 的维度顺序一致: (dV/dz, dV/dy)
    grad_z, grad_y = np.gradient(-V, dz, dy)
    
    # E = -∇V，所以 Ey = -dV/dy, Ez = -dV/dz
    Ey = grad_y
    Ez = grad_z
    
    print("电场计算完成.")
    # 3. 返回电场的 y 和 z 分量
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
    # plt.contour(y_grid, z_grid, V, levels=contourf_plot.levels, colors='white', linewidths=0.5)
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
    # 范围可以以圆环半径 a 为单位
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
