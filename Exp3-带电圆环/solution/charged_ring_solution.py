import numpy as np
from scipy.integrate import quad
from scipy.constants import epsilon_0

# --- 常量定义 ---
a = 1.0  # 圆环半径 (m)
q = 1.0  # 参数 q，总电荷 Q = 4 * pi * epsilon_0 * q
C = q / (2 * np.pi)  # 计算常数

# --- 电势计算函数 ---
def calculate_potential(x, y, z):
    """
    计算点 (x, y, z) 处的电势 V。
    
    参数:
        x, y, z (float): 空间点坐标
        
    返回:
        V (float): 电势值
    """
    def integrand(phi):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        r = np.sqrt((x - a * cos_phi)**2 + (y - a * sin_phi)**2 + z**2)
        # 避免除零错误
        r = np.maximum(r, 1e-10)
        return 1 / r
    
    integral, _ = quad(integrand, 0, 2 * np.pi)
    V = C * integral
    return V

# --- 网格电势计算 ---
def calculate_potential_on_grid(y_coords, z_coords, x=0.0):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V。
    
    参数:
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组
        x (float): x 坐标 (默认 0.0)
        
    返回:
        V (np.ndarray): 在 (y, z) 网格上的电势值
        y_grid (np.ndarray): y 坐标网格
        z_grid (np.ndarray): z 坐标网格
    """
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')
    V = np.zeros_like(Y)
    
    for i in range(len(z_coords)):
        for j in range(len(y_coords)):
            V[i, j] = calculate_potential(x, Y[i, j], Z[i, j])
    
    return V, Y, Z

# --- 电场计算函数 ---
def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    根据电势 V 计算 yz 平面上的电场 E = -∇V。
    使用 np.gradient 进行数值微分。
    
    参数:
        V (np.ndarray): 电势网格
        y_coords (np.ndarray): y 坐标数组
        z_coords (np.ndarray): z 坐标数组
        
    返回:
        Ey (np.ndarray): 电场的 y 分量
        Ez (np.ndarray): 电场的 z 分量
    """
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]
    
    grad_y, grad_z = np.gradient(V, dy, dz)
    
    Ey = -grad_y
    Ez = -grad_z
    
    return Ey, Ez

# --- 可视化函数 ---
def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制 yz 平面上的等势线和电场线。
    
    参数:
        y_coords, z_coords: 定义网格的坐标
        V: 电势网格
        Ey, Ez: 电场分量网格
        y_grid, z_grid: 绘图用的二维网格坐标
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))

    # 1. 绘制等势线
    plt.subplot(1, 2, 1)
    contourf_plot = plt.contourf(y_grid, z_grid, V, levels=20, cmap='viridis')
    plt.colorbar(contourf_plot, label='Electric Potential V')
    contour_plot = plt.contour(y_grid, z_grid, V, levels=contourf_plot.levels, colors='white', linewidths=0.5)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Equipotential Lines')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. 绘制电场线
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    plt.streamplot(y_grid, z_grid, Ey, Ez, color=E_magnitude, cmap='autumn', density=1.5, arrowstyle='->', arrowsize=1.0)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Electric Field Lines')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.plot([-1, 1], [0, 0], 'ro', markersize=5, label='Ring Cross-section')
    plt.legend()

    plt.tight_layout()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域 (yz 平面, x=0)
    y_range = np.linspace(-2*a, 2*a, 50)
    z_range = np.linspace(-2*a, 2*a, 50)

    # 1. 计算电势
    print("正在计算电势...")
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    print("电势计算完成.")

    # 2. 计算电场
    print("正在计算电场...")
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    print("电场计算完成.")

    # 3. 可视化
    print("正在绘图...")
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
    print("绘图完成.")
