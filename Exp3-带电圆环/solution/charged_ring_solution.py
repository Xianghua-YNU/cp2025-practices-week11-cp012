import numpy as np

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
