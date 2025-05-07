import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 常量定义
a = 1.0  # 圆环半径
q = 1.0  # 电荷参数
epsilon0 = 1.0  # 真空介电常数(简化计算)

def potential_integrand(phi, x, y, z):
    """计算电势积分的被积函数"""
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    distance = np.sqrt((x - x_ring)**2 + (y - y_ring)**2 + z**2)
    return 1.0 / distance

def calculate_potential(x, y, z):
    """计算点(x,y,z)处的电势"""
    # 处理接近圆环的情况
    if np.abs(z) < 1e-10 and np.abs(np.sqrt(x**2 + y**2) - a) < 1e-10:
        return np.inf
    
    integral, _ = quad(potential_integrand, 0, 2*np.pi, args=(x, y, z))
    return (q / (2 * np.pi)) * integral

def electric_field_integrand(phi, x, y, z):
    """计算电场积分的被积函数"""
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    r_vec = np.array([x - x_ring, y - y_ring, z])
    r = np.linalg.norm(r_vec)
    if r < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    return r_vec / r**3

def calculate_electric_field(x, y, z):
    """计算点(x,y,z)处的电场"""
    # 处理接近圆环的情况
    if np.abs(z) < 1e-10 and np.abs(np.sqrt(x**2 + y**2) - a) < 1e-10:
        return np.array([np.inf, np.inf, np.inf])
    
    Ex_integral, _ = quad(lambda phi: electric_field_integrand(phi, x, y, z)[0], 0, 2*np.pi)
    Ey_integral, _ = quad(lambda phi: electric_field_integrand(phi, x, y, z)[1], 0, 2*np.pi)
    Ez_integral, _ = quad(lambda phi: electric_field_integrand(phi, x, y, z)[2], 0, 2*np.pi)
    
    return (q / (2 * np.pi)) * np.array([Ex_integral, Ey_integral, Ez_integral])

def plot_potential_and_field():
    """在yz平面绘制等势线和电场"""
    # 创建网格
    y = np.linspace(-2*a, 2*a, 30)
    z = np.linspace(-2*a, 2*a, 30)
    Y, Z = np.meshgrid(y, z)
    
    # 计算电势和电场
    V = np.zeros_like(Y)
    Ey = np.zeros_like(Y)
    Ez = np.zeros_like(Y)
    
    for i in range(len(y)):
        for j in range(len(z)):
            x_point = 0.0  # 在yz平面(x=0)
            V[j,i] = calculate_potential(x_point, y[i], z[j])
            E = calculate_electric_field(x_point, y[i], z[j])
            Ey[j,i] = E[1]
            Ez[j,i] = E[2]
    
    # 绘制等势线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    levels = np.linspace(np.min(V[V > -np.inf]), np.max(V[V < np.inf]), 20)
    contour = plt.contour(Y, Z, V, levels=levels, cmap='viridis')
    plt.colorbar(label='Electric Potential')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Equipotential Lines in yz Plane')
    plt.scatter([-a, a], [0, 0], c='red', marker='o')  # 标记圆环位置
    
    # 绘制电场
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    plt.streamplot(Y, Z, Ey, Ez, color=E_magnitude, cmap='autumn', 
                  linewidth=1, density=1.5, arrowsize=1.0)
    plt.colorbar(label='Electric Field Magnitude')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Electric Field in yz Plane')
    plt.scatter([-a, a], [0, 0], c='red', marker='o')  # 标记圆环位置
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_potential_and_field()
