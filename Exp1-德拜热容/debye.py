import numpy as np
import matplotlib.pyplot as plt

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K


def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2

    参数：
    x : float 或 numpy.ndarray
        积分变量

    返回：
    float 或 numpy.ndarray：被积函数的值

    注意：
    - 为了避免数值溢出，当x>100时，分母近似为e^(2x)
    - 将输入转换为数组以处理标量和数组输入
    """
    x = np.asarray(x)  # 确保输入为NumPy数组
    result = np.zeros_like(x)

    # 对于x较大的情况使用近似处理
    mask = x > 100
    if np.any(mask):
        result[mask] = (x[mask] ** 4) * np.exp(-x[mask])

    # 对于正常情况计算精确值
    mask = ~mask
    if np.any(mask):
        exp_x = np.exp(x[mask])
        numerator = (x[mask] ** 4) * exp_x
        denominator = (exp_x - 1) ** 2
        result[mask] = numerator / denominator

    return result


def gauss_quadrature(f, a, b, n=50):
    """实现高斯-勒让德积分

    参数：
    f : callable
        被积函数
    a, b : float
        积分区间的端点
    n : int
        高斯点的数量，默认为50

    返回：
    float：积分结果

    高斯-勒让德积分步骤：
    1. 获取标准区间[-1,1]上的高斯点和权重
    2. 将高斯点变换到[a,b]区间
    3. 计算积分值
    """
    # 获取标准区间[-1,1]上的高斯点和权重
    xi, wi = np.polynomial.legendre.leggauss(n)

    # 将积分区间从[-1,1]变换到[a,b]
    x_transformed = 0.5 * (b - a) * xi + 0.5 * (b + a)

    # 计算积分值
    integral = 0.5 * (b - a) * np.sum(wi * f(x_transformed))

    return integral


def cv(T):
    """计算给定温度T下的热容

    参数：
    T : float
        温度，单位：K

    返回：
    float：热容值，单位：J/K

    计算步骤：
    1. 检查温度是否为0（避免除以0错误）
    2. 计算积分上限θ_D/T
    3. 使用高斯积分计算积分部分
    4. 根据德拜公式计算热容
    """
    # 如果温度为0K，热容为0
    if T == 0:
        return 0.0

    # 计算积分上限
    upper_limit = theta_D / T

    # 使用高斯积分计算积分部分，使用n=50个点
    integral = gauss_quadrature(integrand, 0, upper_limit)

    # 根据德拜公式计算热容
    Cv = 9 * V * rho * kB * (T / theta_D) ** 3 * integral

    return Cv


def plot_cv():
    """绘制热容随温度的变化曲线

    绘制步骤：
    1. 创建温度数组 (5K到500K)
    2. 计算每个温度对应的热容
    3. 使用matplotlib绘制曲线
    4. 添加标签和标题
    """
    # 创建温度数组，从5K到500K，共100个点
    temperatures = np.linspace(5, 500, 100)

    # 计算每个温度对应的热容
    heat_capacities = [cv(T) for T in temperatures]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(temperatures, heat_capacities, 'b-', linewidth=2)

    # 添加标签和标题
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Heat Capacity (J/K)', fontsize=12)
    plt.title('Heat Capacity of Aluminum vs Temperature (Debye Model)', fontsize=14)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图形
    plt.show()


def test_cv():
    """测试热容计算函数

    测试步骤：
    1. 选择几个特征温度点
    2. 计算并打印热容值
    3. 验证结果是否合理
    """
    test_temperatures = [5, 100, 300, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)")
    print("-" * 40)
    for T in test_temperatures:
        result = cv(T)
        print(f"{T:8.1f}\t{result:10.3e}")


def main():
    # 运行测试
    test_cv()

    # 绘制热容曲线
    plot_cv()


if __name__ == '__main__':
    main()
