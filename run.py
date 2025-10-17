import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# -----------------------------
# Model (same as before)
# -----------------------------
class TaxaSimulator:
    def __init__(self, t_end=100, dt=0.2, Xs0=1000.0, Xr0=10.0, g_base=0.8, fit_cost=0.9,
                 s_base=0.01, K_T=1e9, ab_start=20.0, ab_end=100.0,
                 selection_pressure=0.5, input_s=1.0, input_r=1.0,
                 conversion_rate=0.002):
        self.t_end, self.dt = t_end, dt
        self.t = np.linspace(0, t_end, int(round(t_end / dt)) + 1)
        self.g_base, self.g_r = g_base, fit_cost * g_base
        self.s_base, self.K_T = s_base, K_T
        self.ab_start, self.ab_end = ab_start, ab_end
        self.selection_pressure = selection_pressure
        self.input_s, self.input_r = input_s, input_r
        self.conversion_rate = conversion_rate
        self.Xs, self.Xr = np.zeros_like(self.t), np.zeros_like(self.t)
        self.Xs[0], self.Xr[0] = Xs0, Xr0

    def run(self):
        for k in range(len(self.t) - 1):
            ab_active = (self.ab_start <= self.t[k] <= self.ab_end)
            g_s = self.g_base * (1.0 - self.selection_pressure) if ab_active else self.g_base
            N = self.Xs[k] + self.Xr[k]
            if N > self.K_T:
                g_avg = (self.Xs[k] * g_s + self.Xr[k] * self.g_r) / max(N, 1e-12)
                g_s_eff, g_r_eff = g_s - g_avg, self.g_r - g_avg
            else:
                g_s_eff, g_r_eff = g_s, self.g_r
            conversion_s_to_r = self.conversion_rate * self.Xs[k]
            conversion_r_to_s = self.conversion_rate * self.Xr[k]
            dXs = (self.Xs[k] * g_s_eff - self.s_base * self.Xs[k] +
                   self.input_s - conversion_s_to_r + conversion_r_to_s)
            dXr = (self.Xr[k] * g_r_eff - self.s_base * self.Xr[k] +
                   self.input_r + conversion_s_to_r - conversion_r_to_s)
            self.Xs[k + 1] = max(self.Xs[k] + dXs * self.dt, 0.0)
            self.Xr[k + 1] = max(self.Xr[k] + dXr * self.dt, 0.0)
        return self.t, self.Xs, self.Xr

def compute_derivatives(Xs, Xr, selection_pressure, g_base=0.8, fit_cost=0.9,
                       s_base=0.01, K_T=1e9, conversion_rate=0.002,
                       input_s=1.0, input_r=1.0, ab_active=True):
    """计算给定状态下的导数"""
    g_r = fit_cost * g_base
    g_s = g_base * (1.0 - selection_pressure) if ab_active else g_base
    N = Xs + Xr
    
    if N > K_T:
        g_avg = (Xs * g_s + Xr * g_r) / max(N, 1e-12)
        g_s_eff, g_r_eff = g_s - g_avg, g_r - g_avg
    else:
        g_s_eff, g_r_eff = g_s, g_r
    
    conversion_s_to_r = conversion_rate * Xs
    conversion_r_to_s = conversion_rate * Xr
    
    dXs = (Xs * g_s_eff - s_base * Xs + input_s - conversion_s_to_r + conversion_r_to_s)
    dXr = (Xr * g_r_eff - s_base * Xr + input_r + conversion_s_to_r - conversion_r_to_s)
    
    return dXs, dXr

def compute_resistant_fraction_derivative(rf, selection_pressure, N_total=1000,
                                         fit_cost=0.9, conversion_rate=0.002):
    """计算resistant fraction的变化率"""
    # 从resistant fraction反推Xs和Xr
    Xr = rf * N_total
    Xs = (1 - rf) * N_total
    
    # 计算导数
    dXs, dXr = compute_derivatives(Xs, Xr, selection_pressure, 
                                   fit_cost=fit_cost, 
                                   conversion_rate=conversion_rate)
    
    # 计算resistant fraction的导数
    # d(rf)/dt = d(Xr/N)/dt = (N*dXr - Xr*dN)/(N^2)
    # 其中 dN = dXs + dXr
    dN = dXs + dXr
    if N_total > 1e-12:
        drf_dt = (N_total * dXr - Xr * dN) / (N_total ** 2)
    else:
        drf_dt = 0
    
    return drf_dt

# -----------------------------
# 创建2D箭头图（单个图）
# -----------------------------
# 参数设置
FIT_COST = 0.5
CONVERSION_RATE = 0.08

# 创建网格
rf_values = np.linspace(0.01, 0.99, 30)  # resistant fraction
sp_values = np.linspace(0.0, 1.0, 30)     # selective pressure

RF, SP = np.meshgrid(rf_values, sp_values)

# 创建单个图
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# 计算向量场
U = np.zeros_like(RF)  # x方向速度 (resistant fraction变化)
V = np.zeros_like(SP)  # y方向速度 (固定为0，因为selective pressure是参数)

for i in range(len(sp_values)):
    for j in range(len(rf_values)):
        rf = RF[i, j]
        sp = SP[i, j]
        
        # 计算resistant fraction的变化率
        drf_dt = compute_resistant_fraction_derivative(
            rf, sp, N_total=1000, fit_cost=FIT_COST, conversion_rate=CONVERSION_RATE
        )
        
        U[i, j] = drf_dt
        V[i, j] = 0  # selective pressure不变

# 绘制箭头场（使用quiver代替streamplot）
# 降低密度以避免过于密集
skip = 2  # 每2个点采样一次

# 计算速度
speed = np.sqrt(U**2 + V**2)

# 增加速度阈值，确保平衡线附近没有箭头
speed_threshold = speed.max() * 0.1  # 提高到10%，过滤掉更多平衡线附近的箭头

# 创建mask
mask = speed[::skip, ::skip] > speed_threshold

# 归一化箭头长度（对于显示的箭头）
speed_safe = np.where(speed > 1e-10, speed, 1)
U_norm = U / speed_safe
V_norm = V / speed_safe

# 只绘制满足条件的箭头
RF_plot = RF[::skip, ::skip][mask]
SP_plot = SP[::skip, ::skip][mask]
U_plot = U_norm[::skip, ::skip][mask]
V_plot = V_norm[::skip, ::skip][mask]

# 绘制箭头
ax.quiver(RF_plot, SP_plot, U_plot, V_plot,
          color='black', alpha=0.8, scale=25, width=0.004,
          headwidth=4.5, headlength=5.5)

# 添加零线（drf/dt = 0）- 灰色
ax.contour(RF, SP, U, levels=[0], colors='gray', linewidths=2.5, linestyles='--')

# 设置标签
ax.set_xlabel('Resistant fraction', fontsize=13)
ax.set_ylabel('Selective pressure', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('TaxaPhase_2D.png', dpi=220, bbox_inches='tight')
plt.show()

print("2D箭头图已生成！")
print(f"- Fitness cost = {FIT_COST}")
print(f"- Conversion rate = {CONVERSION_RATE}")
print("- X轴：Resistant fraction (Xr/(Xs+Xr))")
print("- Y轴：Selective pressure")
print("- 黑色小箭头：系统演化方向")
print("- 灰色虚线：平衡线 (drf/dt = 0)")
print(f"- 速度阈值：{speed_threshold:.4f} (过滤平衡线附近箭头)")