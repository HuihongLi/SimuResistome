# One transparent 3D plot with all surfaces together.
# Coloring is driven by the z-values, with softened (pastel-like) colors and translucency.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize, ListedColormap

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

def resistant_fraction(Xs, Xr):
    N = Xs + Xr
    return float(Xr[-1] / max(N[-1], 1e-12))

# -----------------------------
# Parameter sweep
# -----------------------------
FIT_COSTS = np.linspace(0, 1.5, 50)          # x
SEL_PRESSURES = np.linspace(0.0, 1, 50)      # y
CONVERSION_RATES = [0.0001, 0.08, 0.2, 0.5, 1.0]

X, Y = np.meshgrid(FIT_COSTS, SEL_PRESSURES)

# Compute all Z surfaces first to get global min/max for normalized coloring
Z_list = []
for cr in CONVERSION_RATES:
    Z = np.zeros((len(SEL_PRESSURES), len(FIT_COSTS)))
    for i, s in enumerate(SEL_PRESSURES):
        for j, fc in enumerate(FIT_COSTS):
            sim = TaxaSimulator(
                t_end=150, dt=0.2,
                Xs0=1000.0, Xr0=10.0,
                g_base=0.8, fit_cost=fc,
                s_base=0.01, K_T=1e9,
                ab_start=0.0, ab_end=150.0,
                selection_pressure=s,
                input_s=1.0, input_r=1.0,
                conversion_rate=cr
            )
            _, Xs, Xr = sim.run()
            Z[i, j] = resistant_fraction(Xs, Xr)
    Z_list.append((cr, Z))

zmin = min(Z.min() for _, Z in Z_list)
zmax = max(Z.max() for _, Z in Z_list)
norm = Normalize(vmin=zmin, vmax=zmax)

# Build a softened (pastel-like) version of a standard colormap
base = plt.cm.viridis  # base colormap
N = 256
colors = base(np.linspace(0, 1, N))
# Mix with white to soften
soft_factor = 0.35  # 0=no softening, 1=all white
white = np.ones_like(colors[:, :3])
colors[:, :3] = (1 - soft_factor) * colors[:, :3] + soft_factor * white
soft_cmap = ListedColormap(colors)

# -----------------------------
# Single 3D plot
# -----------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

alpha_val = 0.55  # translucency

for cr, Z in Z_list:
    # Facecolors from z with soft colormap and transparency
    C = soft_cmap(norm(Z))
    C[..., -1] = alpha_val  # set alpha channel

    surf = ax.plot_surface(X, Y, Z, facecolors=C, rstride=1, cstride=1,
                           linewidth=0.2, antialiased=True, shade=False)

    # Optional: label each surface near the corner (x max, y max)
    xlbl = FIT_COSTS[-1]
    ylbl = SEL_PRESSURES[-1]
    zlbl = Z[-1, -1]
    ax.text(xlbl, ylbl, zlbl, f"M={cr}", fontsize=9, zorder=10)

m = plt.cm.ScalarMappable(norm=norm, cmap=soft_cmap)
m.set_array([])
cbar = fig.colorbar(m, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label("Resistant population fraction")

ax.set_xlabel("Fitness cost factor")
ax.set_ylabel("Selective pressure factor")
ax.set_zlabel("Resistant population fraction")

fig.tight_layout()
fig.savefig("R_prop_surfaces_all_in_one.png", dpi=220, bbox_inches="tight")
plt.show()

