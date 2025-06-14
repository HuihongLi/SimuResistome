import numpy as np
import matplotlib.pyplot as plt


class TaxaSimulator:
    def __init__(self,
                 t_end=500, dt=0.01,
                 Xs0=1000, Xr0=0,
                 g_s_base=0.8, g_s_drug=0.2, g_r=0.3,
                 s_base=0.01, s_drug=1,
                 delta=0.01, mu=0.01,
                 K_T=1e9,
                 ab_start=20, ab_end=30,
                 ab_type=1):

        # Simulation setup
        self.t_end = t_end
        self.dt = dt
        self.t = np.linspace(0, t_end, int(t_end / dt) + 1)

        # Initial states
        self.Xs0 = Xs0
        self.Xr0 = Xr0

        # Growth and kill rates
        self.g_s_base = g_s_base
        self.g_s_drug = g_s_drug
        self.g_r = g_r
        self.s_base = s_base
        self.s_drug = s_drug
        self.delta = delta
        self.mu = mu
        self.K_T = K_T

        # Antibiotic timing and type
        self.ab_start = ab_start
        self.ab_end = ab_end
        self.ab_type = ab_type
        self.ab_label = self._get_ab_label(ab_type)

        # Results containers
        self.Xs = np.zeros_like(self.t)
        self.Xr = np.zeros_like(self.t)
        self.Xs[0], self.Xr[0] = Xs0, Xr0

    def _get_ab_label(self, ab_type):
        return {
            1: "Bacteriostatic (Type I)",
            2: "Bactericidal (Type II)",
            3: "Mixed Action (Type III)"
        }.get(ab_type, "Unknown")

    def run(self):
        for k in range(len(self.t) - 1):
            current_time = self.t[k]
            ab_active = self.ab_start <= current_time <= self.ab_end

            # Determine g_s and s_s based on ab_type
            if self.ab_type == 1:
                g_s = self.g_s_drug if ab_active else self.g_s_base
                s_s = self.s_base
            elif self.ab_type == 2:
                g_s = self.g_s_base
                s_s = self.s_drug if ab_active else self.s_base
            elif self.ab_type == 3:
                g_s = self.g_s_drug if ab_active else self.g_s_base
                s_s = self.s_drug if ab_active else self.s_base
            else:
                raise ValueError("ab_type must be 1, 2, or 3")

            s_r = self.s_base
            N = self.Xs[k] + self.Xr[k]

            if N <= self.K_T:
                g_s_eff, g_r_eff = g_s, self.g_r
            else:
                g_avg = (self.Xs[k] * g_s + self.Xr[k] * self.g_r) / N
                g_s_eff, g_r_eff = g_s - g_avg, self.g_r - g_avg

            dXs = self.Xs[k] * g_s_eff - s_s * self.Xs[k] + self.delta * self.Xr[k]
            dXr = self.Xr[k] * g_r_eff - s_r * self.Xr[k] + self.mu * self.Xs[k]

            self.Xs[k + 1] = max(self.Xs[k] + dXs * self.dt, 0.0)
            self.Xr[k + 1] = max(self.Xr[k] + dXr * self.dt, 0.0)

        return self.t, self.Xs, self.Xr, self.ab_label

    def plot(self):
        N_total = self.Xs + self.Xr
        rel_Xs = np.divide(self.Xs, N_total, where=N_total > 0)
        rel_Xr = np.divide(self.Xr, N_total, where=N_total > 0)
        rel_abundance = np.vstack([rel_Xs, rel_Xr])

        plt.figure(figsize=(10, 6))

        # Absolute densities
        plt.subplot(3, 1, 1)
        plt.plot(self.t, self.Xs, label="Susceptible", linewidth=2)
        plt.plot(self.t, self.Xr, label="Resistant", linewidth=2)
        plt.axvspan(self.ab_start, self.ab_end, color='red', alpha=0.1, label="Antibiotic")
        plt.ylabel("Density [CFU/mL]")
        plt.title(f"Population Dynamics — {self.ab_label}")
        plt.legend()

        # Relative abundances
        plt.subplot(3, 1, 2)
        plt.stackplot(self.t, rel_abundance, labels=["Susceptible", "Resistant"], alpha=0.85)
        plt.axvspan(self.ab_start, self.ab_end, color='red', alpha=0.1)
        plt.ylabel("Relative Abundance")
        plt.ylim(0, 1)
        plt.legend(loc="upper right")

        # Total population
        plt.subplot(3, 1, 3)
        plt.plot(self.t, N_total, color="black", label="Total Population")
        plt.axvspan(self.ab_start, self.ab_end, color='red', alpha=0.1)
        plt.xlabel("Time [h]")
        plt.ylabel("Total [CFU/mL]")
        plt.legend()

        plt.tight_layout()
        plt.show()


# === Run a test case ===
if __name__ == "__main__":
    sim = TaxaSimulator(ab_type=2)  # Try 1, 2, or 3
    sim.run()
    sim.plot()
