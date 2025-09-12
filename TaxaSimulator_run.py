import numpy as np
import matplotlib.pyplot as plt

class TaxaSimulator:
    def __init__(self, t_end=100, dt=0.01, Xs0=0, Xr0=0, g_base=0.8, fit_cost=0.9,
                 s_base=0.01, K_T=1e9, ab_start=20, ab_end=30,
                 selection_pressure=0.5, input_s=1.0, input_r=1.0,
                 conversion_rate=0.002):
        """
        TaxaSimulator with symmetric conversion between susceptible and resistant populations.
        """

        self.t_end, self.dt = t_end, dt
        self.t = np.linspace(0, t_end, int(t_end / dt) + 1)
        self.g_base, self.g_r = g_base, fit_cost * g_base
        self.s_base, self.K_T = s_base, K_T
        self.ab_start, self.ab_end = ab_start, ab_end
        self.selection_pressure = selection_pressure
        self.input_s, self.input_r = input_s, input_r
        
        # Symmetric conversion rate
        self.conversion_rate = conversion_rate
        
        self.Xs, self.Xr = np.zeros_like(self.t), np.zeros_like(self.t)
        self.Xs[0], self.Xr[0] = Xs0, Xr0

    def run(self):
        for k in range(len(self.t) - 1):
            ab_active = self.ab_start <= self.t[k] <= self.ab_end
            
            # Apply selective pressure during antibiotic treatment
            g_s = self.g_base * (1 - self.selection_pressure) if ab_active else self.g_base
            
            N = self.Xs[k] + self.Xr[k]
            if N > self.K_T:
                g_avg = (self.Xs[k] * g_s + self.Xr[k] * self.g_r) / N
                g_s_eff, g_r_eff = g_s - g_avg, self.g_r - g_avg
            else:
                g_s_eff, g_r_eff = g_s, self.g_r

            # Symmetric conversion terms
            conversion_s_to_r = self.conversion_rate * self.Xs[k]
            conversion_r_to_s = self.conversion_rate * self.Xr[k]
            
            # Updated differential equations with symmetric conversion
            dXs = (self.Xs[k] * g_s_eff - self.s_base * self.Xs[k] + 
                   self.input_s - conversion_s_to_r + conversion_r_to_s)
            dXr = (self.Xr[k] * g_r_eff - self.s_base * self.Xr[k] + 
                   self.input_r + conversion_s_to_r - conversion_r_to_s)
            
            self.Xs[k + 1] = max(self.Xs[k] + dXs * self.dt, 0.0)
            self.Xr[k + 1] = max(self.Xr[k] + dXr * self.dt, 0.0)
            
        return self.t, self.Xs, self.Xr

    def plot(self):
        N_total = self.Xs + self.Xr
        rel_Xs = np.divide(self.Xs, N_total, where=N_total > 0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

        ax1.plot(self.t, self.Xs, label="Susceptible", linewidth=2)
        ax1.plot(self.t, self.Xr, label="Resistant", linewidth=2)
        ax1.axvspan(self.ab_start, self.ab_end, color='grey', alpha=0.1, label="Selection Pressure")
        ax1.set_ylabel("Abundance (CFU/g)")
        ax1.set_title(f"Selection Pressure: {self.selection_pressure}, "
                     f"Mutualism Rate: {self.conversion_rate}")
        ax1.legend(loc="upper left", fontsize=6, frameon=False)
        
        ax2.stackplot(self.t, [rel_Xs, 1-rel_Xs], alpha=0.85)
        ax2.axvspan(self.ab_start, self.ab_end, color='grey', alpha=0.1)
        ax2.set_ylabel("Relative Abundance (%)")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")
    
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example with symmetric conversion
    sim = TaxaSimulator(selection_pressure=0.3, input_s=1.0, input_r=1.0, 
                       ab_start=20, ab_end=100, Xs0=1e5, Xr0=1e5,
                       conversion_rate=0.01)
    sim.run()
    sim.plot()
