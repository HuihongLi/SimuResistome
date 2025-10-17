import numpy as np
import matplotlib.pyplot as plt

class TaxaSimulator:
    def __init__(self, t_end=100, dt=0.01, Xs0=0, Xr0=0, g_base=0.8, fit_cost=1.0,
                 s_base=0.01, K_T=1e9, ab_start=20, ab_end=100,
                 pressure_type='constant', max_pressure=0.5, 
                 input_s=1.0, input_r=1.0, conversion_rate=0.002):
        """
        TaxaSimulator with time-dependent selective pressure functions.
        
        pressure_type: 'constant', 'decreasing', 'increasing', 'periodic'
        """
        self.t_end, self.dt = t_end, dt
        self.t = np.linspace(0, t_end, int(t_end / dt) + 1)
        self.g_base = g_base
        self.g_r = fit_cost * g_base  # fit_cost=1.0 means no fitness cost
        self.s_base, self.K_T = s_base, K_T
        self.ab_start, self.ab_end = ab_start, ab_end
        self.max_pressure = max_pressure
        self.pressure_type = pressure_type
        self.input_s, self.input_r = input_s, input_r
        self.conversion_rate = conversion_rate
        
        self.Xs, self.Xr = np.zeros_like(self.t), np.zeros_like(self.t)
        self.Xs[0], self.Xr[0] = Xs0, Xr0
        
        # Store pressure values for plotting
        self.pressure_values = np.zeros_like(self.t)

    def get_selection_pressure(self, t_current):
        """
        Calculate selection pressure based on time and pressure type.
        """
        if t_current < self.ab_start:
            return 0.0
        
        # Time since antibiotics started
        t_relative = t_current - self.ab_start
        duration = self.ab_end - self.ab_start
        
        if t_current > self.ab_end:
            t_relative = duration
        
        if self.pressure_type == 'constant':
            # Constant pressure
            return self.max_pressure
        
        elif self.pressure_type == 'decreasing':
            # Linear decrease to 0
            progress = min(t_relative / (0.2 * duration), 1.0)
            return self.max_pressure * (1 - progress)
        
        elif self.pressure_type == 'increasing':
            # Linear increase
            progress = min(t_relative / duration, 1.0)
            return self.max_pressure * progress
        
        elif self.pressure_type == 'periodic':
            # Periodic function (sine wave)
            frequency = 2 * np.pi / (duration / 4)  # 4 cycles during treatment
            return self.max_pressure * (0.5 + 0.5 * np.sin(frequency * t_relative))
        
        return 0.0

    def run(self):
        for k in range(len(self.t) - 1):
            current_pressure = self.get_selection_pressure(self.t[k])
            self.pressure_values[k] = current_pressure
            
            ab_active = self.ab_start <= self.t[k] <= self.ab_end
            
            # Apply selective pressure during antibiotic treatment
            g_s = self.g_base * (1 - current_pressure) if ab_active else self.g_base
            
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
        
        # Store last pressure value
        self.pressure_values[-1] = self.get_selection_pressure(self.t[-1])
        
        return self.t, self.Xs, self.Xr


def plot_comparison():
    """
    Plot comparison of 4 different selective pressure scenarios in 2x2 layout.
    Each subplot shows relative abundance (left axis) and selective pressure (right axis).
    """
    pressure_types = ['decreasing', 'constant', 'increasing', 'periodic']
    titles = ['Decreasing', 'Constant', 'Increasing', 'Periodic']
    
    # Common parameters
    params = {
        't_end': 200,
        'dt': 0.01,
        'Xs0': 1e5,
        'Xr0': 1e5,
        'g_base': 0.8,
        'fit_cost': 1.0,  # No fitness cost
        'ab_start': 20,
        'ab_end': 200,
        'max_pressure': 0.5,
        'input_s': 1.0,
        'input_r': 1.0,
        'conversion_rate': 0.01
    }
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    axes = axes.flatten()
    
    # Store handles for unified legend
    legend_handles = []
    legend_labels = []
    
    for i, (p_type, title) in enumerate(zip(pressure_types, titles)):
        # Run simulation
        sim = TaxaSimulator(pressure_type=p_type, **params)
        t, Xs, Xr = sim.run()
        
        N_total = Xs + Xr
        rel_Xs = np.divide(Xs, N_total, where=N_total > 0)
        
        # Create primary axis for relative abundance
        ax1 = axes[i]
        
        # Plot relative abundance as stacked area
        p1 = ax1.fill_between(t, 0, rel_Xs, alpha=0.7, color='#1f77b4', label='Susceptible')
        p2 = ax1.fill_between(t, rel_Xs, 1, alpha=0.7, color='#d62728', label='Resistant')
        
        ax1.set_xlabel("Time (days)", fontsize=10)
        ax1.set_ylabel("Relative Abundance", fontsize=10)
        ax1.set_ylim(0, 1)
        ax1.set_title(title, fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Create secondary axis for selective pressure
        ax2 = ax1.twinx()
        p4, = ax2.plot(t, sim.pressure_values, linewidth=2.5, color='black', 
                label='Selective Pressure', linestyle='--', alpha=0.8)
        ax2.set_ylabel("Selective Pressure", fontsize=10, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.set_ylim(-0.05, 0.6)
        
        # Collect legend handles from first subplot only
        if i == 0:
            legend_handles = [p1, p2, p4]
            legend_labels = ['Susceptible', 'Resistant', 'Selective Pressure']

    # Add unified legend outside the subplots
    fig.legend(legend_handles, legend_labels, 
              loc='upper center', bbox_to_anchor=(0.5, 0.98),
              ncol=4, fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # Run comparison plot
    plot_comparison()