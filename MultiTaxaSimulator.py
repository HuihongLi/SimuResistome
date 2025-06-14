#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.colors as mcolors


def _distinct_colors(n: int):
    """Return *n* visually distinct RGB colors."""
    if n <= 20:                                 # use the tab20 palette
        cmap = plt.cm.get_cmap('tab20', n)
        return [cmap(i) for i in range(n)]
    # evenly sample the hue wheel for larger n
    return [mcolors.hsv_to_rgb((i / n, 0.65, 0.9)) for i in range(n)]


class MultiTaxaSimulator:
    """gLV microbiome model with S/R sub-populations and global carrying capacity."""
    def __init__(self, n_taxa: int = 3, *, t_end=5000.0, dt=0.02,
                 Xs0=None, Xr0=None,
                 g_s_base=None, g_s_drug=None, g_r=None,
                 s_base=0.1, s_drug=1.0,
                 delta=0.001, mu=0.001,
                 K_T=1e9,
                 ab_start=20.0, ab_end=30.0, ab_type=2,
                 interaction_matrix=None, self_interaction=None,
                 rng_seed=None):

        if rng_seed is not None:
            np.random.seed(rng_seed)

        self.t = np.linspace(0, t_end, int(t_end / dt) + 1)
        self.dt = dt
        self.n_taxa = n_taxa

        Xs0 = np.full(n_taxa, 100.) if Xs0 is None else np.asarray(Xs0, float)
        Xr0 = np.full(n_taxa, 100.) if Xr0 is None else np.asarray(Xr0, float)

        g_s_base = np.full(n_taxa, 0.8) if g_s_base is None else np.asarray(g_s_base)
        g_s_drug = np.full(n_taxa, 0.2) if g_s_drug is None else np.asarray(g_s_drug)
        g_r      = np.full(n_taxa, 0.7) if g_r      is None else np.asarray(g_r)

        if interaction_matrix is None:
            interaction_matrix = np.random.normal(0, 1e-10, size=(n_taxa, n_taxa))
        interaction_matrix = np.asarray(interaction_matrix, float)
        if interaction_matrix.shape != (n_taxa, n_taxa):
            raise ValueError("interaction_matrix must be (n_taxa, n_taxa)")
        np.fill_diagonal(interaction_matrix, 0.0)
        self.A = interaction_matrix

        if self_interaction is None:
            self_interaction = np.full(n_taxa, 1e-8)
        self.self_interaction = np.asarray(self_interaction, float)

        self.Xs0, self.Xr0 = Xs0, Xr0
        self.g_s_base, self.g_s_drug, self.g_r = g_s_base, g_s_drug, g_r
        self.s_base, self.s_drug = s_base, s_drug
        self.delta, self.mu = delta, mu
        self.K_T = K_T
        self.ab_start, self.ab_end, self.ab_type = ab_start, ab_end, ab_type
        self.ab_label = {1: "Bacteriostatic (Type I)",
                         2: "Bactericidal (Type II)",
                         3: "Mixed Action (Type III)"}[ab_type]

        steps = len(self.t)
        self.Xs = np.zeros((steps, n_taxa))
        self.Xr = np.zeros((steps, n_taxa))
        self.Xs[0], self.Xr[0] = Xs0, Xr0

    # ------------------------------------------------------------------
    def run(self):
        for k, now in enumerate(self.t[:-1]):
            ab_on = self.ab_start <= now <= self.ab_end
            if self.ab_type == 1:
                g_s = np.where(ab_on, self.g_s_drug, self.g_s_base); s_s = self.s_base
            elif self.ab_type == 2:
                g_s = self.g_s_base; s_s = self.s_drug if ab_on else self.s_base
            else:                                       # mixed
                g_s = np.where(ab_on, self.g_s_drug, self.g_s_base)
                s_s = self.s_drug if ab_on else self.s_base
            s_r = self.s_base

            total = self.Xs[k] + self.Xr[k]
            N = total.sum()

            interaction = (self.A @ total) - self.self_interaction * total
            g_s_raw = g_s + interaction
            g_r_raw = self.g_r + interaction

            cap = 1.0 - N / self.K_T
            g_s_eff = g_s_raw * cap
            g_r_eff = g_r_raw * cap

            dXs = self.Xs[k] * g_s_eff - s_s * self.Xs[k] + self.delta * self.Xr[k]
            dXr = self.Xr[k] * g_r_eff - s_r * self.Xr[k] + self.mu    * self.Xs[k]

            self.Xs[k + 1] = np.maximum(self.Xs[k] + dXs * self.dt, 0.0)
            self.Xr[k + 1] = np.maximum(self.Xr[k] + dXr * self.dt, 0.0)

        return self.t, self.Xs, self.Xr, self.ab_label

    # ------------------------------------------------------------------
    def plot(self, taxa_labels=None, colors=None, show_interactions=False):
        if taxa_labels is None:
            taxa_labels = [f"Taxon {i+1}" for i in range(self.n_taxa)]
        if colors is None:
            colors = _distinct_colors(self.n_taxa)

        plt.figure(figsize=(11, 6))
        ax1 = plt.subplot(2, 1, 1)
        for i, (lbl, c) in enumerate(zip(taxa_labels, colors)):
            ax1.plot(self.t, self.Xs[:, i],  c=c,        label=f"{lbl} S")
            ax1.plot(self.t, self.Xr[:, i],  c=c, ls="--", label=f"{lbl} R")
        ax1.axvspan(self.ab_start, self.ab_end, color='red', alpha=0.10, label="Antibiotic")
        ax1.set_ylabel("Density [CFU mL⁻¹]")
        ax1.set_title(f"Absolute Abundance — {self.ab_label}")
        ax1.legend(ncol=3, fontsize=8)

        series = list(chain.from_iterable((self.Xs[:, i], self.Xr[:, i]) for i in range(self.n_taxa)))
        rel = np.vstack(series); rel /= rel.sum(axis=0, keepdims=True)
        labels_rel = list(chain.from_iterable((f"{lbl} S", f"{lbl} R") for lbl in taxa_labels))

        ax2 = plt.subplot(2, 1, 2)
        ax2.stackplot(self.t, rel, labels=labels_rel, alpha=0.85, colors=list(chain.from_iterable((c, c) for c in colors)))
        ax2.axvspan(self.ab_start, self.ab_end, color='red', alpha=0.10)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Relative Abundance")
        ax2.set_xlabel("Time [h]")
        ax2.set_title("Community Composition (Relative)")
        ax2.legend(ncol=3, fontsize=8, loc="upper right")

        if show_interactions:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax1, width="25%", height="40%", loc="upper right", borderpad=1)
            im = axins.imshow(self.A, cmap="bwr", interpolation='nearest')
            axins.set_title("a_ij"); axins.set_xticks([]); axins.set_yticks([])
            plt.colorbar(im, ax=axins, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n = 100                                            # ← 100 taxa
    # gLV interaction matrix: mean 0, σ = 1 × 10⁻⁸
    A = rng.normal(0.0, 1e-8, size=(n, n))
    np.fill_diagonal(A, 0.0)

    # initial abundances
    Xs0 = rng.integers(5e4, 5e5, size=n)               # 5×10⁴ – 5×10⁵
    Xr0 = rng.integers(1e4, 1e5, size=n)               # 1×10⁴ – 1×10⁵

    # growth rates: gently decreasing across taxa for variety
    g_s_base = np.linspace(1.05, 0.50, n)
    g_s_drug = g_s_base * 0.10                         # 90 % inhibition
    g_r      = g_s_base - 0.25                         # R a bit slower

    sim = MultiTaxaSimulator(
        n_taxa=n,
        Xs0=Xs0, Xr0=Xr0,
        g_s_base=g_s_base, g_s_drug=g_s_drug, g_r=g_r,
        ab_start=0, ab_end=0, ab_type=2,           # bactericidal course
        interaction_matrix=A,
        rng_seed=42,
    )

    sim.run()
    # A full legend for 100 taxa is unreadable; disable it for clarity:
    sim.plot(show_interactions=False, taxa_labels=[f"T{i+1}" for i in range(n)], colors=_distinct_colors(n))
