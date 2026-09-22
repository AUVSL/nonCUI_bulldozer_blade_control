"""
Calibration and model-fitting utilities for BulldozerSimulation.
These are legacy / analysis tools kept separate to reduce clutter in main.py.
"""
from __future__ import annotations
from typing import Optional
from sim_types import FloatArray, Scalar

import os
from math import comb as _comb

import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


class BulldozerCalibration:
    """Inherits from BulldozerSimulation; adds sweep, threshold, and torque-fit tools."""

    def plot_track_force_sweep(self) -> None:
        """Overlay XY trajectories: left track fixed at F_track_base, right track swept over 100 steps from 0 to 1 x F_track_base."""
        from main import BulldozerSimulation

        fractions = np.linspace(0, 1, 100)
        colors = plt.cm.viridis(np.linspace(0, 1, 100))

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, fraction in enumerate(fractions):
            sim = BulldozerSimulation()
            sim.F_track[0] = sim.F_track_base
            sim.F_track[1] = fraction * sim.F_track_base
            sim.run()

            data = np.array(sim.log)
            ax.plot(data[:, 1], data[:, 2], color=colors[i], linewidth=0.8, alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=ax, label="right track fraction of F_base")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("XY trajectories — left track = F_base, right track = 0 to 1 x F_base (100 steps)")
        ax.set_aspect("equal")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("figures/track_force_sweep.png", dpi=150)
        plt.close(fig)
        print("Saved track_force_sweep.png")

    def threshold_run(self, fraction: Scalar) -> Scalar:
        from main import BulldozerSimulation

        sim = BulldozerSimulation()
        sim.F_track[0] = sim.F_track_base
        sim.F_track[1] = fraction * sim.F_track_base
        sim.run()
        data = np.array(sim.log)
        return np.max(np.abs(data[:, 2]))   # peak |y| displacement

    def find_straight_threshold(self, tol: float=1e-4, n_iter: int=1000) -> float:
        """Binary search for the largest fraction where peak |y| > tol."""
        lo, hi = 0.0, 1.0
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            if self.threshold_run(mid) > tol:
                lo = mid
            else:
                hi = mid
        print(f"Converged to straight threshold: {(lo + hi) / 2}")
        return (lo + hi) / 2

    def build_4pl(self, threshold: float=0.799, n_samples: int=100) -> tuple[FloatArray, tuple[Scalar, Scalar, Scalar], float, Scalar]:
        """Fit 4PL-q25/50/75 piecewise-linear model and return (coeffs, knots, threshold, ang_max)."""
        from main import BulldozerSimulation

        fractions = np.linspace(0.0, threshold, n_samples)
        yaws = []
        for frac in fractions:
            sim = BulldozerSimulation()
            sim.stop_time  = 0.5
            sim.F_track[0] = sim.F_track_base
            sim.F_track[1] = frac * sim.F_track_base
            sim.run()
            yaws.append(np.array(sim.log)[-1, 6])
        ang     = np.abs(np.array(yaws))
        ang_max = ang.max()
        T       = threshold - fractions

        k1, k2, k3 = np.percentile(ang, [25, 50, 75])
        X = np.c_[ang,
                  np.maximum(ang - k1, 0),
                  np.maximum(ang - k2, 0),
                  np.maximum(ang - k3, 0)]
        coeffs, _, _, _ = np.linalg.lstsq(X, T, rcond=None)
        print(f"4PL built  ang_max={ang_max:.4f} rad  knots=({k1:.4f},{k2:.4f},{k3:.4f})  "
              f"coeffs={np.array2string(coeffs, precision=4)}")
        return coeffs, (k1, k2, k3), threshold, ang_max

    def fit_torque(self, threshold: float=0.799, n_samples: int=600) -> None:
        """
        Fit angle→torque-fraction mappings using three approaches:
          1. Single-term basis functions (power/log/exp/trig)
          2. Piecewise linear/quadratic with hinge basis
          3. Bezier (Bernstein) curves degrees 2–8
        Simulation data is generated once and shared across all three fits.
        """
        from main import BulldozerSimulation

        # ── Shared data generation ─────────────────────────────────────────────
        fractions = np.linspace(0.0, threshold, n_samples)
        yaws = []
        for frac in fractions:
            sim = BulldozerSimulation()
            sim.F_track[0] = sim.F_track_base
            sim.F_track[1] = frac * sim.F_track_base
            sim.run()
            yaws.append(np.array(sim.log)[-1, 6])
        yaws    = np.array(yaws)
        T       = threshold - fractions
        ang     = np.abs(yaws)
        ang_max = ang.max()

        # ── 1. Single-term basis fits ──────────────────────────────────────────
        def _b(x: FloatArray, name: str) -> FloatArray:
            return {
                "ang^0.25":             x ** 0.25,
                "ang^0.5":              x ** 0.5,
                "ang^(2/3)":            x ** (2.0/3.0),
                "ang^0.75":             x ** 0.75,
                "ang^1":                x,
                "ang^1.25":             x ** 1.25,
                "ang^1.5":              x ** 1.5,
                "ang^(5/3)":            x ** (5.0/3.0),
                "ang^2":                x ** 2,
                "ang^2.5":              x ** 2.5,
                "ang^3":                x ** 3,
                "log(ang+1)":           np.log(x + 1),
                "log(ang+1)^2":         np.log(x + 1) ** 2,
                "ang*log(ang+1)":       x * np.log(x + 1),
                "ang^2*log(ang+1)":     x ** 2 * np.log(x + 1),
                "ang/log(ang+2)":       x / np.log(x + 2),
                "exp(ang)-1":           np.exp(x) - 1,
                "exp(ang)-1-ang":       np.exp(x) - 1 - x,
                "exp(ang^0.5)-1":       np.exp(x ** 0.5) - 1,
                "exp(ang^0.75)-1":      np.exp(x ** 0.75) - 1,
                "exp(ang^1.25)-1":      np.exp(x ** 1.25) - 1,
                "exp(ang^1.5)-1":       np.exp(x ** 1.5) - 1,
                "exp(ang^2)-1":         np.exp(x ** 2) - 1,
                "(exp(ang)-1)^0.5":     np.sqrt(np.exp(x) - 1),
                "(exp(ang)-1)^1.5":     (np.exp(x) - 1) ** 1.5,
                "(exp(ang)-1)^2":       (np.exp(x) - 1) ** 2,
                "ang*exp(ang)":         x * np.exp(x),
                "ang^2*exp(ang)":       x ** 2 * np.exp(x),
                "sinh(ang)":            np.sinh(x),
                "sinh(ang^0.5)":        np.sinh(x ** 0.5),
                "sinh(ang^0.75)":       np.sinh(x ** 0.75),
                "sinh(ang^1.5)":        np.sinh(x ** 1.5),
                "sinh(ang^2)":          np.sinh(x ** 2),
                "cosh(ang)-1":          np.cosh(x) - 1,
                "cosh(ang^0.5)-1":      np.cosh(x ** 0.5) - 1,
                "sinh(ang)*ang":        np.sinh(x) * x,
                "tanh(ang)":            np.tanh(x),
                "ang/tanh(ang+1e-9)-1": x / np.tanh(x + 1e-9) - 1,
                "1-cos(ang)":           1 - np.cos(x),
            }[name]

        all_names = [
            "ang^0.25","ang^0.5","ang^(2/3)","ang^0.75","ang^1",
            "ang^1.25","ang^1.5","ang^(5/3)","ang^2","ang^2.5","ang^3",
            "log(ang+1)","log(ang+1)^2","ang*log(ang+1)","ang^2*log(ang+1)","ang/log(ang+2)",
            "exp(ang)-1","exp(ang)-1-ang",
            "exp(ang^0.5)-1","exp(ang^0.75)-1","exp(ang^1.25)-1","exp(ang^1.5)-1","exp(ang^2)-1",
            "(exp(ang)-1)^0.5","(exp(ang)-1)^1.5","(exp(ang)-1)^2",
            "ang*exp(ang)","ang^2*exp(ang)",
            "sinh(ang)","sinh(ang^0.5)","sinh(ang^0.75)","sinh(ang^1.5)","sinh(ang^2)",
            "cosh(ang)-1","cosh(ang^0.5)-1","sinh(ang)*ang",
            "tanh(ang)","ang/tanh(ang+1e-9)-1","1-cos(ang)",
        ]

        print(f"\n{'Function':<24}  {'RMSE':>10}  {'MAE':>10}  Coefficients")
        print("-" * 80)
        results_single = {}
        for name in all_names:
            basis = _b(ang, name)
            denom = np.dot(basis, basis)
            A = np.dot(basis, T) / denom if denom > 0 else 0.0
            residuals = T - A * basis
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae  = np.mean(np.abs(residuals))
            results_single[name] = (A, rmse, mae)
            print(f"{name:<24}  {rmse:>10.6f}  {mae:>10.6f}  [{A:.6f}]")

        ranked_single = sorted(results_single, key=lambda k: results_single[k][1])
        best_single   = ranked_single[0]
        A_best        = results_single[best_single][0]
        best_rmse, best_mae = results_single[best_single][1], results_single[best_single][2]
        print(f"\n{'Best fit':<24}  {best_rmse:>10.6f}  {best_mae:>10.6f}  [{A_best:.6f}]")
        print(f"  fraction(yaw) = {threshold:.4f} - {A_best:.6f} * ({best_single})")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle("Angle → torque fraction fits", fontsize=13)

        ang_dense  = np.linspace(1e-9, ang_max, 300)
        colors_top = plt.cm.tab10(np.linspace(0, 1, 10))
        ax1.scatter(ang, fractions, color='black', s=20, zorder=5, label='simulation data')
        for i, name in enumerate(ranked_single[:10]):
            A = results_single[name][0]
            ax1.plot(ang_dense, threshold - A * _b(ang_dense, name), linewidth=1.5,
                     linestyle='-' if name == best_single else '--', color=colors_top[i],
                     label=f"{name}  ({results_single[name][1]:.4f})")
        ax1.axhline(threshold, color='grey', linestyle=':', linewidth=1, label=f'threshold={threshold}')
        ax1.set_xlabel("|Final yaw angle| (rad)")
        ax1.set_ylabel("Required fraction  (F_right / F_base)")
        ax1.set_title("Single-term basis (top 10)")
        ax1.legend(fontsize=7)
        ax1.grid(True)

        # ── 2. Piecewise fits ──────────────────────────────────────────────────
        def H(x: FloatArray, k: Scalar) -> FloatArray:
            return np.maximum(x - k, 0)

        def _fit(X: FloatArray) -> tuple[FloatArray, Scalar, Scalar]:
            c, _, _, _ = np.linalg.lstsq(X, T, rcond=None)
            r = T - X @ c
            return c, np.sqrt(np.mean(r**2)), np.mean(np.abs(r))

        knots = {
            'q10': np.percentile(ang, 10),
            'q25': np.percentile(ang, 25),
            'q33': np.percentile(ang, 33),
            'q50': np.percentile(ang, 50),
            'q67': np.percentile(ang, 67),
            'q75': np.percentile(ang, 75),
            'q90': np.percentile(ang, 90),
        }
        results_pw = {}
        for kn, k in knots.items():
            c, rmse, mae = _fit(np.c_[ang,      H(ang, k)      ]); results_pw[f"2PL-{kn}"]     = (c, rmse, mae, '2PL',  k)
            c, rmse, mae = _fit(np.c_[ang**2,   H(ang, k)**2   ]); results_pw[f"PQ2-{kn}"]     = (c, rmse, mae, 'PQ2',  k)
            c, rmse, mae = _fit(np.c_[ang,      H(ang, k)**2   ]); results_pw[f"L+Q-{kn}"]     = (c, rmse, mae, 'LQ',   k)
            c, rmse, mae = _fit(np.c_[ang**1.5, H(ang, k)      ]); results_pw[f"P1.5+PL-{kn}"] = (c, rmse, mae, 'P15L', k)
            c, rmse, mae = _fit(np.c_[ang**2,   H(ang, k)      ]); results_pw[f"P2+PL-{kn}"]   = (c, rmse, mae, 'P2L',  k)
            c, rmse, mae = _fit(np.c_[np.sinh(ang), H(ang, k)  ]); results_pw[f"sinh+PL-{kn}"] = (c, rmse, mae, 'sinhL',k)
            c, rmse, mae = _fit(np.c_[np.exp(ang)-1, H(ang, k) ]); results_pw[f"exp+PL-{kn}"]  = (c, rmse, mae, 'expL', k)

        for lbl, k1, k2 in [
            ('q25+q75', np.percentile(ang,25), np.percentile(ang,75)),
            ('q33+q67', np.percentile(ang,33), np.percentile(ang,67)),
            ('q20+q60', np.percentile(ang,20), np.percentile(ang,60)),
            ('q40+q80', np.percentile(ang,40), np.percentile(ang,80)),
        ]:
            c, rmse, mae = _fit(np.c_[ang, H(ang,k1), H(ang,k2)])
            results_pw[f"3PL-{lbl}"] = (c, rmse, mae, '3PL', (k1,k2))

        k1, k2, k3 = np.percentile(ang, [25, 50, 75])
        c, rmse, mae = _fit(np.c_[ang, H(ang,k1), H(ang,k2), H(ang,k3)])
        results_pw["4PL-q25/50/75"] = (c, rmse, mae, '4PL', (k1,k2,k3))

        ranked_pw = sorted(results_pw, key=lambda k: results_pw[k][1])
        print(f"\n{'Function':<22}  {'RMSE':>10}  {'MAE':>10}  Coefficients")
        print("-" * 80)
        for name in ranked_pw[:15]:
            c, rmse, mae, *_ = results_pw[name]
            print(f"{name:<22}  {rmse:>10.6f}  {mae:>10.6f}  [{'  '.join(f'{v:.5f}' for v in c)}]")
        best_pw = ranked_pw[0]
        print(f"\nBest: {best_pw}  RMSE={results_pw[best_pw][1]:.6f}")

        def build_X(name: str, x: FloatArray) -> Optional[FloatArray]:
            ftype, extra = results_pw[name][3], results_pw[name][4]
            k = extra
            if ftype == '2PL':   return np.c_[x,           H(x, k)           ]
            if ftype == 'PQ2':   return np.c_[x**2,        H(x, k)**2        ]
            if ftype == 'LQ':    return np.c_[x,            H(x, k)**2        ]
            if ftype == 'P15L':  return np.c_[x**1.5,      H(x, k)           ]
            if ftype == 'P2L':   return np.c_[x**2,        H(x, k)           ]
            if ftype == 'sinhL': return np.c_[np.sinh(x),  H(x, k)           ]
            if ftype == 'expL':  return np.c_[np.exp(x)-1, H(x, k)           ]
            if ftype == '3PL':   k1,k2=k;    return np.c_[x, H(x,k1), H(x,k2)]
            if ftype == '4PL':   k1,k2,k3=k; return np.c_[x, H(x,k1), H(x,k2), H(x,k3)]

        ad     = np.linspace(0, ang_max, 400)
        colors = plt.cm.tab10(np.linspace(0, 1, 6))
        ax2.scatter(ang, fractions, color='black', s=20, zorder=5, label='data')
        for i, name in enumerate(ranked_pw[:6]):
            c = results_pw[name][0]
            ax2.plot(ad, threshold - build_X(name, ad) @ c, color=colors[i], linewidth=1.5,
                     linestyle='-' if name == best_pw else '--',
                     label=f"{name}  ({results_pw[name][1]:.5f})")
        ax2.axhline(threshold, color='grey', linestyle=':', linewidth=1, label=f'threshold={threshold:.3f}')
        ax2.set_xlabel("|Final yaw angle| (rad)")
        ax2.set_ylabel("Required fraction  (F_right / F_base)")
        ax2.set_title("Piecewise fits (top 6)")
        ax2.legend(fontsize=7)
        ax2.grid(True)

        # ── 3. Bezier fits ─────────────────────────────────────────────────────
        t = ang / ang_max

        def bernstein(t_vec: FloatArray, n: int) -> FloatArray:
            B = np.zeros((len(t_vec), n + 1))
            for i in range(n + 1):
                B[:, i] = _comb(n, i) * t_vec**i * (1 - t_vec)**(n - i)
            return B

        results_bz = {}
        for deg in range(2, 9):
            B = bernstein(t, deg)

            c, _, _, _ = np.linalg.lstsq(B, fractions, rcond=None)
            r = fractions - B @ c
            results_bz[f"Bezier-{deg} (free)"] = (c, np.sqrt(np.mean(r**2)), np.mean(np.abs(r)), deg, 'free')

            rhs  = fractions - threshold * B[:, 0]
            Bint = B[:, 1:-1]
            if Bint.shape[1] > 0:
                c_int, _, _, _ = np.linalg.lstsq(Bint, rhs, rcond=None)
                c_full = np.concatenate([[threshold], c_int, [0.0]])
                r = fractions - B @ c_full
                results_bz[f"Bezier-{deg} (pinned)"] = (
                    c_full, np.sqrt(np.mean(r**2)), np.mean(np.abs(r)), deg, 'pinned')

        ranked_bz = sorted(results_bz, key=lambda k: results_bz[k][1])
        print(f"\n{'Model':<24}  {'params':>6}  {'RMSE':>10}  {'MAE':>10}")
        print("-" * 58)
        for name in ranked_bz:
            c, rmse, mae, deg, mode = results_bz[name]
            n_params = len(c) if mode == 'free' else len(c) - 2
            print(f"{name:<24}  {n_params:>6}  {rmse:>10.6f}  {mae:>10.6f}")
        best_bz = ranked_bz[0]
        print(f"\nBest: {best_bz}  RMSE={results_bz[best_bz][1]:.6f}")
        print(f"  (4PL piecewise baseline RMSE={results_pw['4PL-q25/50/75'][1]:.6f})")
        print(f"  (exp(ang^2)-1 single-basis baseline RMSE={results_single['exp(ang^2)-1'][1]:.6f})")

        t_dense   = np.linspace(0, 1, 400)
        ang_dense = t_dense * ang_max
        colors    = plt.cm.tab10(np.linspace(0, 1, 6))
        ax3.scatter(ang, fractions, color='black', s=20, zorder=5, label='simulation data')
        for i, name in enumerate(ranked_bz[:6]):
            c, rmse, mae, deg, mode = results_bz[name]
            ax3.plot(ang_dense, bernstein(t_dense, deg) @ c, color=colors[i], linewidth=1.5,
                     linestyle='-' if name == best_bz else '--',
                     label=f"{name}  (RMSE={rmse:.5f})")
        ax3.axhline(threshold, color='grey', linestyle=':', linewidth=1, label=f'threshold={threshold:.3f}')
        ax3.set_xlabel("|Final yaw angle| (rad)")
        ax3.set_ylabel("Required fraction  (F_right / F_base)")
        ax3.set_title("Bezier fits (top 6)")
        ax3.legend(fontsize=7)
        ax3.grid(True)

        fig.tight_layout()
        plt.savefig("figures/torque_fits.png", dpi=150)
        plt.close(fig)
        print("Saved torque_fits.png")


# ── convenience entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from main import BulldozerSimulation

    class _Runner(BulldozerCalibration, BulldozerSimulation):
        pass

    r = _Runner()
    r.plot_track_force_sweep()
    r.find_straight_threshold()
    r.fit_torque()
