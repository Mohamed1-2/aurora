"""TUM-compatible physics baseline for WS2.

Uses WS1 ``simparams`` constants (loaded from the versioned ws1-v1 profile) and
the same wheel-drag formula as the WS1 energy engine (``_motive_power_w``), with
deliberate real-data substitutions:

* aux load = **measured** ``hv_aux_power`` (never the sim's assumed 350 W base);
* hvac = 0 — no cabin setpoint/interior temp on TUM (never invented);
* motive = aero + rolling + inertia at the wheel, clipped at 0 per bin
  (conservative: no regen credit and no drivetrain-efficiency factor).

Convention (locked): ``soc_drop_pct = soc_start_pct - soc_end_pct`` is positive
while discharging; ``physics_soc_drop_pct`` and ``residual_pct`` use it.
"""

from __future__ import annotations

from ._bootstrap import load_ws1_engine


def load_simparams_profile():
    import sys as _sys
    return _sys.modules["aurora_ws1.simparams"].DEFAULT_PROFILE


class PhysicsBaseline:
    """Wraps the ws1-v1 simparams profile + WS1 wheel-drag formula."""

    def __init__(self) -> None:
        engine = load_ws1_engine()
        self.profile = load_simparams_profile()
        self._motive_power_w = engine._motive_power_w
        self.g = self.profile.value("g_mps2")
        self.mass_kg = self.profile.value("mass_kg")
        self.crr = self.profile.value("crr")
        self.cd = self.profile.value("cd")
        self.area = self.profile.value("frontal_area_m2")
        self.rho = self.profile.value("rho_air_kgm3")

    def motive_wh(self, bins: list[dict], bin_s: float = 10.0) -> float:
        wh = 0.0
        v_prev = 0.0
        for i, b in enumerate(bins):
            v = max(0.0, b.get("speed_kph", 0.0)) / 3.6
            accel = (v - v_prev) / bin_s if i > 0 else 0.0
            p = self._motive_power_w(v * 3.6, accel)[0]
            wh += max(0.0, p) * bin_s / 3600.0
            v_prev = v
        return wh

    @staticmethod
    def aux_wh(bins: list[dict], bin_s: float = 10.0) -> float:
        return sum(max(0.0, b.get("hv_aux_power_w", 0.0)) for b in bins) * bin_s / 3600.0

    def energy_wh(self, bins: list[dict], bin_s: float = 10.0) -> tuple[float, float, float]:
        m = self.motive_wh(bins, bin_s)
        a = self.aux_wh(bins, bin_s)
        return m, a, m + a  # hvac = 0

    def soc_drop_phys_pct(self, bins: list[dict], capacity_kwh: float, bin_s: float = 10.0) -> float:
        _, _, total_wh = self.energy_wh(bins, bin_s)
        return round(total_wh / (capacity_kwh * 1000.0) * 100.0, 4)