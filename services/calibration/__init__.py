"""AURORA WS2 calibration package.

WS2-prep: turns genuine raw TUM telemetry into 300 s calibration training rows
(physics baseline + measured SOC-drop residual target) with deterministic,
leak-free vehicle-level splitting. No ML model is fitted here — fitting happens
only after the prep artifacts are reviewed.
"""