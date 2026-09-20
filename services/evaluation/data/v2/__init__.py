"""AURORA v2 dataset pipeline: real-telemetry ingestors + scenario builder."""

from .base import ChargingSession, DrivingWindow, Provenance
from .tum_driving import TUMDrivingIngestor, SIGNAL_MAP, DRIVING_SIGNAL_DOC
from .epfl_charging import EPFLChargingIngestor

__all__ = [
    "ChargingSession",
    "DrivingWindow",
    "Provenance",
    "TUMDrivingIngestor",
    "EPFLChargingIngestor",
    "SIGNAL_MAP",
    "DRIVING_SIGNAL_DOC",
]
