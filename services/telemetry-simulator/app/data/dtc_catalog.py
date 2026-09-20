"""OBD-II style diagnostic trouble codes + signal catalog.

These are synthetic but OBD-II realistic — they give the assistant the same
kind of data a real vehicle bus would carry and the RAG knowledge base the
same kind of content an owner's manual would.
"""

DTC_CATALOG: dict[str, dict] = {
    "P0420": {"description": "Catalyst system efficiency below threshold (Bank 1)", "system": "Emissions", "severity": "warning"},
    "U0100": {"description": "Lost communication with ECM/PCM", "system": "CAN Bus / Electrical", "severity": "warning"},
    "B0131": {"description": "Driver frontal airbag circuit short to ground", "system": "Restraints", "severity": "critical"},
    "P0171": {"description": "System too lean (Bank 1)", "system": "Fuel / Powertrain", "severity": "warning"},
    "C0031": {"description": "Left front wheel speed sensor circuit failure — ABS degraded", "system": "Braking", "severity": "critical"},
    "P0B36": {"description": "Hybrid battery pack voltage inconsistent", "system": "High Voltage", "severity": "critical"},
}

SIGNAL_CATALOG: dict[str, tuple[str, str, float, float]] = {
    # key: (unit, description, min, max)
    "vehicle_speed_kph": ("km/h", "Road speed", 0, 220),
    "motor_power_kw": ("kW", "Traction motor power (negative = regen)", -165, 165),
    "regen_power_kw": ("kW", "Regenerative braking power", -55, 0),
    "battery_soc_pct": ("%", "State of charge", 0, 100),
    "battery_temp_c": ("degC", "High voltage battery temperature", -20, 60),
    "coolant_temp_c": ("degC", "Powertrain coolant temperature", 0, 120),
    "distance_to_empty_km": ("km", "Predicted range", 0, 900),
    "tire_pressure_fl_kpa": ("kPa", "Front-left tire pressure", 60, 400),
    "tire_pressure_fr_kpa": ("kPa", "Front-right tire pressure", 60, 400),
    "tire_pressure_rl_kpa": ("kPa", "Rear-left tire pressure", 60, 400),
    "tire_pressure_rr_kpa": ("kPa", "Rear-right tire pressure", 60, 400),
    "outside_temp_c": ("degC", "Ambient temperature", -40, 60),
    "charge_rate_kw": ("kW", "Charging power while plugged in", 0, 150),
    "battery_voltage_v": ("V", "HV battery pack voltage", 250, 400),
    "pack_current_a": ("A", "HV battery pack current (positive = discharge)", -500, 500),
    "cabin_set_temp_c": ("degC", "Cabin setpoint temperature", 16, 28),
    "odometer_km": ("km", "Lifetime odometer", 0, 1_000_000),
    "cabin_temp_c": ("degC", "Measured cabin temperature", -10, 45),
    "cabin_humidity_pct": ("%", "Cabin relative humidity", 0, 100),
    "trip_distance_km": ("km", "Trip distance", 0, 1000),
    "energy_session_kwh": ("kWh", "Energy used this session", 0, 100),
    "charge_port_latched": ("bin", "Charge port latched", 0, 1),
}