# Diagnostics and Trouble Codes

## DTC overview

Diagnostic Trouble Codes (DTCs) are OBD-II style five-character codes the vehicle stores when a subsystem reports out-of-range behaviour. P-codes relate to powertrain and emissions, B-codes to body and restraints, C-codes to chassis and braking, U-codes to network communication.

## Common codes

P0420 Catalyst system efficiency below threshold indicates degraded catalytic converter performance, usually from age or misfueling; continued driving is generally possible but emissions and fuel economy suffer and a service check is advised within 500 km. U0100 Lost communication with ECM/PCM indicates a CAN-bus communication drop, which can cause instrument warnings and degraded control; the vehicle should be serviced promptly and driving limited. B0131 Driver frontal airbag circuit short to ground is a safety-critical restraint fault: the airbag may not deploy and the vehicle must be serviced immediately; the vehicle should not be driven until the SRS warning is resolved. P0171 System too lean indicates the fuel-air mixture is running lean, typically from a vacuum leak or sensors; driving is possible but service is recommended. C0031 Left front wheel speed sensor circuit failure degrades ABS and stability control, which materially increases stopping distances on low-grip surfaces; drive cautiously and service promptly. P0B36 Hybrid battery pack voltage inconsistent indicates a potential isolation or cell issue in the high-voltage pack; charging should stop and a certified technician should inspect the vehicle.

## What the assistant does

When a new DTC appears in the telemetry stream, the assistant triages it against this catalog, prioritising safety-critical codes (restraints, braking, high-voltage) above convenience codes, and recommends an action within the driver's reach: continue, plan service, or stop driving.