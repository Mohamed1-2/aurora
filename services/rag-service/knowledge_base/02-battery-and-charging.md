# Battery and Charging

## State of charge and range

Range is predicted from usable battery capacity, current state of charge and a rolling energy-consumption average measured in Wh/km. Regen charging and driving style change the Wh/km average over the last 400 km of driving, so predicted range always reflects recent behaviour. A 100% charge provides up to 420 km of mixed driving; 80% provides roughly 335 km.

## DC fast charging

Connect to a DC rapid charger for up to 150 kW. The battery warms internally during rapid charging and cooling circuits run continuously. Below 10% SOC, the charge curve intentionally throttles to protect the pack. Expect 10-80% in about 33 minutes on a 150 kW unit.

## AC home charging

A 11 kW single-phase or 22 kW three-phase AC charger fully refills the battery at home overnight. The vehicle reports charge power in the telemetry stream as charge_rate_kw. Charging ends automatically at the configured target (default 80%).

## Battery thermal management

Optimal battery temperature band is 20-35 C. The cooling system keeps the pack inside this window; sustained Sport mode driving or hot-environment fast charging raises pack temperature, which the system reports as battery_temp_c. If battery_temp_c exceeds 42 C the assistant must warn the driver and recommend reduced load or charging, since sustained high temperatures accelerate pack degradation.

## Battery lifecycle

The high-voltage pack is warrantied to retain at least 70% of original usable capacity for 8 years or 160,000 km. To preserve capacity, avoid storing the vehicle for more than 30 days at SOC above 60%, reduce DC fast charging in the hottest months, and keep the battery in its optimal thermal band for normal driving.