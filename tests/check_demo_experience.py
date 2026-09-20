import sys,time
sys.path.insert(0,'services/telemetry-simulator')
from app.engine.vehicle_model import VehicleModel
m=VehicleModel('TEST',seed=42);m.battery_soc=0;m.speed=80
for _ in range(600):f=m.tick(.1)
assert m.speed==0, m.speed
assert f['signals']['engine_power_kw']==0
assert m._depleted
for _ in range(300):m.tick(.1)
assert m.speed==0
m.battery_soc=2
m.tick(.1)
assert not m._depleted
print('Empty battery stops and stays parked; recharge releases latch; no engine assist')
import importlib.util
s=importlib.util.spec_from_file_location('conv','services/assistant-api/app/core/conversation.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
f={'timestamp':time.time(),'signals':{'battery_soc_pct':65,'distance_to_empty_km':100,'destination_remaining_km':8}}
assert '92 kilometres' in c.direct_answer('Can I reach my destination?',f)[0]
f['signals']['distance_to_empty_km']=2
assert 'charging stop' in c.direct_answer('Can I reach my destination?',f)[0]
f['signals']['destination_remaining_km']=-1
assert 'Set a destination' in c.direct_answer('Can I reach my destination?',f)[0]
assert 'Which reading' in c.direct_answer('Why?',f)[0]
print('Destination coverage, shortfall, missing destination and ambiguous follow-up passed')
