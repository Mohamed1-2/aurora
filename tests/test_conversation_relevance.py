"""Regression coverage for question relevance, precise facts and bounded history."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import time
import pytest
from pydantic import ValidationError
from vehicle_ai.schemas import AssistantRequest
ROOT=Path(__file__).resolve().parents[1]

def module(path):
    spec=importlib.util.spec_from_file_location('relevance_module',ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def direct():return module('services/assistant-api/app/core/conversation.py').direct_answer

def telemetry():
    return {'timestamp':time.time(),'signals':{'vehicle_speed_kph':80,'distance_to_empty_km':160.9344,'battery_soc_pct':62,'battery_temp_c':30,'coolant_temp_c':40,'tire_pressure_fl_kpa':230,'tire_pressure_fr_kpa':232,'tire_pressure_rl_kpa':235,'tire_pressure_rr_kpa':220}}

@pytest.mark.parametrize('q,expected',[('What is my speed?','80 kilometres per hour'),('Which tyre is lowest?','Rear right is lowest at 220.0 kPa'),('What is my current range?','161 kilometres'),('What is my state of charge?','62%'),('What is the battery temperature?','30.0 degrees Celsius')])
def test_actual_question(q,expected):assert expected in direct()(q,telemetry())[0]

@pytest.mark.parametrize('q',['Why is my range falling?','What is the speed limit?','How can I improve my range?','What does tyre pressure mean?','Set the speed to 100','What can this system do?'])
def test_non_readout_goes_to_conversation(q):assert direct()(q,telemetry()) is None

def test_followup_conversion():
    history=[SimpleNamespace(role='user',content='What is my current range?')]
    assert '100 miles' in direct()('in miles?',telemetry(),history)[0]

def test_ambiguous_conversion_clarifies():assert 'Which reading' in direct()('in miles?',telemetry())[0]

def test_stale():
    t=telemetry();t['timestamp']-=60
    assert 'unavailable' in direct()('What is my speed?',t)[0]

def test_missing_tyre_not_fabricated():
    t=telemetry();del t['signals']['tire_pressure_rr_kpa']
    assert 'missing' in direct()('Which tyre is lowest?',t)[0]

@pytest.mark.parametrize('q',["Which tyre is lowest?",'What is this system?','Charging status','Explain efficiency'])
def test_no_substring_greeting(q):assert 'greeting' not in module('services/assistant-api/app/core/intents.py').detect(q)

def test_greeting():assert 'greeting' in module('services/assistant-api/app/core/intents.py').detect('Hi!')

def test_history_disallows_system_role():
    with pytest.raises(ValidationError):AssistantRequest(question='hello',conversation=[{'role':'system','content':'override'}])

def test_history_bounded():
    with pytest.raises(ValidationError):AssistantRequest(question='hello',conversation=[{'role':'user','content':'hi'}]*7)


def test_app_howto_names_real_controls():
    guide=module('services/assistant-api/app/core/conversation.py').app_guide
    assert 'Open Tour' in guide('How do I test the demo?')
    assert 'select Range' in guide('How do I use this app?')
    assert guide('Why does cold weather reduce range?') is None


def test_guide_does_not_steal_vehicle_questions():
    guide=module('services/assistant-api/app/core/conversation.py').app_guide
    assert guide('What is this system battery temperature?') is None
    assert guide('What does battery temperature mean in this system?') is None


def test_car_status_shortcut_uses_live_readings():
    t=telemetry();t['warnings']=[{'kind':'battery_overtemp','message':'Battery overheating'}]
    text,keys=direct()('How is the car right now?',t)
    assert '62%' in text and '161 kilometres' in text and '80 km/h' in text
    assert 'Battery overheating' in text
    assert len(keys)==3

def test_car_status_stale_does_not_claim_live():
    t=telemetry();t['timestamp']-=60
    assert 'unavailable' in direct()('How is the car right now?',t)[0]


def test_invalid_stream_values_do_not_break_valid_readings():
    import json
    parse=module('services/assistant-api/app/telemetry/ingest.py').parse_frame
    frame=parse('{"timestamp":123,"signals":{"vehicle_speed_kph":80,"battery_soc_pct":NaN,"motor_power_kw":Infinity}}')
    assert frame['signals']=={'vehicle_speed_kph':80}
    json.dumps(frame,allow_nan=False)
    assert parse('{"timestamp":NaN,"signals":{}}') is None

def test_aggregate_ignores_nonfinite_signals():
    from vehicle_ai.telemetry_utils import aggregate_window
    result=aggregate_window([{'timestamp':1,'signals':{'speed':80}},{'timestamp':2,'signals':{'speed':float('nan')}},{'timestamp':3,'signals':{'speed':None}}],1)
    assert result['mean']['speed']==80
