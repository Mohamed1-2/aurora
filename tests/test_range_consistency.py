import importlib.util, time, pytest
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('conversation',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'services/assistant-api/app/core/conversation.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def frame(soc=.3, distance=.4):return {'timestamp':time.time(),'signals':{'battery_soc_pct':soc,'distance_to_empty_km':distance}}
@pytest.mark.parametrize('q',['how many kilometer I can drive','How many kilometers can I drive?','How far can we go?','What is my distance to empty?','How many miles are left?','What is my remaining range?'])
def test_paraphrases(q):
    answer,keys=m.direct_answer(q,frame())
    assert '105' not in answer and 'distance_to_empty_km' in keys
    assert '0.' in answer

def test_charge_precision():assert '0.3%' in m.direct_answer('what is the battery level',frame())[0]
def test_contradiction():assert 'disagree' in m.direct_answer('how many kilometer I can drive',frame(0,105))[0]
def test_empty():assert '0 kilometres' in m.direct_answer('What is my range?',frame(0,0))[0]
def test_stale():
    f=frame();f['timestamp']-=20
    assert 'unavailable' in m.direct_answer('how many kilometer I can drive',f)[0]
def test_conversion_uses_fresh_reading():
    history=[SimpleNamespace(role='user',content='how many kilometer I can drive'),SimpleNamespace(role='assistant',content='105 kilometres')]
    assert '0.2 miles' in m.direct_answer('in miles?',frame(),history)[0]
