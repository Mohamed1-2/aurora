"""Challenge the evaluation rubric with deliberately wrong answers."""
import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/evaluate_replay.py'
spec=importlib.util.spec_from_file_location('replay_score',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_wrong_unit_is_not_correct_number():
 assert not m.score({'answer':'Range is 50 miles.'},{'numbers':[{'value':50,'unit':'km','tolerance':.5}]})['passed']
def test_outside_tolerance_fails():
 assert not m.score({'answer':'Charge is 63%.'},{'numbers':[{'value':62,'unit':'percent','tolerance':.5}]})['passed']
def test_extra_numeric_claim_fails():
 assert not m.score({'answer':'Charge is 62%; range is 400 km.'},{'numbers':[{'value':62,'unit':'percent'}]})['passed']
def test_decimal_and_units():
 assert m.score({'answer':'Battery is 0.2 percent.'},{'numbers':[{'value':.2,'unit':'percent','tolerance':.01}]})['passed']
def test_keyword_alone_not_sufficient():
 assert not m.score({'answer':'Range is 99 km.'},{'required_groups':[['range']],'numbers':[{'value':50,'unit':'km'}]})['passed']
def test_refusal_flag_required():
 assert not m.score({'answer':'No.','refused':False},{'refused':True})['passed']
def test_no_numbers_in_missing_data():
 assert not m.score({'answer':'Unavailable, perhaps 105.'},{'no_numbers':True})['passed']
def test_human_semantics_never_auto_passes():
 assert m.score({'answer':'Hello.'},{})['human_semantics']=='pending'

def test_extra_number_without_unit_requires_review():
 assert not m.score({'answer':'Charge is 62%. You can drive 400.'},{'numbers':[{'value':62,'unit':'percent'}]})['passed']

def test_equivalent_limitation_contractions():
 for answer in ["I cannot provide live data.","I can't provide live data.","I can’t provide live data."]:
  assert m.score({'answer':answer},{'required_groups':[['cannot','unavailable']]})['passed']
