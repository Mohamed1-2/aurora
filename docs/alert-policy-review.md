# Alert policy review - 13 September 2026

Fresh proactive events use the actual `type`, `severity`, `event_id` and `dismissed` schema. Warning replies are answer-first; critical replies are warning-first. A critical battery_overtemperature event replaces routine replies with a stop-when-safe message. Refusals are preserved. Stale or dismissed events cannot trigger an override.

New critical events interrupt pending requests and speech when voice is enabled. Delivery is deduplicated by event ID and severity; escalation or recurrence can announce again. Acknowledge uses a stable session ID and reports success only after a successful response. Acknowledge is available in the driver view. Muted speech remains muted. Browser audio permission can still prevent playback; the warning remains visible.

Removed unused threshold constants, action tables, classifier and answer recomputation, and replaced the contradictory policy comment. The Energy/direct display path also respects critical thermal overrides.

## Verification

`node tests/test_driving_ui.cjs`: 15 functional assertions for ordering, thermal override, stale and dismissed events, refusal preservation, delivery suppression, recurrence and escalation.

`python scripts/evaluate_experience.py`: live API cases for paraphrases, expected signal references, ambiguity, contextual destination follow-ups and round-trip latency. Rule checks do not replace human semantic evaluation. The default cases primarily exercise deterministic routes, not general model reasoning. Results are stored in docs/experience-evaluation.json.

No new speech model, cabin-noise validation, production-vehicle validation or GitHub publication is claimed. Local Git initialization was not performed because its approval was declined.
