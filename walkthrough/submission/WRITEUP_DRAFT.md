# FieldAid: Offline Disaster Response Copilot

## Subtitle
Local Gemma 4, emergency retrieval, and disaster-image adaptation for response when the network goes down.

## Track
Global Resilience. Secondary relevance: Safety & Trust and Ollama special technology track.

## Summary
FieldAid is an offline-first disaster response copilot for shelter triage, route damage assessment, and responder handoff. During floods, fires, earthquakes, or storms, field teams often have fragmented notes, photos, videos, and no reliable internet. FieldAid turns that chaos into a prioritized incident packet: action plan, supply request, SMS/radio update, Discord handoff draft, verification flags, and cited emergency guidance.

## How FieldAid Uses Gemma 4
FieldAid runs Gemma 4 locally through Ollama as the reasoning engine. Gemma receives field notes, optional images, sampled video-frame summaries, and retrieved guidance snippets. It produces structured incident outputs, not just chat text. The app exposes the tool trace in the UI: `retrieve_guidance`, `create_incident_report`, `create_supply_request`, `generate_sms_update`, `create_discord_handoff`, and `export_offline_sync_packet`.

## Architecture
The app is a local FastAPI service with a static browser UI. SQLite stores incidents, outbound handoff drafts, and sync queue records. Uploaded images and video frames stay in local storage. Retrieval runs over bundled emergency guidance files. The frontend has Mission Mode, Intake, Dashboard, Video, Trust, Offline Proof, Incidents, and Exports pages.

## Domain Adaptation
To improve disaster imagery recognition on constrained hardware, we trained a lightweight MobileNetV3 classifier on a QCRI/MEDIC subset with four labels: building damage, fire, flood, and road damage. The best local checkpoint reached about 63.6% validation accuracy on the balanced validation split. FieldAid uses this model as a local visual evidence stream for Trust photo uploads and sampled video frames. For videos, it aggregates repeated frame predictions, such as “fire in 7/9 sampled frames,” before creating an incident.

## Safety And Trust
FieldAid never declares a bridge, road, route, or structure safe from imagery. Any medical, structural, fire, floodwater, or life-safety output includes human verification language. The Trust page separates facts, inferred recommendations, citations, model/classifier evidence, and verification flags. Discord handoff is manual by design: FieldAid drafts and tracks the message locally, but the user must review and send.

## Offline Design
After dependencies and models are installed, the demo works without cloud inference. Gemma runs through Ollama, the classifier loads from a local checkpoint, guidance is bundled, and incidents sync later through JSON export. This matters for shelters, rural clinics, field operations centers, and community responders where connectivity fails exactly when decisions become urgent.

## Limitations
FieldAid is a proof of concept, not an emergency authority. The classifier is intentionally lightweight and not a replacement for trained responders. The system improves situational awareness and handoff quality, but final action must remain with incident command, medical staff, structural specialists, or local authorities.
