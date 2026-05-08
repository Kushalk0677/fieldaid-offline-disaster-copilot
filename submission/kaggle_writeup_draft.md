# FieldAid: Offline Disaster Response Copilot

## Subtitle
Local Gemma 4 intelligence for the first hours of disaster response, when internet access and coordination are fragile.

## Draft Writeup
FieldAid is an offline-first disaster response copilot for community responders managing shelters and damaged access routes. During floods, storms, earthquakes, and blackouts, early reports often arrive as photos, handwritten notes, and short radio messages. FieldAid turns that messy local information into structured action: a prioritized incident report, SMS-ready update, supply request, and grounded guidance citations.

The prototype runs as a local web app backed by FastAPI, SQLite, and Ollama. Gemma 4 is used through Ollama, with `gemma4:e4b` as the default local model and `gemma4:26b` available for higher-quality workstation demos. The backend sends a multimodal prompt containing the responder note, optional image, scenario type, location, and retrieved emergency guidance. The model returns structured JSON that FieldAid converts into responder-facing tools: incident report creation, supply request drafting, radio/SMS update generation, and status updates.

FieldAid uses bundled emergency guidance documents as an offline retrieval source. Before each Gemma 4 call, the app retrieves relevant snippets for shelter water, vulnerable residents, route damage, supply prioritization, and offline communications. The final Trust panel separates extracted facts from inferred recommendations, displays source snippets, shows confidence, and flags medical, structural, or life-safety outputs for human verification.

The demo scenario centers on a school shelter with 43 people, elderly residents, insulin patients, limited water, and a blocked bridge route. A second damage assessment flow handles a road or bridge image. The combined output shows critical medical continuity, water shortage, blocked access, and a short dispatch update. This story demonstrates why local AI matters: when the network is gone, responders can still reason over local data, keep a timestamped incident log, and synchronize later.

FieldAid is not a replacement for emergency services, clinicians, engineers, or incident command. It is a decision-support tool designed to make field coordination faster, more transparent, and more grounded in the places that need help first.

