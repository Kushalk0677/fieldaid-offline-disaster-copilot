# `app/`

FastAPI backend for FieldAid.

- `main.py`: API routes, static serving, upload handling, and app startup.
- `models.py`: Pydantic schemas shared by API responses and tests.
- `ollama_client.py`: local Gemma 4 calls through Ollama.
- `rag.py`: local emergency-guidance retrieval.
- `tools.py`: structured incident, supply, SMS, safety, and fallback logic.
- `grounding.py`: Trust/photo grounding with citations and Discord handoff drafts.
- `image_classifier.py`: optional local MEDIC disaster-image classifier loader.
- `video_scan.py`: video upload, frame sampling, Gemma frame analysis, classifier aggregation, and YOLO support.
- `db.py`: SQLite incident log, sync queue, and outbound handoff outbox.

The backend is designed to keep running even when optional models are missing. Missing Gemma, YOLO, or classifier dependencies return safe fallback outputs instead of crashing the demo.
