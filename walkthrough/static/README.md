# `static/`

Browser UI for the local FieldAid app.

- `index.html`: page structure for Mission Mode, Intake, Dashboard, Video, Trust, Offline Proof, Incidents, and Exports.
- `app.js`: frontend routing, form submission, result rendering, Discord handoff prompts, sync queue UI, and video/classifier display.
- `styles.css`: responsive visual design and layout.

This is a static single-page interface served by FastAPI. It intentionally works against `127.0.0.1` so the demo can run offline after setup.
