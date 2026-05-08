# FieldAid Architecture

```mermaid
flowchart LR
  A["Responder browser"] --> B["FastAPI local service"]
  B --> C["Ollama / Gemma 4"]
  B --> D["Bundled emergency RAG"]
  B --> E["SQLite incident store"]
  B --> F["Local upload storage"]
  B --> G["MEDIC image classifier"]
  B --> H["YOLO optional detector"]
  C --> I["Structured incident output"]
  D --> I
  G --> I
  H --> I
  I --> J["Action board"]
  I --> K["SMS / Discord draft"]
  I --> L["Handoff packet"]
  I --> M["Sync-later export"]
```

## Local Components
- Browser UI: Mission Mode, Intake, Dashboard, Video, Trust, Offline Proof, Incidents, Exports.
- FastAPI backend: upload handling, analysis endpoints, sync/outbox APIs.
- Gemma 4 through Ollama: local reasoning and multimodal frame/photo interpretation.
- SQLite: incidents, status, sync queue, outbound handoff drafts.
- RAG: local emergency guidance snippets for grounded citations.
- MEDIC classifier: MobileNetV3 checkpoint for disaster image/frame evidence.
- YOLO: optional supporting object detector for people/vehicle/access observations.

## Safety Contract
- Never declare a route, bridge, or structure safe from imagery.
- Medical, fire, floodwater, structural, and life-safety outputs require human verification.
- Discord handoff is drafted and recorded, not auto-sent.
- No cloud endpoint is required after setup.
