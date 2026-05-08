# FieldAid: Offline Disaster Response Copilot

## Subtitle
Local Gemma 4 intelligence for the first hours of disaster response, when internet access and coordination are fragile.

## Track
Global Resilience (primary). Secondary: Safety and Trust, Ollama special technology track.

---

We built FieldAid after seeing a pattern across disaster reports: the hardest decisions always happen in the first few hours, exactly when connectivity goes down. Shelter coordinators with a notebook and a radio. EMTs with photos on a phone that cannot load Copilot or Gemini. Teams triaging by instinct because the tools that could help them require a network that no longer exists. The disconnect was clear. AI models capable of reasoning over field reports and analyzing images were sitting idle in disaster zones, not because they were too weak, but because they were all cloud-only. Gemma 4 changed that constraint. At 9.6 GB, the E4B variant runs locally on a laptop through Ollama. That made it possible to put frontier-level reasoning into the same machine responders already carry.

The school building has 43 people inside. That includes six residents over 75 and two who need insulin kept below 8 degrees. The water supply will last eight hours at current draw. The bridge to the main road collapsed an hour ago. Radios crackle with partial information from three other shelters in the district. No one has a clear picture of what is happening.

This is the gap FieldAid closes.

FieldAid is a local-first disaster response copilot that runs on a single laptop with no internet. It takes responder notes, photos, and short field video clips and returns a prioritized incident report: action plan, supply request, SMS update, Discord handoff draft, cited emergency guidance, and verification flags. The reasoning engine is Gemma 4 (E4B), running locally through Ollama. Everything else is local too: a 6 MB MobileNetV3 image classifier, a SQLite incident database, bundled guidance documents, and a Python-based retrieval layer.

---

### Architecture

FieldAid is built around four components that run entirely on one machine.

The browser UI is a static single-page application served from the same Python process that handles API requests. It has eight pages: Mission Mode for guided intake, Dashboard for real-time incident status, Video for frame-level disaster analysis, Trust for photo-grounded safety checks, Incidents for structured logging, Exports for offline sync, and Offline Proof for system verification.

The backend is a FastAPI service with 13 endpoints. The core endpoints are:

- `POST /api/analyze` accepts a scenario type (shelter, damage, combined), a text note, optional images, and returns a structured `AnalysisResult`.
- `POST /api/video/scan` accepts a video upload, samples frames with OpenCV, runs Gemma 4 on up to 6 frames and a MobileNetV3 classifier on all 20, runs YOLOv11n for supporting detections, and aggregates everything into a `VideoScanResult`.
- `POST /api/grounding` takes a safety question and optional photo, runs the classifier first, then calls Gemma 4 for a photo-grounded analysis, and returns a structured answer with citations and verification flags.
- `GET /api/sync/export` downloads a JSON containing all incident records and sync queue entries for later synchronization.

The database is SQLite in WAL mode with three tables holding incidents, outbound message drafts, and the sync queue. Incidents auto-generate sync queue entries on creation and move to exported status after a sync export.

Gemini 4 runs locally through Ollama at `http://127.0.0.1:11434/api/chat` with temperature 0.1 for deterministic output. The default model is `gemma4:e4b` (9.6 GB). Workstation demos can swap to `gemma4:26b` for stronger reasoning. The system prompt is fixed: it tells the model to return only valid JSON, never claim to replace emergency services, and always flag medical, structural, and life-safety recommendations for human verification.

---

### How Gemma 4 Is Used

FieldAid uses Gemma 4 in three distinct modes, each with its own prompt structure and output shape.

**Shelter and damage intake.** The model receives a multimodal prompt with scenario type, location, responder note, optional before and after images, and up to three retrieved guidance snippets. It returns a `summary`, `urgency` level, `extracted_facts`, `inferred_recommendations`, an `action_plan` with priority, title, rationale, owner, and next step for each item, a `supply_request` list, an `sms_update` string capped at 320 characters, a confidence score, and `verification_flags`. The confidence defaults to 0.72 for model-sourced output.

The prompt includes a JSON schema, which constrains the model to produce structured output instead of free text. The parser attempts direct JSON deserialization first, then falls back to regex extraction of the outermost `{...}` block. This handles the common case where the model wraps JSON in markdown fences or adds a brief preamble.

**Video frame analysis.** FieldAid samples up to 20 frames from a video at configurable intervals (default 2 seconds, minimum 1 second). It sends the first 6 frames to Gemma 4 for disaster type classification. The model must choose from fire, flood, road damage, building damage, crowd risk, smoke, or unknown. The prompt explicitly forbids declaring any route, bridge, building, or fire scene safe. If uncertain, the model must return unknown and require human review.

**Photo-grounded trust analysis.** When a responder uploads a photo with a safety question, FieldAid runs the MobileNetV3 classifier first, injects the top predictions into the prompt as supporting evidence (for example, "flood 70 percent, building damage 22 percent"), and sends the image plus question to Gemma 4. The model returns a status label, scene type, answer, visual evidence, recommended actions, a Discord-ready message draft, confidence, and verification flags.

When Gemma 4 is unreachable, all three paths fall back to deterministic logic. The fallback parser scans responder text for population counts, water hours, insulin mentions, blocked routes, and fire keywords. It assigns urgency based on keyword matching, builds an action plan from a conditional rule table, generates supply items, and composes a 320-character SMS update. The fallback confidence is 0.62 with text and 0.48 without text.

---

### Safety Contract

FieldAid enforces a safety contract at three levels.

At the prompt level, the system prompt requires the model to flag medical, structural, and life-safety recommendations for human verification. Video and trust prompts explicitly forbid safe declarations.

At the application level, the `safety_flags` function scans every output against a `HIGH_RISK_TERMS` dictionary with 19 keywords across three categories: medical (insulin, diabetic, injury, medicine, patient, elderly), structural (bridge, collapse, crack, road blocked, building, power line), and life safety (trapped, evacuate, flood, fire, landslide). When any keyword matches, a corresponding flag is appended: "Medical guidance requires human verification" or "Structural damage assessment requires inspection by authorized personnel." Damage scenarios always receive a structural inspection flag regardless of content.

At the grounding level, the trust page separates facts from inferences, displays retrieval citations with source file and snippet, shows the model or classifier confidence, and requires human verification before any action is dispatched. Discord handoff messages are drafted locally and recorded in an outbox with status drafted, not sent automatically.

The fallback photo analysis adds two more layers. If the classifier produces uncertain results, FieldAid runs OpenCV heuristics directly on the image: HSV color space orange detection for fire (threshold 2.5 percent of pixels) and low-saturation mid-gray detection for smoke (threshold 18 percent of pixels). Neither heuristic ever declares a scene safe.

---

### Domain Adaptation

FieldAid trains a lightweight MobileNetV3 Small classifier on a subset of the QCRI/MEDIC disaster image dataset. The full dataset contains 71,196 images. We used a 25 percent labeled subset of 17,799 images with four classes: building damage (11,392 images, 64 percent), flood (5,022, 28 percent), road damage (839, 4.7 percent), and fire (546, 3.1 percent).

The class imbalance was severe: building damage images outnumbered fire images by 21 to 1. We built a balanced training set of 6,279 samples and a validation set of 1,106 samples using 25 percent sampling per class with a fixed seed for determinism. Training ran for 8 epochs with standard ImageNet normalization on 224 by 224 pixel crops.

The model peaked at 63.56 percent validation accuracy at epoch 2. By epoch 8, training accuracy had reached 93.84 percent while validation sat at 62.84 percent, confirming overfitting. The saved checkpoint is from epoch 2. The model weighs 6.2 MB and runs on CPU in under 100 milliseconds per image.

Confidence bands are set at 70 percent for likely, 45 percent for possible, and below 45 percent for uncertain. These thresholds are generous on purpose: a 55 percent confidence prediction that an image shows flood damage should still trigger the right response workflow.

The classifier serves as a visual evidence stream alongside Gemma 4: it runs first on every trust photo upload and every sampled video frame, its output is injected into the Gemma 4 prompt, and for video it aggregates results across frames with the formula `min(0.95, avg_confidence * 0.7 + frame_ratio * 0.3)`. If Gemma 4 returns unknown or lower confidence than the classifier aggregation, the classifier wins.

We also maintain LoRA fine-tuning scripts for Gemma 4 text and vision adapters using Unsloth. The text adapter configures r=16, alpha=32, dropout=0, learning rate 2e-4, 3 epochs, effective batch size 8, and targets all attention and MLP projection layers. The vision adapter uses r=4, alpha=8, 2 epochs, and selective freezing options for 12 GB GPUs. Both are ready to run on a single A100 or T4 instance and export to GGUF via the Ollama Modelfile for local deployment.

---

### Offline Design

The offline guarantee is the entire point. After the initial install of dependencies, Ollama, and the Gemma 4 model, the application requires zero internet connectivity.

Retrieval runs over four bundled Markdown files with 2,600 tokens total: shelter water and vulnerable residents, damage assessment and route safety, supply request priorities, and offline communications and situation reports. The retrieval mechanism is BM25-like term overlap scoring with a two-point bonus for scenario matching. It selects the top three documents and extracts the paragraph with the highest token overlap as a citation snippet capped at 260 characters.

Incidents are stored in SQLite and exported as JSON for manual sync. The sync queue tracks entity type, entity ID, queued status, and last exported timestamp. When connectivity returns, responders download the export file and transfer it to incident command.

The SMS layer localizes updates to Hindi, Tamil, and Spanish in addition to English, based on an audience parameter in the intake form. This covers the four most common languages in flood-affected regions of South Asia.

---

### Evaluation

FieldAid runs 34 tests across five files: 12 API endpoint tests, 8 core logic tests, 1 image classifier test, 6 training asset tests, and 7 video scan tests. All pass in CI on Python 3.11.

The image classifier achieved 63.56 percent validation accuracy on 1,106 balanced samples. Combined with the OpenCV heuristic fallback (fire detection confidence capped at 0.82, smoke at 0.72), the visual pipeline covers the four disaster types and two atmospheric hazards.

The deterministic fallback analysis correctly identifies population counts with the regex `\d+\s*(people|persons|residents|evacuees)`, water hours with `water.*\d+\s*(hour|hr|hrs|hours)`, and insurance mentions. It assigns urgency through a three-tier escalation: medium by default, high on any of 12 common keywords, critical on unconscious, life threat, severe bleeding, or fire spreading.

Every path in the system produces verification flags, tool traces, and structured JSON output. Nothing silently returns a safe declaration.

---

### Limitations and What Comes Next

The classifier accuracy of 63.56 percent reflects the challenge of classifying real disaster imagery under heavy class imbalance. More balanced training data and a larger model (EfficientNet-B0 or the 59-class variant currently in training) will close that gap. The current checkpoint is intentionally small enough to run on a CPU.

Gemma 4 E4B is the right size for edge deployment but occasionally produces verbose summaries. The 26B variant improves reasoning quality and is available on workstations with more memory.

The LoRA adapters have full training scripts but have not been exported to GGUF format yet. Publishing a fine-tuned `fieldaid-gemma4:e4b` adapter would demonstrate measurable improvement over the base model on disaster scenarios.

FieldAid currently runs as a single-process web server. A Docker container for the web app is available, but Ollama runs as a separate process. A combined Docker Compose setup would simplify deployment.

This is a decision-support tool. Final operational decisions remain with incident command, medical staff, structural engineers, and local authorities. FieldAid makes the information reaching those decisions more organized, more cited, and more transparent.

