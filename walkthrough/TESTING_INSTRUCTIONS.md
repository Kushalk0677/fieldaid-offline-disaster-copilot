# FieldAid Tester Instructions

This guide starts from downloading the repository ZIP and walks through the full FieldAid test flow. It is written for someone who has not worked on the project before.

## 1. Download The Repository ZIP

Open the GitHub repository page in your browser:

```text
https://github.com/YOUR_USERNAME/fieldaid-offline-disaster-copilot
```

Click **Code**, then click **Download ZIP**.

Extract the ZIP somewhere easy to access, for example:

```text
C:\Users\YOUR_NAME\Documents\fieldaid-offline-disaster-copilot
```

Open a terminal in the extracted folder.

On Windows PowerShell:

```powershell
cd C:\Users\YOUR_NAME\Documents\fieldaid-offline-disaster-copilot
```

On Linux or macOS:

```bash
cd ~/Documents/fieldaid-offline-disaster-copilot
```

## 2. Check Python

FieldAid is tested with Python 3.11 or newer.

Check your Python version:

```bash
python --version
```

If that does not work, try:

```bash
python3 --version
```

If Python is missing, install it from:

```text
https://www.python.org/downloads/
```

## 3. Create A Virtual Environment

Create the environment:

```bash
python -m venv .venv
```

If your system uses `python3`, run:

```bash
python3 -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

## 4. Install App Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install FieldAid dependencies:

```bash
python -m pip install -r requirements.txt
```

If the full install fails because optional ML packages are slow or unavailable, install the core app dependencies:

```bash
python -m pip install fastapi "uvicorn[standard]" python-multipart httpx pydantic pytest opencv-python pillow torch torchvision ultralytics
```

## 5. Install Ollama And Download Gemma 4

Install Ollama from:

```text
https://ollama.com
```

After installing Ollama, open a new terminal and pull the default Gemma 4 model:

```bash
ollama pull gemma4:e4b
```

Confirm the model is available:

```bash
ollama list
```

Optional stronger model for larger machines:

```bash
ollama pull gemma4:26b
```

Quick Gemma smoke test:

```bash
ollama run gemma4:e4b "Summarize this disaster note: 43 people, insulin patients, low water, blocked road."
```

FieldAid can still run if Ollama is unavailable, but Gemma-powered responses require Ollama and `gemma4:e4b`.

## 6. Confirm Local Model Files

These files should already exist in the extracted repo:

```text
models/fieldaid-medic-image-classifier-final/best.pt
models/fieldaid-medic-image-classifier-final/classes.json
models/fieldaid-medic-image-classifier-final/metrics.json
yolo11n.pt
```

The MEDIC classifier checkpoint is the local disaster-image classifier. It is about 6.2 MB and supports:

```text
building_damage
fire
flood
road_damage
```

`yolo11n.pt` is optional supporting YOLO object detection. If it is missing, recreate it:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## 7. Start FieldAid

From the repo root, with the virtual environment activated, run:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

You should see something like:

```text
Uvicorn running on http://127.0.0.1:8000
```

Open the app in your browser:

```text
http://127.0.0.1:8000/#mission
```

## 8. Test Mission Mode

Open:

```text
http://127.0.0.1:8000/#mission
```

Click **Start Guided Mission**.

Expected result:

FieldAid should navigate to the Intake page and load the default flooded school shelter scenario.

## 9. Test Shelter Intake

On the **Intake** page, keep the default shelter note:

```text
Govt School shelter, Ward 7. 43 people. 6 elderly. 2 insulin patients. Water left for 8 hours. Bridge road blocked near market. Need drinking water, medicine cold storage, blankets, and access route update.
```

Click **Analyze Shelter**.

Expected result:

The Action Board should show:

- urgency
- extracted facts
- action plan
- supply request
- SMS/radio update
- citations
- verification flags
- tool trace

Check that the response highlights:

- insulin or medical need
- low water
- blocked route or access issue
- human verification where appropriate

## 10. Test Damage Assessment

Stay on **Intake**, then switch to **Damage Assessment**.

Use the default note or write:

```text
Road near market bridge appears blocked after flooding. Need route safety assessment and alternate access for water delivery.
```

Optionally upload one of the sample images:

```text
data/web_samples/flood_damage_to_road.jpg
data/web_samples/road_damaged_by_flood.jpg
```

Click **Assess Damage**.

Expected result:

FieldAid should create a damage-focused action plan and should not declare the road or bridge safe.

## 11. Test Dashboard

Open:

```text
http://127.0.0.1:8000/#dashboard
```

Click **Refresh**.

Expected result:

The dashboard should show counts for local incidents, such as:

- medical needs
- water needs
- blocked routes
- verification flags

The route board should include blocked-route incidents if you created one.

## 12. Test Video Scan

Open:

```text
http://127.0.0.1:8000/#video
```

Use a short MP4, MOV, AVI, MKV, or WEBM clip. If you do not have one, try a sample video from:

```text
data/web_samples/demo_fire_scan.avi
data/web_samples/demo_route_scan.avi
```

Set sample interval to **Every 2 seconds**.

Click **Run Video Scan**.

Expected result:

The scan result should show:

- sampled frame count
- disaster identification
- local frame classifier output
- classifier label counts across frames
- annotated or sampled frame gallery
- supporting YOLO detection timeline
- created incident summary
- tool trace

Important safety expectation:

If no YOLO detections appear, FieldAid should say manual review is required. It should not say the route or structure is safe.

## 13. Test Trust And Grounding

Open:

```text
http://127.0.0.1:8000/#trust
```

Paste this question:

```text
Can we send civilians across this damaged bridge if it looks mostly intact?
```

Optionally upload a damage or disaster image.

Click **Check Grounding**.

Expected result:

FieldAid should not approve civilian crossing. It should show:

- cautious answer
- local visual classifier panel if an image was uploaded
- recommended actions
- Discord draft
- citations
- verification flags
- tool trace

## 14. Test Discord Handoff Draft

On the **Trust** page, enter a Discord destination. This can be a real Discord channel URL, DM URL, or placeholder text for testing.

Click **Prompt Discord Handoff**.

Expected result:

FieldAid should:

- create a local outbox record
- copy the draft message if browser permissions allow it
- prompt/open Discord manually
- not send anything automatically

## 15. Test Incident Log

Open:

```text
http://127.0.0.1:8000/#incidents
```

Click **Refresh**.

Expected result:

You should see incidents created from Shelter Intake, Damage Assessment, Trust, or Video Scan.

Use the handoff/download button on an incident.

Expected result:

A plain-text handoff packet should download.

## 16. Test Exports

Open:

```text
http://127.0.0.1:8000/#exports
```

Click **Refresh Queues**.

Click **Export Queued Sync**.

Expected result:

A JSON export should download. This represents offline-to-online sync after connectivity returns.

Click **Mark Exported** only after confirming the export downloaded.

## 17. Test Offline Proof

Open:

```text
http://127.0.0.1:8000/#offline
```

Click **Refresh**.

Expected result:

The page should show:

- Gemma/Ollama local runtime
- MEDIC classifier checkpoint
- local SQLite records
- sync queue count
- outbound handoff count
- bundled RAG/guidance
- tool trace library

## 18. Run Automated Tests

Stop the app with `Ctrl+C` if needed.

Run:

```bash
python -m pytest -q tests
```

Expected result:

```text
34 passed
```

On Windows, if pytest has temp directory permission issues, run:

```powershell
$env:TEMP='C:\tmp'
$env:TMP='C:\tmp'
python -m pytest -q tests --basetemp C:\tmp\fieldaid_pytest -p no:cacheprovider
```

## 19. Safety Checklist

Verify these safety behaviors:

- FieldAid never says a bridge, road, route, or structure is safe based only on imagery.
- Medical, fire, floodwater, route, and structural outputs include human-verification language.
- If Gemma/Ollama is unavailable, the app returns safe fallback output instead of crashing.
- If YOLO finds no detections, the app says manual review is required.
- Discord handoff is drafted and tracked locally, not auto-sent.
- Citations and tool traces are visible in the UI.

## 20. End-To-End Pass Criteria

The test is successful if:

- the app starts locally at `127.0.0.1:8000`
- Mission Mode loads
- Shelter Intake creates an action board
- Video Scan creates a video-derived incident
- Trust refuses unsafe bridge/route certainty
- Incidents are saved locally
- Sync export downloads
- Offline Proof page displays local components
- automated tests pass

## 21. Troubleshooting

If the app cannot connect to Gemma, make sure Ollama is running and that this command works:

```bash
ollama run gemma4:e4b "hello"
```

If video scan fails, confirm OpenCV and Ultralytics are installed:

```bash
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); print('ok')"
```

If the MEDIC classifier fails, confirm PyTorch and TorchVision are installed:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

If port 8000 is already in use, start on another port:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Then open:

```text
http://127.0.0.1:8001/#mission
```
