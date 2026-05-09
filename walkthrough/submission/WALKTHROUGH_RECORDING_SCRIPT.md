# FieldAid Walkthrough Recording Script

Use this as the spoken script for the judged demo video. The flow assumes the Colab runtime is already prepared for recording, with dependencies installed, `gemma4:e4b` pulled, Ollama running, and the webapp open.

Target length: 4 to 6 minutes for a complete walkthrough, or 3 minutes if you trim the step commentary.

## Recording Setup

Before recording:

1. Open the GitHub repository on the `wt/walkthrough` branch.
2. Open the Colab notebook from the walkthrough branch.
3. Make sure the Colab runtime has already completed setup.
4. Make sure the FieldAid webapp is open in the Colab-served browser window.
5. Start recording only after the app is loaded and ready.

## Opening On GitHub

Screen: GitHub repository, branch selector showing `wt/walkthrough`.

Narration:

```text
This is FieldAid, an offline disaster response copilot built for the Gemma 4 Good hackathon. I am showing the walkthrough branch because this branch contains the automated judge demo, the Colab launcher, and the bundled media used by the walkthrough.

The core idea is simple: when floods, fires, earthquakes, or storms knock out internet access, responders still need intelligence. FieldAid runs locally with Gemma 4 through Ollama, stores incidents in SQLite, uses bundled emergency guidance for citations, and creates handoff packets that can be synced later.
```

Show briefly:

```text
walkthrough/AutoDemo.ipynb
walkthrough/static/
walkthrough/app/
walkthrough/media/
```

Narration:

```text
The walkthrough folder contains a self-contained guided demo. It includes the Colab notebook, the webapp changes, the walkthrough controller, and local media files for flood, fire, hurricane, and earthquake scenarios.
```

## Opening Colab

Screen: Colab notebook.

Narration:

```text
For video purposes, I already ran the Colab setup. The notebook clones this branch, installs dependencies, starts Ollama, checks that the Gemma 4 E4B model is available, copies the walkthrough files into the app, and starts the FastAPI server.

The first model pull can take a while because Gemma 4 E4B is about 9.6 gigabytes. That is expected. After the model is present in the runtime, the notebook skips the pull and starts much faster.
```

Point to cells:

```text
Clone repo
Install dependencies
Install or start Ollama
Verify gemma4:e4b
Apply walkthrough overlay
Launch server
```

Narration:

```text
The important point is that after setup, the app runs locally. The model runs through Ollama, the database is local, the media stays local, and the emergency guidance is bundled with the project.
```

## Opening The Webapp

Screen: FieldAid website before pressing Start Walkthrough.

Narration:

```text
This is the FieldAid webapp. It is designed for the first chaotic hours of a disaster response, when teams may only have partial field notes, phone photos, short video clips, and radio updates.

The app turns those fragments into incident packets: a summary, action plan, supply request, SMS or radio update, citations, verification flags, and exportable handoff records.

Gemma 4 is the reasoning engine. The rest of the system supports it with local retrieval, local storage, a lightweight visual classifier, and optional YOLO detections for video frames.
```

Point to top navigation:

```text
Mission
Intake
Dashboard
Incidents
Video
Trust
Offline Proof
Exports
```

Narration:

```text
Instead of manually clicking through every page, I built an automated walkthrough. It fills the forms, attaches the local images or videos, calls the same backend APIs a responder would use, and shows the results in the normal UI.
```

Action: Click `Start Walkthrough`.

Narration:

```text
I will start the guided walkthrough now.
```

## Step 1: Shelter Intake

Screen: Intake page, Shelter tab.

Action: Click Next.

Narration while it runs:

```text
The first step is shelter intake. FieldAid fills a field report for a school shelter with 43 people, elderly residents, insulin patients, limited water, and a blocked bridge.

Gemma 4 receives the responder note, the location, the role, the requested SMS language, and local guidance snippets. It returns a structured incident packet instead of free text.
```

After result appears:

```text
Here we get the summary, urgency, extracted facts, prioritized action plan, supply request, SMS or radio update, citations, and verification flags. This is meant to help a district operations team move from a messy field note to an auditable response packet.
```

What this shows:

```text
Gemma 4 structured JSON output
Local RAG citations
Supply and action planning
Safety flags for medical and route risks
Incident creation in local storage
```

## Step 2: Flood Damage Assessment

Screen: Intake page, Damage tab.

Action: Click Next.

Narration while it runs:

```text
The second step switches to damage assessment. The walkthrough fills a road and bridge damage report and attaches a local flood image.

This demonstrates baseline multimodal response planning: Gemma 4 reasons from the field note and image while the app keeps route safety claims cautious.
```

After result appears:

```text
Notice that FieldAid does not declare the route safe. It turns the evidence into operational priorities: closure, alternate route planning, inspection, and communication.
```

What this shows:

```text
Image plus text intake
Route damage reasoning
Structural verification flags
Local media upload handling
```

## Step 3: Online Context Enrichment

Screen: Intake page, Damage tab.

Action: Click Next.

Narration while it runs:

```text
This step adds the optional online enrichment layer. If connectivity is available, responders can add weather, satellite, map, or Street View context. The connectivity badge reinforces the operating mode: offline operational, low-connectivity aware, with deferred sync.

The key design choice is that online context is supporting evidence only. FieldAid still works without it, and Gemma 4 is explicitly told not to declare roads, bridges, buildings, or hazard zones safe from online context or imagery alone.
```

What this shows:

```text
Optional weather context
Optional map or Street View context
Optional satellite or before-assessment context
Offline-first fallback when online sources are unavailable
```

## Step 4: Fire Video Scan

Screen: Video page.

Action: Click Next.

Narration while it runs:

```text
Now the walkthrough uploads a wildfire video. FieldAid samples frames locally, sends representative frames to Gemma 4, runs local classifier support where available, and creates an incident from the video evidence.
```

After result appears:

```text
The output includes the likely disaster type, model summary, sampled frames, local classifier aggregation, supporting detections, and a created incident record.
```

What this shows:

```text
Video frame sampling
Gemma 4 visual reasoning
Local classifier support
Automatic incident creation from video
```

## Step 5: Hurricane Video Scan

Screen: Video page.

Action: Click Next.

Narration:

```text
The next video tests a different disaster pattern. This is a hurricane impact clip. The same local pipeline is reused: sample frames, analyze the event, aggregate evidence, and create a response record.

The goal is not only classification. The goal is converting video into operational awareness when cloud tools are unavailable.
```

What this shows:

```text
Reusable disaster video pipeline
Different event type
Evidence aggregation across frames
```

## Step 6: Trust Gate, Fire Safety Question

Screen: Trust page.

Action: Click Next.

Narration:

```text
This step is a safety and trust gate. The question is whether an area near the fire is safe for civilian access.

FieldAid should not make unsupported safety declarations. It should identify visible evidence, cite guidance, recommend caution, and route the decision back to human responders.
```

After result appears:

```text
The important behavior is restraint. The system provides evidence and recommended actions, but it does not replace incident command, firefighters, medical staff, or structural inspectors.
```

What this shows:

```text
Safety-centered prompting
Grounded visual answer
Verification flags
Discord-ready handoff draft
```

## Step 7: Trust Gate, Flooded Highway Question

Screen: Trust page.

Action: Click Next.

Narration:

```text
The second trust gate asks whether civilians can safely cross a flooded highway.

This is exactly the kind of question where an AI system should be careful. FieldAid is designed to refuse unsupported safety claims and instead recommend inspection, closure, alternate routes, and human verification.
```

What this shows:

```text
No unsafe route claims
Flood-specific caution
Human verification workflow
```

## Step 8: Earthquake Video Scan

Screen: Video page.

Action: Click Next.

Narration:

```text
This step uploads an earthquake impact video. This demonstrates that the same workflow can handle multiple disaster categories without changing tools.

Again, the important output is not just a label. It is the incident packet, visual evidence, priority summary, and downstream handoff record.
```

What this shows:

```text
Multi-disaster coverage
Video-to-incident workflow
Local operation with bundled media
```

## Step 9: Operations Dashboard

Screen: Dashboard page.

Action: Click Next.

Narration:

```text
Now the walkthrough opens the operations dashboard. The dashboard aggregates the incidents created during the previous intake and video steps.

This is where a response lead can see urgent medical needs, water shortages, blocked routes, and verification flags across the local incident log.
```

What this shows:

```text
Local SQLite incident aggregation
Operational dashboard view
Priority indicators
```

## Step 10: Incident Log

Screen: Incidents page.

Action: Click Next.

Narration:

```text
The incident log stores every generated record locally. Each incident can be reviewed, status-updated, and exported as a handoff packet.

This matters because disaster response needs auditability. FieldAid is not just answering questions. It is creating records responders can review and transfer.
```

What this shows:

```text
Local persistence
Reviewable incident records
Status workflow
Handoff packet generation
```

## Step 11: Discord Handoff Draft

Screen: Exports page.

Action: Click Next.

Narration:

```text
This step demonstrates handoff and sync preparation. FieldAid can create message drafts and exportable records, but it does not automatically send them.

That is intentional. The system supports responders while keeping final dispatch decisions under human control.
```

What this shows:

```text
Human-in-the-loop handoff
Local outbox model
No automatic external dispatch
```

## Step 12: Offline Sync Export

Screen: Exports page.

Action: Click Next.

Narration:

```text
The final step shows offline sync export. When the network is unavailable, incidents stay local. When connectivity returns, responders can export the queue and move it to another system manually.

This closes the offline-first loop: local reasoning, local records, local handoffs, and sync later.
```

What this shows:

```text
Offline-first design
Sync-later JSON export
Audit-friendly records
```

## Closing Statement

Screen: Any strong result page, preferably Dashboard, Trust, or Exports.

Narration:

```text
FieldAid is not a replacement for responders. It is a local disaster response copilot for the first hours when information is fragmented and the network is down.

Gemma 4 provides the reasoning layer. The application adds local retrieval, structured incident records, visual evidence, safety flags, and handoff workflows.

When the network goes down, intelligence should still show up.
```

## Short Version For A 3-Minute Video

If time is tight, use this compressed narration:

```text
FieldAid is an offline disaster response copilot built with Gemma 4. I am launching the guided demo from the walkthrough branch and Colab. The runtime is already prepared for video purposes because the first Gemma model pull is large.

The app runs locally through FastAPI and Ollama. Gemma 4 is the reasoning engine, while local guidance, SQLite, media files, a lightweight classifier, and optional YOLO support provide the surrounding workflow.

I will press Start Walkthrough. The demo now fills each form, attaches local media, calls the actual app APIs, and renders the normal UI results.

First, shelter intake turns a field note about 43 people, insulin patients, low water, and blocked access into an action plan, supply request, SMS update, citations, and verification flags.

Next, damage assessment combines a flood image with a route report plus optional weather, satellite, and map context. FieldAid avoids unsafe route claims and recommends closure, inspection, and alternate access.

Then video scanning samples fire, hurricane, and earthquake clips. Gemma 4 analyzes the visual evidence, the local classifier adds supporting signals, and the app creates incident records.

The trust gates ask risky safety questions about fire access and flooded highways. FieldAid refuses unsupported safety declarations and routes the decision back to human responders.

Finally, the dashboard, incident log, handoff drafts, and sync export show the operational layer: local records, reviewable packets, and export later when connectivity returns.

FieldAid does not replace emergency teams. It gives them local intelligence, grounded evidence, and auditable handoffs when the network goes down.
```

## On-Screen Checklist

During recording, make sure these are visible at least once:

1. GitHub branch selector: `wt/walkthrough`.
2. Colab notebook with setup cells completed.
3. Webapp loaded in browser.
4. Start Walkthrough button.
5. Gemma model selector showing `gemma4:e4b`.
6. Shelter result with action plan and citations.
7. Image or video preview loaded into the UI.
8. Trust response with verification flags.
9. Dashboard or incident log with created records.
10. Export or sync page.

## Phrases To Avoid

Avoid saying:

```text
It automatically decides what responders should do.
It sends messages for responders.
It determines whether a bridge or route is safe.
The classifier is fully accurate.
It works without setup forever.
```

Use instead:

```text
It supports incident command.
It drafts handoffs for human review.
It flags safety questions for verification.
The classifier is supporting evidence.
After initial setup, the app runs locally.
```
