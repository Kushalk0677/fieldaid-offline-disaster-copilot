from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.models import AnalysisResult, IncidentRecord, OutboundMessageRecord, SyncQueueRecord


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "fieldaid.sqlite"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scenario_type TEXT NOT NULL,
                location TEXT NOT NULL,
                urgency TEXT NOT NULL,
                summary TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new',
                sms_update TEXT NOT NULL,
                analysis_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outbound_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                incident_id INTEGER,
                channel TEXT NOT NULL,
                destination TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'drafted',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                payload_json TEXT NOT NULL,
                last_exported_at TEXT,
                UNIQUE(entity_type, entity_id)
            )
            """
        )
        _backfill_incident_sync_queue(conn)


def save_incident(result: AnalysisResult, db_path: Path = DB_PATH) -> AnalysisResult:
    init_db(db_path)
    payload = result.model_dump(mode="json")
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO incidents
            (scenario_type, location, urgency, summary, status, sms_update, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.scenario_type,
                result.location,
                result.urgency,
                result.summary,
                "new",
                result.sms_update,
                json.dumps(payload),
            ),
        )
        row = conn.execute("SELECT created_at FROM incidents WHERE id = ?", (cursor.lastrowid,)).fetchone()
        payload["report_id"] = int(cursor.lastrowid)
        if row:
            payload["created_at"] = row["created_at"]
        conn.execute(
            """
            INSERT OR REPLACE INTO sync_queue
            (entity_type, entity_id, status, payload_json, last_exported_at)
            VALUES (?, ?, 'queued', ?, NULL)
            """,
            ("incident", int(cursor.lastrowid), json.dumps(payload)),
        )
    result.report_id = int(cursor.lastrowid)
    result.created_at = row["created_at"] if row else None
    return result


def list_incidents(db_path: Path = DB_PATH) -> list[IncidentRecord]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()
    return [_row_to_record(row) for row in rows]


def update_status(report_id: int, status: str, db_path: Path = DB_PATH) -> IncidentRecord | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("UPDATE incidents SET status = ? WHERE id = ?", (status, report_id))
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (report_id,)).fetchone()
        if row:
            record = _row_to_record(row)
            payload = dict(record.analysis_json)
            payload["status"] = record.status
            payload["report_id"] = record.id
            payload["created_at"] = record.created_at
            conn.execute(
                """
                INSERT OR REPLACE INTO sync_queue
                (entity_type, entity_id, status, payload_json, last_exported_at)
                VALUES (?, ?, 'queued', ?, NULL)
                """,
                ("incident", report_id, json.dumps(payload)),
            )
    return _row_to_record(row) if row else None


def list_sync_queue(status: str | None = None, db_path: Path = DB_PATH) -> list[SyncQueueRecord]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        if status:
            rows = conn.execute("SELECT * FROM sync_queue WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM sync_queue ORDER BY id DESC").fetchall()
    return [_sync_row_to_record(row) for row in rows]


def mark_sync_exported(ids: list[int] | None = None, db_path: Path = DB_PATH) -> list[SyncQueueRecord]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE sync_queue
                SET status = 'exported', last_exported_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                tuple(ids),
            )
        else:
            conn.execute(
                """
                UPDATE sync_queue
                SET status = 'exported', last_exported_at = CURRENT_TIMESTAMP
                WHERE status = 'queued'
                """
            )
    return list_sync_queue(db_path=db_path)


def create_outbound_message(
    *,
    channel: str,
    destination: str,
    message: str,
    incident_id: int | None = None,
    metadata_json: dict[str, Any] | None = None,
    db_path: Path = DB_PATH,
) -> OutboundMessageRecord:
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO outbound_messages
            (incident_id, channel, destination, message, status, metadata_json)
            VALUES (?, ?, ?, ?, 'drafted', ?)
            """,
            (incident_id, channel, destination, message, json.dumps(metadata_json or {})),
        )
        row = conn.execute("SELECT * FROM outbound_messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _outbound_row_to_record(row)


def list_outbound_messages(db_path: Path = DB_PATH) -> list[OutboundMessageRecord]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM outbound_messages ORDER BY id DESC").fetchall()
    return [_outbound_row_to_record(row) for row in rows]


def update_outbound_status(message_id: int, status: str, db_path: Path = DB_PATH) -> OutboundMessageRecord | None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("UPDATE outbound_messages SET status = ? WHERE id = ?", (status, message_id))
        row = conn.execute("SELECT * FROM outbound_messages WHERE id = ?", (message_id,)).fetchone()
    return _outbound_row_to_record(row) if row else None


def _row_to_record(row: sqlite3.Row) -> IncidentRecord:
    analysis_json: dict[str, Any] = json.loads(row["analysis_json"])
    return IncidentRecord(
        id=row["id"],
        created_at=row["created_at"],
        scenario_type=row["scenario_type"],
        location=row["location"],
        urgency=row["urgency"],
        summary=row["summary"],
        status=row["status"],
        sms_update=row["sms_update"],
        analysis_json=analysis_json,
    )


def _sync_row_to_record(row: sqlite3.Row) -> SyncQueueRecord:
    return SyncQueueRecord(
        id=row["id"],
        created_at=row["created_at"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        status=row["status"],
        payload_json=json.loads(row["payload_json"]),
        last_exported_at=row["last_exported_at"],
    )


def _outbound_row_to_record(row: sqlite3.Row) -> OutboundMessageRecord:
    return OutboundMessageRecord(
        id=row["id"],
        created_at=row["created_at"],
        incident_id=row["incident_id"],
        channel=row["channel"],
        destination=row["destination"],
        message=row["message"],
        status=row["status"],
        metadata_json=json.loads(row["metadata_json"]),
    )


def _backfill_incident_sync_queue(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT * FROM incidents").fetchall()
    for row in rows:
        record = _row_to_record(row)
        payload = dict(record.analysis_json)
        payload["report_id"] = record.id
        payload["created_at"] = record.created_at
        payload["status"] = record.status
        conn.execute(
            """
            INSERT OR IGNORE INTO sync_queue
            (entity_type, entity_id, status, payload_json)
            VALUES (?, ?, 'queued', ?)
            """,
            ("incident", record.id, json.dumps(payload)),
        )
