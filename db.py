# db.py
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # predictions table: id, ts, model_version, input_json, prob, pred, user_role, notes
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        model_version TEXT,
        input_json TEXT,
        prob REAL,
        pred INTEGER,
        user TEXT,
        notes TEXT
    )
    """)
    # model_versions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT,
        saved_at TEXT,
        notes TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_prediction(model_version, input_dict, prob, pred, user="anonymous", notes=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (ts, model_version, input_json, prob, pred, user, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.utcnow().isoformat(), model_version, json.dumps(input_dict), float(prob), int(pred), user, notes))
    conn.commit()
    conn.close()

def add_model_version(version, saved_at=None, notes=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO model_versions (version, saved_at, notes) VALUES (?, ?, ?)",
              (version, saved_at or datetime.utcnow().isoformat(), notes))
    conn.commit()
    conn.close()

def fetch_predictions(limit=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, ts, model_version, input_json, prob, pred, user, notes FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "ts": r[1],
            "model_version": r[2],
            "input": json.loads(r[3]),
            "prob": r[4],
            "pred": r[5],
            "user": r[6],
            "notes": r[7]
        })
    return results
