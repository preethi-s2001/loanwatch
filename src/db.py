"""
SQLite Database Persistence module for LoanWatch NBFC Early Warning System.

Manages persistent storage for:
- loans_scored: Scored loan snapshots, calibrated probabilities, risk tiers, warning signs.
- followups: Workflow audit trail for collection follow-up interactions and status transitions.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import DB_PATH, FOLLOWUP_STATUSES

logger = logging.getLogger(__name__)


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory configured."""
    target_path = Path(db_path or DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes tables and indexes if they do not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loans_scored (
        loan_id TEXT PRIMARY KEY,
        observation_date TEXT,
        scored_at TEXT NOT NULL,
        probability REAL NOT NULL,
        bucket TEXT NOT NULL,
        warnings TEXT NOT NULL,
        collector_id TEXT NOT NULL,
        loan_amount REAL NOT NULL,
        outstanding_amount REAL NOT NULL,
        emi_amount REAL NOT NULL,
        monthly_income REAL NOT NULL,
        days_past_due INTEGER NOT NULL,
        suggested_action TEXT NOT NULL
    );
    """)

    # Gracefully add observation_date column if table existed previously without it
    try:
        cursor.execute("ALTER TABLE loans_scored ADD COLUMN observation_date TEXT;")
    except sqlite3.OperationalError:
        pass

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_loans_scored_collector ON loans_scored(collector_id);
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_loans_scored_priority ON loans_scored(probability DESC, outstanding_amount DESC);
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS followups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        loan_id TEXT NOT NULL,
        collector_id TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        updated_by TEXT NOT NULL,
        FOREIGN KEY(loan_id) REFERENCES loans_scored(loan_id)
    );
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_followups_loan ON followups(loan_id);
    """)
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_followups_collector_status ON followups(collector_id, status);
    """)

    conn.commit()
    conn.close()
    logger.info("Initialized SQLite database schema at %s", db_path or DB_PATH)


def save_scored_loans(
    df_scored: pd.DataFrame,
    db_path: Optional[Path] = None,
    clear_existing: bool = False,
) -> int:
    """
    Persists scored loan records into SQLite and auto-seeds initial Pending followups.
    If clear_existing is True, clears existing scored loans and followups first.
    """
    if df_scored.empty:
        return 0

    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    if clear_existing:
        cursor.execute("DELETE FROM followups")
        cursor.execute("DELETE FROM loans_scored")
        conn.commit()

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert or replace scored loans
    insert_sql = """
    INSERT OR REPLACE INTO loans_scored (
        loan_id, observation_date, scored_at, probability, bucket, warnings,
        collector_id, loan_amount, outstanding_amount, emi_amount,
        monthly_income, days_past_due, suggested_action
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    records = []
    for _, row in df_scored.iterrows():
        warnings_val = row["warning_signs"]
        warnings_json = json.dumps(warnings_val) if isinstance(warnings_val, list) else str(warnings_val)
        obs_dt = str(row["observation_date"]) if "observation_date" in row and pd.notna(row["observation_date"]) else ""

        records.append((
            str(row["loan_id"]),
            obs_dt,
            now_iso,
            float(row["probability"]),
            str(row["risk_bucket"]),
            warnings_json,
            str(row["collector_id"]),
            float(row["loan_amount"]),
            float(row["outstanding_amount"]),
            float(row["emi_amount"]),
            float(row["monthly_income"]),
            int(row["days_past_due"]),
            str(row["suggested_action"]),
        ))

    cursor.executemany(insert_sql, records)

    # Seed initial "Pending" followup for accounts that do not have any followup records yet
    cursor.execute("""
    SELECT DISTINCT loan_id FROM followups
    """)
    existing_loan_ids = {r[0] for r in cursor.fetchall()}

    seed_followups = []
    for r in records:
        l_id = r[0]
        c_id = r[5]
        if l_id not in existing_loan_ids:
            seed_followups.append((
                l_id,
                c_id,
                "Pending",
                "Automated assignment to daily collection queue.",
                now_iso,
                None,
                "System (Auto-assigned)",
            ))

    if seed_followups:
        cursor.executemany("""
        INSERT INTO followups (loan_id, collector_id, status, notes, created_at, completed_at, updated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, seed_followups)

    conn.commit()
    conn.close()

    logger.info("Saved %d scored loans (seeded %d initial followups).", len(records), len(seed_followups))
    return len(records)


def get_all_scored_loans_with_status(
    db_path: Optional[Path] = None,
    collector_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches scored loans joined with their latest follow-up status.
    Ordered by probability descending, tie-breaking by outstanding_amount descending.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)

    query = """
    WITH LatestFollowup AS (
        SELECT 
            loan_id,
            status AS followup_status,
            notes AS latest_notes,
            created_at AS followup_created_at,
            completed_at,
            updated_by,
            ROW_NUMBER() OVER(PARTITION BY loan_id ORDER BY id DESC) as rn
        FROM followups
    )
    SELECT 
        s.loan_id,
        s.observation_date,
        s.scored_at,
        s.probability,
        s.bucket AS risk_bucket,
        s.warnings,
        s.collector_id,
        s.loan_amount,
        s.outstanding_amount,
        s.emi_amount,
        s.monthly_income,
        s.days_past_due,
        s.suggested_action,
        COALESCE(lf.followup_status, 'Pending') AS followup_status,
        lf.latest_notes,
        lf.completed_at,
        lf.updated_by
    FROM loans_scored s
    LEFT JOIN LatestFollowup lf ON s.loan_id = lf.loan_id AND lf.rn = 1
    """

    params = []
    if collector_id:
        query += " WHERE s.collector_id = ?"
        params.append(collector_id)

    query += " ORDER BY s.probability DESC, s.outstanding_amount DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Parse JSON warnings back into Python list
    if not df.empty and "warnings" in df.columns:
        parsed_warnings = []
        for w in df["warnings"]:
            try:
                parsed_warnings.append(json.loads(w))
            except Exception:
                parsed_warnings.append([str(w)])
        df["warning_signs"] = parsed_warnings

    return df


def add_or_update_followup(
    loan_id: str,
    collector_id: str,
    status: str,
    notes: str,
    updated_by: str = "Collector",
    db_path: Optional[Path] = None,
) -> int:
    """
    Appends a new follow-up interaction record for an account.
    If status is Completed or Paid, sets completed_at timestamp.
    """
    if status not in FOLLOWUP_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {FOLLOWUP_STATUSES}")

    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completed_at = now_iso if status in ["Completed", "Paid"] else None

    cursor.execute("""
    INSERT INTO followups (loan_id, collector_id, status, notes, created_at, completed_at, updated_by)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (loan_id, collector_id, status, notes, now_iso, completed_at, updated_by))

    followup_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return followup_id


def get_loan_followup_history(loan_id: str, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Fetches complete chronological audit trail of follow-ups for a specific loan."""
    init_db(db_path)
    conn = get_db_connection(db_path)

    query = """
    SELECT id, loan_id, collector_id, status, notes, created_at, completed_at, updated_by
    FROM followups
    WHERE loan_id = ?
    ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn, params=[loan_id])
    conn.close()
    return df


def get_collector_workload_metrics(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Computes summary KPI breakdown of accounts per collector and status."""
    init_db(db_path)
    conn = get_db_connection(db_path)

    query = """
    WITH LatestFollowup AS (
        SELECT 
            loan_id,
            status,
            ROW_NUMBER() OVER(PARTITION BY loan_id ORDER BY id DESC) as rn
        FROM followups
    )
    SELECT 
        s.collector_id,
        COUNT(s.loan_id) AS total_assigned,
        SUM(CASE WHEN s.bucket = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_count,
        SUM(CASE WHEN COALESCE(lf.status, 'Pending') = 'Pending' THEN 1 ELSE 0 END) AS pending_count,
        SUM(CASE WHEN COALESCE(lf.status, 'Pending') IN ('Contacted – No Answer', 'Contacted – Promised to Pay', 'Escalated') THEN 1 ELSE 0 END) AS in_progress_count,
        SUM(CASE WHEN COALESCE(lf.status, 'Pending') IN ('Completed', 'Paid') THEN 1 ELSE 0 END) AS completed_count,
        ROUND(SUM(s.outstanding_amount), 2) AS total_outstanding
    FROM loans_scored s
    LEFT JOIN LatestFollowup lf ON s.loan_id = lf.loan_id AND lf.rn = 1
    GROUP BY s.collector_id
    ORDER BY s.collector_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
