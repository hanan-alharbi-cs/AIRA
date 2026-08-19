"""
AthleteGuard AI - Database Layer

SQLite database for:
- Users
- Player profiles
- Training / readiness observations
- Upcoming events
- Contracts
- Achievements

Roles:
- coach
- player

This is a hackathon MVP database layer.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "database"
)

DB_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

DB_PATH = (
    DB_DIRECTORY
    / "athleteguard.db"
)


# ============================================================
# Connection
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection.

    Row factory allows dictionary-style access:
        row["username"]
    """

    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# Password hashing
# ============================================================

def hash_password(
    password: str,
) -> str:
    """
    Hash a password using SHA-256 with a random salt.

    Stored format:
        salt$hash
    """

    if not password:
        raise ValueError(
            "Password cannot be empty."
        )

    salt = secrets.token_hex(16)

    password_hash = hashlib.sha256(
        (
            salt
            + password
        ).encode("utf-8")
    ).hexdigest()

    return (
        f"{salt}${password_hash}"
    )


def verify_password(
    password: str,
    stored_hash: str,
) -> bool:
    """
    Verify a password against the stored salt/hash.
    """

    if not password or not stored_hash:
        return False

    try:
        salt, expected_hash = (
            stored_hash.split(
                "$",
                1,
            )
        )

    except ValueError:
        return False

    actual_hash = hashlib.sha256(
        (
            salt
            + password
        ).encode("utf-8")
    ).hexdigest()

    return secrets.compare_digest(
        actual_hash,
        expected_hash,
    )


# ============================================================
# Database initialization
# ============================================================

def initialize_database() -> None:
    """
    Create all required tables if they do not exist.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Users
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK (role IN ('coach', 'player')),
                player_id TEXT NULL,
                full_name TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (player_id)
                    REFERENCES players(player_id)
                    ON DELETE SET NULL
            )
            """
        )

        # ----------------------------------------------------
        # Players
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                team_name TEXT,
                position TEXT,
                date_of_birth TEXT,
                contract_start TEXT,
                contract_end TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # ----------------------------------------------------
        # Athlete observations
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS athlete_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,

                match_id TEXT,
                observation_date TEXT NOT NULL,

                training_duration_min REAL,
                rpe REAL,

                sleep_duration REAL,
                sleep_quality REAL,

                reaction_time_ms REAL,
                recovery_score REAL,

                session_load REAL,
                acute_load REAL,
                chronic_load REAL,
                acwr REAL,

                sleep_deviation REAL,
                reaction_time_deviation REAL,
                recovery_drop REAL,
                warning_points REAL,

                source TEXT NOT NULL DEFAULT 'manual',

                created_at TEXT NOT NULL,

                FOREIGN KEY (player_id)
                    REFERENCES players(player_id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Upcoming events
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_name TEXT NOT NULL,
                event_date TEXT NOT NULL,
                competition TEXT,
                importance INTEGER NOT NULL
                    CHECK (
                        importance BETWEEN 1 AND 5
                    ),
                created_at TEXT NOT NULL
            )
            """
        )

        # ----------------------------------------------------
        # Contracts
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                contract_name TEXT,
                start_date TEXT,
                end_date TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,

                FOREIGN KEY (player_id)
                    REFERENCES players(player_id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Achievements
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                title TEXT NOT NULL,
                achievement_date TEXT,
                description TEXT,
                created_at TEXT NOT NULL,

                FOREIGN KEY (player_id)
                    REFERENCES players(player_id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Training schedule
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Team Training Sessions
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS team_training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                session_name TEXT NOT NULL,
                training_type TEXT NOT NULL,
                session_date TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                duration_min REAL,
                intensity REAL,
                status TEXT NOT NULL DEFAULT 'scheduled'
                    CHECK (status IN ('scheduled','completed','cancelled')),
                notes TEXT,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS team_training_attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_training_session_id INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                attendance_status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (attendance_status IN ('pending','present','absent','excused')),
                attendance_note TEXT,
                updated_by TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (team_training_session_id, player_id),
                FOREIGN KEY (team_training_session_id) REFERENCES team_training_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (player_id) REFERENCES players(player_id) ON DELETE CASCADE
            )
            """
        )


        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                session_name TEXT NOT NULL,
                session_date TEXT NOT NULL,
                duration_min REAL,
                intensity REAL,
                notes TEXT,
                created_at TEXT NOT NULL,

                FOREIGN KEY (player_id)
                    REFERENCES players(player_id)
                    ON DELETE CASCADE
            )
            """
        )

        # ----------------------------------------------------
        # Training attendance
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_session_id INTEGER NOT NULL,
                player_id TEXT NOT NULL,
                attendance_status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (
                        attendance_status IN (
                            'pending',
                            'present',
                            'absent',
                            'excused',
                            'cancelled'
                        )
                    ),
                attendance_note TEXT,
                updated_by TEXT,
                updated_at TEXT NOT NULL,

                UNIQUE (
                    training_session_id,
                    player_id
                ),

                FOREIGN KEY (training_session_id)
                    REFERENCES training_sessions(id)
                    ON DELETE CASCADE,

                FOREIGN KEY (player_id)
                    REFERENCES players(player_id)
                    ON DELETE CASCADE
            )
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# Seed demo data
# ============================================================

def seed_demo_users() -> None:
    """
    Create demo coach and player accounts.

    Credentials:

        Coach
        username: coach
        password: coach123

        Player
        username: player
        password: player123
    """

    initialize_database()

    connection = get_connection()

    try:

        cursor = connection.cursor()

        now = datetime.now().isoformat(
            timespec="seconds"
        )

        # ----------------------------------------------------
        # Demo player profile
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT OR IGNORE INTO players (
                player_id,
                full_name,
                team_name,
                position,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                "P001",
                "Demo Player",
                "AthleteGuard Demo Team",
                "Forward",
                now,
            ),
        )

        # ----------------------------------------------------
        # Demo coach
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
            """,
            ("coach",),
        )

        coach_exists = cursor.fetchone()

        if coach_exists is None:

            cursor.execute(
                """
                INSERT INTO users (
                    username,
                    password_hash,
                    role,
                    player_id,
                    full_name,
                    is_active,
                    created_at
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    "coach",
                    hash_password(
                        "coach123"
                    ),
                    "coach",
                    None,
                    "Demo Coach",
                    1,
                    now,
                ),
            )

        # ----------------------------------------------------
        # Demo player
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username = ?
            """,
            ("player",),
        )

        player_exists = cursor.fetchone()

        if player_exists is None:

            cursor.execute(
                """
                INSERT INTO users (
                    username,
                    password_hash,
                    role,
                    player_id,
                    full_name,
                    is_active,
                    created_at
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    "player",
                    hash_password(
                        "player123"
                    ),
                    "player",
                    "P001",
                    "Demo Player",
                    1,
                    now,
                ),
            )

        # ----------------------------------------------------
        # Additional demo player profiles + accounts
        # ----------------------------------------------------

        demo_players = [
            (
                "P002",
                "Mohammed",
                "AthleteGuard Demo Team",
                "Midfielder",
                "mohammed",
                "mohammed123",
            ),
            (
                "P003",
                "Abdulaziz",
                "AthleteGuard Demo Team",
                "Forward",
                "abdulaziz",
                "abdulaziz123",
            ),
            (
                "P004",
                "Sara",
                "AthleteGuard Demo Team",
                "Forward",
                "sara",
                "sara123",
            ),
            (
                "P005",
                "Faisal",
                "AthleteGuard Demo Team",
                "Midfielder",
                "faisal",
                "faisal123",
            ),
        ]

        for (
            demo_player_id,
            demo_full_name,
            demo_team_name,
            demo_position,
            demo_username,
            demo_password,
        ) in demo_players:

            cursor.execute(
                """
                INSERT OR IGNORE INTO players (
                    player_id,
                    full_name,
                    team_name,
                    position,
                    created_at
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    demo_player_id,
                    demo_full_name,
                    demo_team_name,
                    demo_position,
                    now,
                ),
            )

            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE username = ?
                """,
                (
                    demo_username,
                ),
            )

            demo_user_exists = cursor.fetchone()

            if demo_user_exists is None:

                cursor.execute(
                    """
                    INSERT INTO users (
                        username,
                        password_hash,
                        role,
                        player_id,
                        full_name,
                        is_active,
                        created_at
                    )
                    VALUES (
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?
                    )
                    """,
                    (
                        demo_username,
                        hash_password(
                            demo_password
                        ),
                        "player",
                        demo_player_id,
                        demo_full_name,
                        1,
                        now,
                    ),
                )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# Authentication
# ============================================================

def authenticate_user(
    username: str,
    password: str,
) -> dict[str, Any] | None:
    """
    Authenticate a user and return their basic profile.

    Returns None if credentials are invalid.
    """

    if not username or not password:
        return None

    initialize_database()

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT
                id,
                username,
                password_hash,
                role,
                player_id,
                full_name,
                is_active
            FROM users
            WHERE username = ?
            """,
            (
                username.strip(),
            ),
        ).fetchone()

        if row is None:
            return None

        if not bool(
            row["is_active"]
        ):
            return None

        if not verify_password(
            password,
            row["password_hash"],
        ):
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "player_id": row["player_id"],
            "full_name": row["full_name"],
        }

    finally:

        connection.close()


# ============================================================
# Player management
# ============================================================

def create_player(
    *,
    player_id: str,
    full_name: str,
    team_name: str | None = None,
    position: str | None = None,
    date_of_birth: str | None = None,
    contract_start: str | None = None,
    contract_end: str | None = None,
) -> None:
    """
    Create a player profile.
    """

    if not player_id:
        raise ValueError(
            "player_id is required."
        )

    if not full_name:
        raise ValueError(
            "full_name is required."
        )

    initialize_database()

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO players (
                player_id,
                full_name,
                team_name,
                position,
                date_of_birth,
                contract_start,
                contract_end,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                player_id,
                full_name,
                team_name,
                position,
                date_of_birth,
                contract_start,
                contract_end,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        connection.commit()

    finally:

        connection.close()


def get_player(
    player_id: str,
) -> dict[str, Any] | None:
    """
    Get one player by ID.
    """

    initialize_database()

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT *
            FROM players
            WHERE player_id = ?
            """,
            (
                player_id,
            ),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:

        connection.close()


def get_all_players() -> list[dict[str, Any]]:
    """
    Return all player profiles.
    """

    initialize_database()

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM players
            ORDER BY player_id
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# Athlete observations
# ============================================================

def save_observation(
    *,
    player_id: str,
    match_id: str | None,
    observation_date: str,
    training_duration_min: float | None,
    rpe: float | None,
    sleep_duration: float | None,
    sleep_quality: float | None,
    reaction_time_ms: float | None,
    recovery_score: float | None,
    session_load: float | None = None,
    acute_load: float | None = None,
    chronic_load: float | None = None,
    acwr: float | None = None,
    sleep_deviation: float | None = None,
    reaction_time_deviation: float | None = None,
    recovery_drop: float | None = None,
    warning_points: float | None = None,
    source: str = "manual",
) -> int:
    """
    Save a real observation submitted by a user/system.

    Returns:
        inserted observation ID.
    """

    initialize_database()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO athlete_observations (
                player_id,
                match_id,
                observation_date,
                training_duration_min,
                rpe,
                sleep_duration,
                sleep_quality,
                reaction_time_ms,
                recovery_score,
                session_load,
                acute_load,
                chronic_load,
                acwr,
                sleep_deviation,
                reaction_time_deviation,
                recovery_drop,
                warning_points,
                source,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                player_id,
                match_id,
                observation_date,
                training_duration_min,
                rpe,
                sleep_duration,
                sleep_quality,
                reaction_time_ms,
                recovery_score,
                session_load,
                acute_load,
                chronic_load,
                acwr,
                sleep_deviation,
                reaction_time_deviation,
                recovery_drop,
                warning_points,
                source,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:

        connection.close()


def get_latest_observation(
    player_id: str,
) -> dict[str, Any] | None:
    """
    Return the latest observation for a player.
    """

    initialize_database()

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT *
            FROM athlete_observations
            WHERE player_id = ?
            ORDER BY
                observation_date DESC,
                id DESC
            LIMIT 1
            """,
            (
                player_id,
            ),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:

        connection.close()


def get_player_observations(
    player_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return recent observations for a player.
    """

    initialize_database()

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM athlete_observations
            WHERE player_id = ?
            ORDER BY
                observation_date DESC,
                id DESC
            LIMIT ?
            """,
            (
                player_id,
                int(limit),
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# Events
# ============================================================

def create_event(
    *,
    event_id: str,
    event_name: str,
    event_date: str,
    competition: str | None,
    importance: int,
) -> None:
    """
    Save an upcoming event.
    """

    if not 1 <= int(
        importance
    ) <= 5:

        raise ValueError(
            "importance must be between 1 and 5."
        )

    initialize_database()

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO events (
                event_id,
                event_name,
                event_date,
                competition,
                importance,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                event_id,
                event_name,
                event_date,
                competition,
                int(importance),
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        connection.commit()

    finally:

        connection.close()


def get_upcoming_events() -> list[dict[str, Any]]:
    """
    Return all events sorted by date.
    """

    initialize_database()

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM events
            ORDER BY event_date ASC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# Contracts
# ============================================================

def add_contract(
    *,
    player_id: str,
    contract_name: str,
    start_date: str | None,
    end_date: str | None,
    contract_status: str = "active",
) -> int:
    """
    Add a player contract.
    """

    initialize_database()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO contracts (
                player_id,
                contract_name,
                start_date,
                end_date,
                status,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                player_id,
                contract_name,
                start_date,
                end_date,
                contract_status,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:

        connection.close()


def get_player_contracts(
    player_id: str,
) -> list[dict[str, Any]]:
    """
    Return contracts for one player.
    """

    initialize_database()

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM contracts
            WHERE player_id = ?
            ORDER BY end_date DESC
            """,
            (
                player_id,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# Achievements
# ============================================================

def add_achievement(
    *,
    player_id: str,
    title: str,
    achievement_date: str | None,
    description: str | None = None,
) -> int:
    """
    Add an achievement for a player.
    """

    initialize_database()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO achievements (
                player_id,
                title,
                achievement_date,
                description,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                player_id,
                title,
                achievement_date,
                description,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:

        connection.close()


def get_player_achievements(
    player_id: str,
) -> list[dict[str, Any]]:
    """
    Return achievements for one player.
    """

    initialize_database()

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM achievements
            WHERE player_id = ?
            ORDER BY achievement_date DESC
            """,
            (
                player_id,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# Training schedule
# ============================================================

def add_training_session(
    *,
    player_id: str,
    session_name: str,
    session_date: str,
    duration_min: float | None = None,
    intensity: float | None = None,
    notes: str | None = None,
) -> int:
    """
    Add a training session.
    """

    initialize_database()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO training_sessions (
                player_id,
                session_name,
                session_date,
                duration_min,
                intensity,
                notes,
                created_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                player_id,
                session_name,
                session_date,
                duration_min,
                intensity,
                notes,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:

        connection.close()


def get_player_training_sessions(
    player_id: str,
) -> list[dict[str, Any]]:
    """
    Return training sessions for a player.
    """

    initialize_database()

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM training_sessions
            WHERE player_id = ?
            ORDER BY session_date DESC
            """,
            (
                player_id,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()



# ============================================================
# Training Attendance
# ============================================================

def add_training_attendance(
    *,
    training_session_id: int,
    player_id: str,
    attendance_status: str = "pending",
    attendance_note: str | None = None,
    updated_by: str = "coach",
) -> int:
    """
    Create or update attendance for a training session.

    Coach/admin is the source of the official attendance status.
    """

    allowed = {
        "pending",
        "present",
        "absent",
        "excused",
        "cancelled",
    }

    if attendance_status not in allowed:
        raise ValueError(
            "Invalid attendance status."
        )

    initialize_database()

    connection = get_connection()

    try:
        now = datetime.now().isoformat(
            timespec="seconds"
        )

        cursor = connection.execute(
            """
            INSERT INTO training_attendance (
                training_session_id,
                player_id,
                attendance_status,
                attendance_note,
                updated_by,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(
                training_session_id,
                player_id
            )
            DO UPDATE SET
                attendance_status =
                    excluded.attendance_status,
                attendance_note =
                    excluded.attendance_note,
                updated_by =
                    excluded.updated_by,
                updated_at =
                    excluded.updated_at
            """,
            (
                int(training_session_id),
                str(player_id),
                attendance_status,
                attendance_note,
                updated_by,
                now,
            ),
        )

        connection.commit()

        row = connection.execute(
            """
            SELECT id
            FROM training_attendance
            WHERE training_session_id = ?
              AND player_id = ?
            """,
            (
                int(training_session_id),
                str(player_id),
            ),
        ).fetchone()

        return int(row["id"])

    finally:
        connection.close()


def get_player_training_attendance(
    player_id: str,
) -> list[dict[str, Any]]:
    """
    Return training sessions with their attendance status
    for one player.
    """

    initialize_database()

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                ts.id AS training_session_id,
                ts.player_id,
                ts.session_name,
                ts.session_date,
                ts.duration_min,
                ts.intensity,
                ts.notes,
                ts.created_at,
                COALESCE(
                    ta.attendance_status,
                    'pending'
                ) AS attendance_status,
                ta.attendance_note,
                ta.updated_by,
                ta.updated_at
            FROM training_sessions ts
            LEFT JOIN training_attendance ta
                ON ta.training_session_id = ts.id
               AND ta.player_id = ts.player_id
            WHERE ts.player_id = ?
            ORDER BY ts.session_date DESC,
                     ts.id DESC
            """,
            (
                str(player_id),
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


def get_all_training_attendance(
    session_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return training/attendance records for the coach dashboard.
    Optional filter by session date.
    """

    initialize_database()

    connection = get_connection()

    try:
        if session_date:

            rows = connection.execute(
                """
                SELECT
                    ts.id AS training_session_id,
                    ts.player_id,
                    p.full_name,
                    ts.session_name,
                    ts.session_date,
                    ts.duration_min,
                    ts.intensity,
                    ts.notes,
                    COALESCE(
                        ta.attendance_status,
                        'pending'
                    ) AS attendance_status,
                    ta.attendance_note
                FROM training_sessions ts
                INNER JOIN players p
                    ON p.player_id = ts.player_id
                LEFT JOIN training_attendance ta
                    ON ta.training_session_id = ts.id
                   AND ta.player_id = ts.player_id
                WHERE ts.session_date = ?
                ORDER BY p.full_name ASC
                """,
                (
                    session_date,
                ),
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT
                    ts.id AS training_session_id,
                    ts.player_id,
                    p.full_name,
                    ts.session_name,
                    ts.session_date,
                    ts.duration_min,
                    ts.intensity,
                    ts.notes,
                    COALESCE(
                        ta.attendance_status,
                        'pending'
                    ) AS attendance_status,
                    ta.attendance_note
                FROM training_sessions ts
                INNER JOIN players p
                    ON p.player_id = ts.player_id
                LEFT JOIN training_attendance ta
                    ON ta.training_session_id = ts.id
                   AND ta.player_id = ts.player_id
                ORDER BY ts.session_date DESC,
                         p.full_name ASC
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


# ============================================================
# Team Training Sessions
# ============================================================

def add_team_training_session(*, team_name: str, session_name: str, training_type: str, session_date: str, start_time: str | None = None, end_time: str | None = None, duration_min: float | None = None, intensity: float | None = None, status: str = "scheduled", notes: str | None = None, created_by: str = "coach") -> int:
    """Create one team session and initialize attendance for all players in the team."""
    if status not in {"scheduled", "completed", "cancelled"}:
        raise ValueError("Invalid training status.")
    initialize_database()
    con = get_connection()
    try:
        cur = con.execute("""INSERT INTO team_training_sessions (team_name, session_name, training_type, session_date, start_time, end_time, duration_min, intensity, status, notes, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (team_name.strip(), session_name.strip(), training_type.strip(), session_date, start_time, end_time, duration_min, intensity, status, notes, created_by, datetime.now().isoformat(timespec="seconds")))
        session_id = int(cur.lastrowid)
        players = con.execute("SELECT player_id FROM players WHERE team_name = ?", (team_name.strip(),)).fetchall()
        now = datetime.now().isoformat(timespec="seconds")
        for row in players:
            con.execute("""INSERT OR IGNORE INTO team_training_attendance (team_training_session_id, player_id, attendance_status, updated_by, updated_at) VALUES (?, ?, 'pending', ?, ?)""", (session_id, str(row["player_id"]), created_by, now))
        con.commit()
        return session_id
    finally:
        con.close()


def get_team_training_sessions(team_name: str | None = None, session_date: str | None = None) -> list[dict[str, Any]]:
    initialize_database()
    con = get_connection()
    try:
        query = "SELECT * FROM team_training_sessions WHERE 1=1"
        params: list[Any] = []
        if team_name:
            query += " AND team_name = ?"; params.append(team_name)
        if session_date:
            query += " AND session_date = ?"; params.append(session_date)
        query += " ORDER BY session_date DESC, id DESC"
        return [dict(r) for r in con.execute(query, tuple(params)).fetchall()]
    finally:
        con.close()


def set_team_training_attendance(*, team_training_session_id: int, player_id: str, attendance_status: str, attendance_note: str | None = None, updated_by: str = "coach") -> int:
    if attendance_status not in {"pending", "present", "absent", "excused"}:
        raise ValueError("Invalid attendance status.")
    initialize_database()
    con = get_connection()
    try:
        now = datetime.now().isoformat(timespec="seconds")
        con.execute("""INSERT INTO team_training_attendance (team_training_session_id, player_id, attendance_status, attendance_note, updated_by, updated_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(team_training_session_id, player_id) DO UPDATE SET attendance_status=excluded.attendance_status, attendance_note=excluded.attendance_note, updated_by=excluded.updated_by, updated_at=excluded.updated_at""", (int(team_training_session_id), str(player_id), attendance_status, attendance_note, updated_by, now))
        con.commit()
        row=con.execute("SELECT id FROM team_training_attendance WHERE team_training_session_id=? AND player_id=?", (int(team_training_session_id), str(player_id))).fetchone()
        return int(row["id"])
    finally:
        con.close()


def get_team_training_attendance(team_training_session_id: int | None = None, player_id: str | None = None) -> list[dict[str, Any]]:
    initialize_database()
    con = get_connection()
    try:
        query = """SELECT
                tta.id,
                tta.team_training_session_id,
                tta.player_id,
                p.full_name,
                p.team_name,
                p.position,
                tta.attendance_status,
                tta.attendance_note,
                tta.updated_by,
                tta.updated_at,
                tts.session_name,
                tts.training_type,
                tts.session_date,
                tts.start_time,
                tts.end_time,
                tts.duration_min,
                tts.intensity,
                tts.notes,
                tts.status AS training_status
            FROM team_training_attendance tta
            INNER JOIN players p
                ON p.player_id = tta.player_id
            INNER JOIN team_training_sessions tts
                ON tts.id = tta.team_training_session_id
            WHERE 1=1"""
        params: list[Any]=[]
        if team_training_session_id is not None:
            query += " AND tta.team_training_session_id = ?"; params.append(int(team_training_session_id))
        if player_id is not None:
            query += " AND tta.player_id = ?"; params.append(str(player_id))
        query += " ORDER BY tts.session_date DESC, p.full_name ASC"
        return [dict(r) for r in con.execute(query, tuple(params)).fetchall()]
    finally:
        con.close()


# ============================================================
# Startup
# ============================================================

def initialize_and_seed() -> None:
    """
    Initialize database and create demo accounts.
    """

    initialize_database()
    seed_demo_users()


if __name__ == "__main__":

    initialize_and_seed()

    print(
        "AthleteGuard database initialized."
    )

    print(
        f"Database: {DB_PATH}"
    )

    print(
        "Demo coach: "
        "coach / coach123"
    )

    print(
        "Demo player: "
        "player / player123"
    )
    # ============================================================
# MATCHES — Past & Upcoming
# ============================================================

def initialize_match_database() -> None:
    """
    Create the matches table used for:
    - Previous matches
    - Upcoming matches
    - Match results
    - Competition details
    - Venue
    - Importance
    """

    initialize_database()

    connection = get_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                match_id TEXT NOT NULL UNIQUE,

                match_date TEXT,

                status TEXT NOT NULL
                    CHECK (
                        status IN (
                            'past',
                            'upcoming'
                        )
                    ),

                competition TEXT NOT NULL,

                match_type TEXT NOT NULL,

                team_home TEXT NOT NULL,

                team_away TEXT NOT NULL,

                home_score INTEGER,

                away_score INTEGER,

                venue TEXT,

                importance INTEGER NOT NULL DEFAULT 3
                    CHECK (
                        importance BETWEEN 1 AND 5
                    ),

                notes TEXT DEFAULT '',

                created_by TEXT NOT NULL DEFAULT 'system',

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_matches_date
            ON matches(match_date)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_matches_status
            ON matches(status)
            """
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# Save / Update Match
# ============================================================

def save_match(
    *,
    match_id: str,
    match_date: str | None,
    status: str,
    competition: str,
    match_type: str,
    team_home: str,
    team_away: str,
    home_score: int | None = None,
    away_score: int | None = None,
    venue: str | None = None,
    importance: int = 3,
    notes: str = "",
    created_by: str = "system",
) -> None:
    """
    Create or update a match.

    status:
        past
        upcoming

    For upcoming matches, scores are automatically cleared.
    """

    if status not in {
        "past",
        "upcoming",
    }:

        raise ValueError(
            "status must be 'past' or 'upcoming'."
        )

    if not 1 <= int(importance) <= 5:

        raise ValueError(
            "importance must be between 1 and 5."
        )

    if status == "upcoming":

        home_score = None
        away_score = None

    initialize_match_database()

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO matches (
                match_id,
                match_date,
                status,
                competition,
                match_type,
                team_home,
                team_away,
                home_score,
                away_score,
                venue,
                importance,
                notes,
                created_by
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )

            ON CONFLICT(match_id)
            DO UPDATE SET
                match_date = excluded.match_date,
                status = excluded.status,
                competition = excluded.competition,
                match_type = excluded.match_type,
                team_home = excluded.team_home,
                team_away = excluded.team_away,
                home_score = excluded.home_score,
                away_score = excluded.away_score,
                venue = excluded.venue,
                importance = excluded.importance,
                notes = excluded.notes,
                created_by = excluded.created_by
            """,
            (
                match_id.strip(),
                match_date,
                status,
                competition.strip(),
                match_type.strip(),
                team_home.strip(),
                team_away.strip(),
                home_score,
                away_score,
                venue,
                int(importance),
                notes,
                created_by,
            ),
        )

        connection.commit()

    finally:

        connection.close()



# ============================================================
# Update Match
# ============================================================

def update_match(
    *,
    match_id: str,
    match_date: str | None,
    status: str,
    competition: str,
    match_type: str,
    team_home: str,
    team_away: str,
    home_score: int | None = None,
    away_score: int | None = None,
    venue: str | None = None,
    importance: int = 3,
    notes: str = "",
    created_by: str = "coach",
) -> None:
    """
    Update an existing match record.

    Past matches may contain scores.
    Upcoming matches have no scores.
    """

    if status not in {
        "past",
        "upcoming",
    }:
        raise ValueError(
            "status must be 'past' or 'upcoming'."
        )

    if not 1 <= int(importance) <= 5:
        raise ValueError(
            "importance must be between 1 and 5."
        )

    if status == "upcoming":
        home_score = None
        away_score = None

    initialize_match_database()

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            UPDATE matches
            SET
                match_date = ?,
                status = ?,
                competition = ?,
                match_type = ?,
                team_home = ?,
                team_away = ?,
                home_score = ?,
                away_score = ?,
                venue = ?,
                importance = ?,
                notes = ?,
                created_by = ?
            WHERE match_id = ?
            """,
            (
                match_date,
                status,
                competition.strip(),
                match_type.strip(),
                team_home.strip(),
                team_away.strip(),
                home_score,
                away_score,
                venue,
                int(importance),
                notes,
                created_by,
                match_id.strip(),
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Match '{match_id}' was not found."
            )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# Get Matches
# ============================================================

def get_matches(
    status: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get all matches.

    status=None:
        return past + upcoming

    status="past":
        previous matches only

    status="upcoming":
        future matches only
    """

    initialize_match_database()

    connection = get_connection()

    try:

        if status is None:

            rows = connection.execute(
                """
                SELECT *
                FROM matches
                ORDER BY
                    CASE
                        WHEN match_date IS NULL
                        THEN 1
                        ELSE 0
                    END,
                    match_date ASC,
                    id ASC
                """
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT *
                FROM matches
                WHERE status = ?

                ORDER BY
                    CASE
                        WHEN match_date IS NULL
                        THEN 1
                        ELSE 0
                    END,
                    match_date ASC,
                    id ASC
                """,
                (
                    status,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# Get One Match
# ============================================================

def get_match(
    match_id: str,
) -> dict[str, Any] | None:
    """
    Get one match by match_id.
    """

    initialize_match_database()

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT *
            FROM matches
            WHERE match_id = ?
            """,
            (
                match_id,
            ),
        ).fetchone()

        if row is None:

            return None

        return dict(row)

    finally:

        connection.close()


# ============================================================
# Import Existing Match Results
# ============================================================

def import_previous_matches() -> int:
    """
    Import the existing historical matches from:

        data/raw/matches.csv

    The source currently provides:
        match_id
        team_home
        team_away
        home_score
        away_score

    It does NOT provide:
        date
        competition

    Therefore those missing fields are intentionally left
    as 'غير مسجل' rather than inventing information.
    """

    import csv

    initialize_match_database()

    raw_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "matches.csv"
    )

    if not raw_file.exists():

        return 0

    imported_count = 0

    with raw_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            match_id = (
                row.get(
                    "match_id",
                    "",
                )
                .strip()
            )

            if not match_id:

                continue

            existing = get_match(
                match_id
            )

            if existing:

                continue

            home_score = int(
                row["home_score"]
            )

            away_score = int(
                row["away_score"]
            )

            save_match(
                match_id=match_id,
                match_date=None,
                status="past",
                competition="غير مسجل",
                match_type="مباراة سابقة",
                team_home=(
                    row.get(
                        "team_home",
                        "",
                    ).strip()
                ),
                team_away=(
                    row.get(
                        "team_away",
                        "",
                    ).strip()
                ),
                home_score=home_score,
                away_score=away_score,
                importance=3,
                notes=(
                    "مستوردة من matches.csv. "
                    "التاريخ ونوع المسابقة غير موجودين "
                    "في المصدر الحالي."
                ),
                created_by="system",
            )

            imported_count += 1

    return imported_count


# ============================================================
# Match Result Helper
# ============================================================

def get_match_result_label(
    match: dict[str, Any],
) -> str:
    """
    Return:
        فوز
        تعادل
        خسارة

    based on the team's home/away position.
    """

    home_score = match.get(
        "home_score"
    )

    away_score = match.get(
        "away_score"
    )

    if (
        home_score is None
        or away_score is None
    ):

        return "لم تلعب بعد"

    if home_score == away_score:

        return "تعادل"

    if home_score > away_score:

        return "فوز الفريق المضيف"

    return "فوز الفريق الضيف"