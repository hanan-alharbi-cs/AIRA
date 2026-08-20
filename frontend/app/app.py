from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from ai.scoring.event_readiness import calculate_event_readiness
from backend.database import (
    authenticate_user,
    get_player,
    get_latest_observation,
    initialize_and_seed,
    save_observation,
    initialize_match_database,
    import_previous_matches,
    get_matches,
    save_match,
    update_match,
    add_contract,
    get_player_contracts,
    add_achievement,
    get_player_achievements,
    add_training_session,
    get_player_training_sessions,
    add_training_attendance,
    get_player_training_attendance,
    get_all_training_attendance,
    get_all_players,
    add_team_training_session,
    get_team_training_sessions,
    set_team_training_attendance,
    get_team_training_attendance,
)
from ai.scoring.event_readiness import calculate_event_readiness
from backend.database import (
    authenticate_user,
    get_player,
    initialize_and_seed,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dataset.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "temporal_baseline_predictions.csv"
)

UPCOMING_EVENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "upcoming_events.csv"
)

API_BASE_URL = os.getenv(
    "HARES AI _API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

import os

API_BASE_URL = os.getenv(
    "HARES AI _API_URL",
    "http://127.0.0.1:8001",
).rstrip("/")

BACKEND_URL = (
    f"{API_BASE_URL}/get_hybrid_risk"
)

WHAT_IF_URL = (
    f"{API_BASE_URL}/get_what_if"
)

EARLY_WARNING_THRESHOLD = 0.40
EARLY_WARNING_DISPLAY_COUNT = 10


# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="AIRA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Authentication / Roles
# ============================================================

initialize_and_seed()
initialize_match_database()
import_previous_matches()


def seed_upcoming_matches_from_csv() -> None:
    """Seed upcoming matches from upcoming_events.csv once.

    Existing match records are never overwritten.
    Missing team/venue information remains explicitly
    'غير مسجل' rather than being invented.
    """

    if not UPCOMING_EVENTS_FILE.exists():
        return

    try:
        events = pd.read_csv(
            UPCOMING_EVENTS_FILE
        )
    except Exception:
        return

    required = {
        "event_id",
        "event_name",
        "event_date",
        "competition",
        "importance",
    }

    if not required.issubset(events.columns):
        return

    existing = {
        row["match_id"]
        for row in get_matches()
    }

    for _, event in events.iterrows():
        event_id = str(
            event["event_id"]
        ).strip()

        if not event_id or event_id in existing:
            continue

        event_date = pd.to_datetime(
            event["event_date"],
            errors="coerce",
        )

        if pd.isna(event_date):
            continue

        competition = str(
            event["competition"]
        ).strip()

        importance = int(
            max(
                1,
                min(
                    5,
                    int(event["importance"]),
                ),
            )
        )

        save_match(
            match_id=event_id,
            match_date=event_date.date().isoformat(),
            status="upcoming",
            competition=competition,
            match_type=str(
                event["event_name"]
            ).strip(),
            team_home="غير مسجل",
            team_away="غير مسجل",
            venue="غير مسجل",
            importance=importance,
            notes=(
                "مباراة قادمة مستوردة من "
                "upcoming_events.csv. "
                "يمكن للمدرب تحديث الخصم "
                "والملعب والتفاصيل."
            ),
            created_by="system",
        )


seed_upcoming_matches_from_csv()


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user" not in st.session_state:
    st.session_state.user = None


def render_login() -> None:
    st.markdown(
        '''
<div class="ag-card">
    <div class="ag-title">⚡ HARES AI </div>
</div>
''',
        unsafe_allow_html=True,
    )

    st.subheader("🔐 تسجيل الدخول")

    login_col1, login_col2 = st.columns([1, 1])

    with login_col1:
        username = st.text_input(
            "اسم المستخدم",
            placeholder="coach أو player",
        )

    with login_col2:
        password = st.text_input(
            "كلمة المرور",
            type="password",
        )

    if st.button(
        "دخول",
        use_container_width=True,
        type="primary",
    ):
        user = authenticate_user(
            username,
            password,
        )

        if user is None:
            st.error(
                "اسم المستخدم أو كلمة المرور غير صحيحة."
            )
            return

        st.session_state.authenticated = True
        st.session_state.user = user
        st.rerun()

    st.info(
        "حسابات Demo:\n"
        "- Coach: coach / coach123\n"
        "- Player: player / player123"
    )


if not st.session_state.authenticated:
    render_login()
    st.stop()

current_user = st.session_state.user

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(14,165,233,0.12),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #0b1120 0%,
            #111827 55%,
            #0f172a 100%
        );
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background: #080d18 !important;
    border-right: 1px solid #1e293b;
}

div[data-testid="stMetric"] {
    background: rgba(30,41,59,0.92);
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-weight: 800 !important;
}

.ag-card {
    background: rgba(30,41,59,0.78);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
}

.ag-title {
    font-size: 36px;
    font-weight: 800;
    color: #f8fafc;
}

.ag-subtitle {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.7;
    margin-top: 8px;
}

.what-if-card {
    background: rgba(30,41,59,0.80);
    border: 1px solid #475569;
    border-radius: 18px;
    padding: 20px;
    margin-top: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_project_data() -> pd.DataFrame:

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = {
        "player_id",
        "name",
        "match_id",
        "minutes_played",
        "rpe",
        "session_load",
        "acute_load",
        "chronic_load",
        "acwr",
        "sleep_duration",
        "sleep_quality",
        "reaction_time_ms",
        "recovery_score",
        "baseline_sleep",
        "baseline_reaction_time",
        "baseline_training_load",
        "sleep_deviation",
        "reaction_time_deviation",
        "recovery_drop",
        "warning_points",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "final_dataset.csv is missing: "
            f"{sorted(missing)}"
        )

    df["player_id"] = (
        df["player_id"]
        .astype(str)
    )

    df["match_number"] = (
        df["match_id"]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .astype(int)
    )

    return (
        df
        .sort_values(
            [
                "player_id",
                "match_number",
            ]
        )
        .reset_index(drop=True)
    )


@st.cache_data
def load_predictions() -> pd.DataFrame:

    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    required = {
        "player_id",
        "match_id",
        "match_number",
        "predicted_probability",
        "predicted_class",
        "injury_soon_target",
        "matches_until_injury",
    }

    missing = required - set(
        predictions.columns
    )

    if missing:
        raise ValueError(
            "temporal_baseline_predictions.csv is missing: "
            f"{sorted(missing)}"
        )

    predictions["player_id"] = (
        predictions["player_id"]
        .astype(str)
    )

    predictions["match_number"] = (
        predictions["match_number"]
        .astype(int)
    )

    predictions["predicted_probability"] = (
        predictions["predicted_probability"]
        .astype(float)
        .clip(0.0, 1.0)
    )

    return (
        predictions
        .sort_values(
            [
                "player_id",
                "match_number",
            ]
        )
        .reset_index(drop=True)
    )


@st.cache_data
def load_upcoming_events() -> pd.DataFrame:

    if not UPCOMING_EVENTS_FILE.exists():
        return pd.DataFrame()

    events = pd.read_csv(
        UPCOMING_EVENTS_FILE
    )

    required = {
        "event_id",
        "event_name",
        "event_date",
        "competition",
        "importance",
    }

    missing = required - set(
        events.columns
    )

    if missing:
        raise ValueError(
            "upcoming_events.csv is missing: "
            f"{sorted(missing)}"
        )

    events["event_date"] = pd.to_datetime(
        events["event_date"],
        errors="coerce",
    )

    events["importance"] = pd.to_numeric(
        events["importance"],
        errors="coerce",
    )

    return (
        events
        .dropna(
            subset=[
                "event_date",
                "importance",
            ]
        )
        .sort_values("event_date")
        .reset_index(drop=True)
    )


# ============================================================
# API helpers
# ============================================================

def convert_player_id(
    player_id: object,
) -> int:

    text = str(
        player_id
    ).strip()

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    if not digits:
        raise ValueError(
            f"Invalid player ID: {player_id}"
        )

    return int(digits)


def build_hybrid_payload(
    row: pd.Series,
) -> dict:

    return {
        "player_id": convert_player_id(
            row["player_id"]
        ),
        "training_duration_min": float(
            row["minutes_played"]
        ),
        "rpe": float(
            row["rpe"]
        ),
        "sleep_duration": float(
            row["sleep_duration"]
        ),
        "sleep_quality": float(
            row["sleep_quality"]
        ),
        "reaction_time_ms": float(
            row["reaction_time_ms"]
        ),
        "recovery_score": float(
            row["recovery_score"]
        ),
        "baseline_sleep": float(
            row["baseline_sleep"]
        ),
        "baseline_reaction_time_ms": float(
            row["baseline_reaction_time"]
        ),
        "baseline_training_load": float(
            row["baseline_training_load"]
        ),
        "acute_load": float(
            row["acute_load"]
        ),
        "chronic_load": float(
            row["chronic_load"]
        ),
        "injury_context": False,
        "session_load": float(
            row["session_load"]
        ),
        "acwr": float(
            row["acwr"]
        ),
        "sleep_deviation": float(
            row["sleep_deviation"]
        ),
        "reaction_time_deviation": float(
            row["reaction_time_deviation"]
        ),
        "recovery_drop": float(
            row["recovery_drop"]
        ),
        "warning_points": float(
            row["warning_points"]
        ),
    }


def post_json(
    url: str,
    payload: dict,
) -> dict:

    request = Request(
        url,
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type": (
                "application/json"
            )
        },
        method="POST",
    )

    try:

        with urlopen(
            request,
            timeout=10,
        ) as response:

            return json.loads(
                response
                .read()
                .decode("utf-8")
            )

    except HTTPError as exc:

        detail = (
            exc
            .read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            f"Backend error {exc.code}: {detail}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            "تعذر الاتصال بالـBackend. "
            "تأكدي أن FastAPI يعمل على "
            "http://127.0.0.1:8001"
        ) from exc


def call_hybrid_api(
    row: pd.Series,
) -> dict:

    return post_json(
        BACKEND_URL,
        build_hybrid_payload(row),
    )


# ============================================================
# Risk helpers
# ============================================================

def risk_label(
    risk: str,
) -> str:

    return {
        "low": "🟢 منخفض",
        "moderate": "🟡 متوسط",
        "elevated": "🚨 مرتفع",
    }.get(
        risk,
        str(risk),
    )


def probability_to_risk(
    probability: float,
) -> str:

    if probability >= 0.70:
        return "elevated"

    if probability >= 0.40:
        return "moderate"

    return "low"


def match_score_text(
    match: dict,
) -> str:
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
        return "—"

    return (
        f"{home_score} - "
        f"{away_score}"
    )


def match_date_text(
    match: dict,
) -> str:
    value = match.get(
        "match_date"
    )

    if not value:
        return "غير مسجل"

    try:
        return pd.Timestamp(
            value
        ).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def upcoming_match_event_result(
    match: dict,
    hybrid_result: dict,
) -> dict | None:
    date_value = match.get(
        "match_date"
    )

    if not date_value:
        return None

    try:
        date_value = pd.Timestamp(
            date_value
        ).date()
    except Exception:
        return None

    return calculate_event_readiness(
        event_name=str(
            match.get(
                "match_type",
                "Upcoming Match",
            )
        ),
        event_date=date_value,
        event_importance=int(
            match.get(
                "importance",
                3,
            )
        ),
        hybrid_result=hybrid_result,
    )


# ============================================================
# Scenario builder
# ============================================================

def build_scenario_changes(
    row: pd.Series,
    scenario: str,
) -> dict:

    current_minutes = float(
        row["minutes_played"]
    )

    current_rpe = float(
        row["rpe"]
    )

    current_sleep = float(
        row["sleep_duration"]
    )

    current_sleep_quality = float(
        row["sleep_quality"]
    )

    current_recovery = float(
        row["recovery_score"]
    )

    current_acute = float(
        row["acute_load"]
    )

    current_chronic = float(
        row["chronic_load"]
    )

    current_reaction = float(
        row["reaction_time_ms"]
    )

    current_sleep_baseline = float(
        row["baseline_sleep"]
    )

    current_reaction_baseline = float(
        row["baseline_reaction_time"]
    )

    # --------------------------------------------------------
    # 10% lower training load
    # --------------------------------------------------------

    if scenario in {"خفض الجهد التدريبي 10%", "خفض الجهد التدريبي 10%"}:

        new_minutes = (
            current_minutes * 0.90
        )

        new_session_load = (
            new_minutes
            * current_rpe
        )

        new_acute = (
            current_acute * 0.95
        )

        new_acwr = (
            new_acute
            / max(current_chronic, 1.0)
        )

        return {
            "training_duration_min":
                round(new_minutes, 2),
            "session_load":
                round(new_session_load, 2),
            "acute_load":
                round(new_acute, 2),
            "acwr":
                round(new_acwr, 3),
            "warning_points":
                max(
                    0.0,
                    float(
                        row["warning_points"]
                    ) - 1.0,
                ),
        }

    # --------------------------------------------------------
    # 20% lower training load
    # --------------------------------------------------------

    if scenario in {"خفض الجهد التدريبي 20%", "خفض الجهد التدريبي 20%"}:

        new_minutes = (
            current_minutes * 0.80
        )

        new_session_load = (
            new_minutes
            * current_rpe
        )

        new_acute = (
            current_acute * 0.90
        )

        new_acwr = (
            new_acute
            / max(current_chronic, 1.0)
        )

        return {
            "training_duration_min":
                round(new_minutes, 2),
            "session_load":
                round(new_session_load, 2),
            "acute_load":
                round(new_acute, 2),
            "acwr":
                round(new_acwr, 3),
            "warning_points":
                max(
                    0.0,
                    float(
                        row["warning_points"]
                    ) - 2.0,
                ),
        }

    # --------------------------------------------------------
    # Recovery day
    # --------------------------------------------------------

    if scenario == "يوم استشفاء":

        new_sleep = min(
            24.0,
            current_sleep + 0.75,
        )

        new_sleep_quality = min(
            100.0,
            current_sleep_quality + 12.0,
        )

        new_recovery = min(
            100.0,
            current_recovery + 12.0,
        )

        new_reaction = max(
            150.0,
            current_reaction - 7.0,
        )

        new_sleep_deviation = (
            new_sleep
            - current_sleep_baseline
        )

        new_reaction_deviation = (
            new_reaction
            - current_reaction_baseline
        )

        new_minutes = max(
            20.0,
            current_minutes * 0.35,
        )

        new_rpe = min(
            current_rpe,
            3.0,
        )

        new_session_load = (
            new_minutes
            * new_rpe
        )

        new_acute = (
            current_acute * 0.75
        )

        new_acwr = (
            new_acute
            / max(current_chronic, 1.0)
        )

        return {
            "training_duration_min":
                round(new_minutes, 2),
            "rpe":
                round(new_rpe, 2),
            "sleep_duration":
                round(new_sleep, 2),
            "sleep_quality":
                round(new_sleep_quality, 2),
            "recovery_score":
                round(new_recovery, 2),
            "reaction_time_ms":
                round(new_reaction, 2),
            "session_load":
                round(new_session_load, 2),
            "acute_load":
                round(new_acute, 2),
            "acwr":
                round(new_acwr, 3),
            "sleep_deviation":
                round(new_sleep_deviation, 3),
            "reaction_time_deviation":
                round(new_reaction_deviation, 3),
            "recovery_drop":
                0.0,
            "warning_points":
                0.0,
        }

    # --------------------------------------------------------
    # +1 hour sleep
    # --------------------------------------------------------

    if scenario == "زيادة النوم ساعة":

        new_sleep = min(
            24.0,
            current_sleep + 1.0,
        )

        new_sleep_quality = min(
            100.0,
            current_sleep_quality + 8.0,
        )

        new_recovery = min(
            100.0,
            current_recovery + 6.0,
        )

        new_reaction = max(
            150.0,
            current_reaction - 3.0,
        )

        return {
            "sleep_duration":
                round(new_sleep, 2),
            "sleep_quality":
                round(new_sleep_quality, 2),
            "recovery_score":
                round(new_recovery, 2),
            "reaction_time_ms":
                round(new_reaction, 2),
            "sleep_deviation":
                round(
                    new_sleep
                    - current_sleep_baseline,
                    3,
                ),
            "reaction_time_deviation":
                round(
                    new_reaction
                    - current_reaction_baseline,
                    3,
                ),
            "recovery_drop":
                round(
                    new_recovery
                    - current_recovery,
                    3,
                ),
        }

    return {}


# ============================================================
# What-If API
# ============================================================

def run_what_if(
    row: pd.Series,
    changes: dict,
) -> dict:

    payload = build_hybrid_payload(
        row
    )

    payload["changes"] = changes

    return post_json(
        WHAT_IF_URL,
        payload,
    )


# ============================================================
# Load
# ============================================================

try:

    df = load_project_data()
    predictions = load_predictions()
    upcoming_events = load_upcoming_events()

except Exception as exc:

    st.error(
        f"تعذر تحميل البيانات: {exc}"
    )

    st.stop()


# ============================================================
# Header
# ============================================================

# تم تعديل هذا الجزء فقط لمنع ظهور HTML كـ code block.
st.markdown(
    """
<div class="ag-card">
    <div class="ag-title">⚡ HARES AI </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
**Past → Current → Next Event**  
Hybrid AI for athlete readiness and early-risk decision support
"""
)


if current_user["role"] == "coach":
    st.success(
        "👨‍🏫 Coach Mode — عرض وإدارة الفريق"
    )
else:
    st.info(
        "🏃 Player Mode — عرض بياناتك الشخصية فقط"
    )


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title(
    "⚡ AIRA"
)

st.sidebar.caption(
    "Readiness + Temporal ML + Event Readiness"
)

st.sidebar.divider()

st.sidebar.write(
    f"👤 **{current_user['full_name']}**"
)

st.sidebar.caption(
    f"Role: {current_user['role'].upper()}"
)

if st.sidebar.button(
    "تسجيل الخروج",
    use_container_width=True,
):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

st.sidebar.divider()

players = (
    df[
        [
            "player_id",
            "name",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "player_id"
    )
)

player_options = {
    f"{row.player_id} — {row.name}":
        row.player_id
    for row in players.itertuples()
}

# ------------------------------------------------------------
# Role-based player access
# ------------------------------------------------------------

if current_user["role"] == "player":
    selected_player = current_user["player_id"]

    if not selected_player:
        st.error(
            "هذا الحساب غير مرتبط بملف لاعب."
        )
        st.stop()

    player_record = get_player(
        selected_player
    )

    if player_record is None:
        st.error(
            "ملف اللاعب غير موجود في قاعدة البيانات."
        )
        st.stop()

    st.sidebar.info(
        "🔒 حساب اللاعب يعرض ملفك الشخصي فقط."
    )

else:
    selected_player_label = (
        st.sidebar.selectbox(
            "👥 اختر اللاعب",
            list(
                player_options.keys()
            ),
        )
    )

    selected_player = (
        player_options[
            selected_player_label
        ]
    )

player_history = (
    df[
        df["player_id"]
        == selected_player
    ]
    .sort_values(
        "match_number"
    )
    .reset_index(drop=True)
)

if player_history.empty:

    st.error(
        "لا توجد بيانات لهذا اللاعب."
    )

    st.stop()

match_options = [
    f"{row.match_id} — Match {row.match_number}"
    for row in (
        player_history
        .sort_values(
            "match_number",
            ascending=False,
        )
        .itertuples()
    )
]

selected_match_label = (
    st.sidebar.selectbox(
        "🏟️ اختر المباراة",
        match_options,
    )
)

selected_match_id = (
    selected_match_label
    .split(" — ")[0]
)

selected_row = (
    player_history[
        player_history["match_id"]
        == selected_match_id
    ]
    .iloc[0]
)
# ============================================================
# Real Athlete Data Entry — Role Based Permissions
# ============================================================

st.divider()

st.subheader(
    "📝 إدخال البيانات الحالية"
)

if current_user["role"] == "player":

    st.caption(
        "أدخل بياناتك الشخصية اليومية. "
        "هذه البيانات تُحفظ في قاعدة البيانات وتُستخدم في التحليل."
    )

else:

    st.caption(
        "أدخل بيانات التدريب أو المباراة. "
        "بيانات اللاعب الشخصية مثل النوم والتعافي يسجلها اللاعب."
    )


with st.expander(
    "➕ إضافة قياسات جديدة",
    expanded=False,
):

    if current_user["role"] == "player":

        # --------------------------------------------------------
        # Player permissions
        # --------------------------------------------------------

        player_input_col1, player_input_col2 = (
            st.columns(2)
        )

        with player_input_col1:

            real_sleep_duration = st.number_input(
                "مدة النوم (ساعة)",
                min_value=0.0,
                max_value=24.0,
                value=float(
                    selected_row["sleep_duration"]
                ),
                step=0.1,
            )

            real_sleep_quality = st.number_input(
                "جودة النوم",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    selected_row["sleep_quality"]
                ),
                step=1.0,
            )

        with player_input_col2:

            real_recovery = st.number_input(
                "درجة التعافي",
                min_value=0.0,
                max_value=100.0,
                value=float(
                    selected_row["recovery_score"]
                ),
                step=1.0,
            )

            real_reaction_time = st.number_input(
                "زمن الاستجابة (ms)",
                min_value=50.0,
                max_value=1000.0,
                value=float(
                    selected_row["reaction_time_ms"]
                ),
                step=1.0,
            )

        real_match_id = str(
            selected_row["match_id"]
        )

        real_date = st.date_input(
            "تاريخ القياس",
        )

        st.info(
            "🏃 بيانات اللاعب: النوم، جودة النوم، "
            "التعافي، وزمن الاستجابة."
        )

    else:

        # --------------------------------------------------------
        # Coach permissions
        # --------------------------------------------------------

        coach_input_col1, coach_input_col2 = (
            st.columns(2)
        )

        with coach_input_col1:

            real_training_duration = st.number_input(
                "مدة التدريب (دقيقة)",
                min_value=1.0,
                max_value=300.0,
                value=float(
                    selected_row["minutes_played"]
                ),
                step=1.0,
            )

            real_rpe = st.number_input(
                "شدة الجهد RPE",
                min_value=0.0,
                max_value=10.0,
                value=float(
                    selected_row["rpe"]
                ),
                step=0.1,
            )

        with coach_input_col2:

            real_match_id = st.text_input(
                "معرف المباراة / الحصة",
                value=str(
                    selected_row["match_id"]
                ),
            )

            real_date = st.date_input(
                "تاريخ القياس",
            )

        # Player-only values come from the latest stored
        # personal observation, or from the selected dataset row.
        latest_personal_observation = (
            get_latest_observation(
                str(selected_player)
            )
        )

        if latest_personal_observation:
            real_sleep_duration = float(
                latest_personal_observation.get(
                    "sleep_duration",
                    selected_row["sleep_duration"],
                )
            )
            real_sleep_quality = float(
                latest_personal_observation.get(
                    "sleep_quality",
                    selected_row["sleep_quality"],
                )
            )
            real_recovery = float(
                latest_personal_observation.get(
                    "recovery_score",
                    selected_row["recovery_score"],
                )
            )
            real_reaction_time = float(
                latest_personal_observation.get(
                    "reaction_time_ms",
                    selected_row["reaction_time_ms"],
                )
            )
        else:
            real_sleep_duration = float(
                selected_row["sleep_duration"]
            )
            real_sleep_quality = float(
                selected_row["sleep_quality"]
            )
            real_recovery = float(
                selected_row["recovery_score"]
            )
            real_reaction_time = float(
                selected_row["reaction_time_ms"]
            )

        st.info(
            "👨‍🏫 بيانات المدرب: مدة التدريب، شدة الجهد، "
            "ومعرف المباراة أو الحصة."
        )

    if st.button(
        "💾 حفظ البيانات وتحليلها",
        use_container_width=True,
        type="primary",
    ):

        # --------------------------------------------------------
        # Base values used by the calculation
        # --------------------------------------------------------

        if current_user["role"] == "player":

            training_duration_for_save = float(
                selected_row["minutes_played"]
            )

            rpe_for_save = float(
                selected_row["rpe"]
            )

        else:

            training_duration_for_save = float(
                real_training_duration
            )

            rpe_for_save = float(
                real_rpe
            )

        session_load = (
            training_duration_for_save
            * rpe_for_save
        )

        baseline_training_load = float(
            selected_row[
                "baseline_training_load"
            ]
        )

        baseline_sleep = float(
            selected_row[
                "baseline_sleep"
            ]
        )

        baseline_reaction = float(
            selected_row[
                "baseline_reaction_time"
            ]
        )

        chronic_load = max(
            baseline_training_load,
            1.0,
        )

        acute_load = session_load

        acwr = (
            acute_load
            / chronic_load
        )

        sleep_deviation = (
            real_sleep_duration
            - baseline_sleep
        )

        reaction_deviation = (
            real_reaction_time
            - baseline_reaction
        )

        recovery_drop = (
            real_recovery
            - float(
                selected_row[
                    "recovery_score"
                ]
            )
        )

        warning_points = 0.0

        if sleep_deviation < -1.0:
            warning_points += 1

        if reaction_deviation > 10:
            warning_points += 1

        if recovery_drop < -10:
            warning_points += 1

        if acwr >= 1.30:
            warning_points += 1

        try:

            observation_id = save_observation(
                player_id=str(
                    selected_player
                ),
                match_id=real_match_id,
                observation_date=str(
                    real_date
                ),
                training_duration_min=(
                    training_duration_for_save
                ),
                rpe=rpe_for_save,
                sleep_duration=(
                    real_sleep_duration
                ),
                sleep_quality=(
                    real_sleep_quality
                ),
                reaction_time_ms=(
                    real_reaction_time
                ),
                recovery_score=(
                    real_recovery
                ),
                session_load=session_load,
                acute_load=acute_load,
                chronic_load=chronic_load,
                acwr=acwr,
                sleep_deviation=(
                    sleep_deviation
                ),
                reaction_time_deviation=(
                    reaction_deviation
                ),
                recovery_drop=(
                    recovery_drop
                ),
                warning_points=(
                    warning_points
                ),
                source=current_user[
                    "role"
                ],
            )

            st.success(
                f"✅ تم حفظ البيانات بنجاح "
                f"(Record #{observation_id})."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"تعذر حفظ البيانات: {exc}"
            )


# ============================================================
# Real Data Helpers
# ============================================================

def apply_real_observation(
    base_row: pd.Series,
    observation: dict,
) -> pd.Series:
    """
    Merge the latest real/manual observation from SQLite
    into the selected player's baseline dataset row.
    """

    row = base_row.copy()

    mapping = {
        "minutes_played": "training_duration_min",
        "rpe": "rpe",
        "sleep_duration": "sleep_duration",
        "sleep_quality": "sleep_quality",
        "reaction_time_ms": "reaction_time_ms",
        "recovery_score": "recovery_score",
        "session_load": "session_load",
        "acute_load": "acute_load",
        "chronic_load": "chronic_load",
        "acwr": "acwr",
        "sleep_deviation": "sleep_deviation",
        "reaction_time_deviation":
            "reaction_time_deviation",
        "recovery_drop": "recovery_drop",
        "warning_points": "warning_points",
    }

    for csv_column, db_column in mapping.items():

        value = observation.get(
            db_column
        )

        if value is not None:
            row[csv_column] = value

    return row


# ============================================================
# Step 4 — Use latest real observation in the AI analysis
# ============================================================

latest_real_observation = get_latest_observation(
    str(selected_player)
)

if latest_real_observation is not None:

    selected_row = apply_real_observation(
        selected_row,
        latest_real_observation,
    )

    st.info(
        "🔄 يستخدم النظام أحدث بيانات فعلية "
        "محفوظة في قاعدة البيانات لهذا اللاعب."
    )


# ============================================================
# Current Hybrid analysis
# ============================================================

try:

    with st.spinner(
        "جاري تشغيل Hybrid AI..."
    ):

        analysis = call_hybrid_api(
            selected_row
        )

except Exception as exc:

    st.error(
        str(exc)
    )

    st.stop()


final_risk = analysis[
    "final_risk_level"
]

readiness_score = float(
    analysis["readiness_score"]
)

readiness_risk = analysis[
    "readiness_risk_level"
]

ml_risk = analysis[
    "ml_risk_level"
]

ml_probability = float(
    analysis["ml_injury_probability"]
)

recommendation = analysis[
    "recommendation"
]

factors = analysis.get(
    "factors",
    [],
)

metrics = analysis.get(
    "metrics",
    {},
)

subscores = analysis.get(
    "subscores",
    {},
)


# ============================================================
# Current team status
# ============================================================

st.subheader(
    "📊 حالة الفريق الحالية"
)

latest_rows = (
    df
    .sort_values(
        "match_number"
    )
    .groupby(
        "player_id",
        as_index=False,
    )
    .tail(1)
)

team_results = []

for _, row in latest_rows.iterrows():

    try:

        result = call_hybrid_api(
            row
        )

        team_results.append(
            {
                "player_id":
                    row["player_id"],
                "name":
                    row["name"],
                "readiness":
                    float(
                        result[
                            "readiness_score"
                        ]
                    ),
                "risk":
                    result[
                        "final_risk_level"
                    ],
            }
        )

    except Exception:
        continue


team_results_df = pd.DataFrame(
    team_results
)

if team_results_df.empty:

    high_count = 0
    moderate_count = 0
    low_count = 0
    average_score = 0.0

else:

    high_count = int(
        (
            team_results_df["risk"]
            == "elevated"
        ).sum()
    )

    moderate_count = int(
        (
            team_results_df["risk"]
            == "moderate"
        ).sum()
    )

    low_count = int(
        (
            team_results_df["risk"]
            == "low"
        ).sum()
    )

    average_score = float(
        team_results_df[
            "readiness"
        ].mean()
    )


team_cols = st.columns(4)

with team_cols[0]:

    st.metric(
        "🚨 خطر مرتفع الآن",
        high_count,
    )

with team_cols[1]:

    st.metric(
        "🟡 خطر متوسط الآن",
        moderate_count,
    )

with team_cols[2]:

    st.metric(
        "🟢 خطر منخفض الآن",
        low_count,
    )

with team_cols[3]:

    st.metric(
        "📊 متوسط الجاهزية الحالية",
        f"{average_score:.1f}",
    )


# ============================================================
# Smart Alerts
# ============================================================

def build_smart_alerts(
    analysis: dict,
    selected_row: pd.Series,
    trajectory: pd.DataFrame,
    current_event_result: dict | None,
) -> list[dict[str, str]]:
    """
    Generate explainable alerts from the current Hybrid AI,
    Digital Twin deviations, recent risk trajectory, and
    the next event.

    Alerts are decision-support messages, not medical clearance.
    """

    alerts: list[dict[str, str]] = []

    final_risk = str(
        analysis.get(
            "final_risk_level",
            "low",
        )
    )

    readiness_score = float(
        analysis.get(
            "readiness_score",
            0.0,
        )
    )

    ml_probability = float(
        analysis.get(
            "ml_injury_probability",
            0.0,
        )
    )

    sleep_deviation = float(
        selected_row.get(
            "sleep_deviation",
            0.0,
        )
    )

    recovery_drop = float(
        selected_row.get(
            "recovery_drop",
            0.0,
        )
    )

    reaction_deviation = float(
        selected_row.get(
            "reaction_time_deviation",
            0.0,
        )
    )

    training_deviation = float(
        selected_row.get(
            "acute_load",
            0.0,
        ) - selected_row.get(
            "baseline_training_load",
            0.0,
        )
    )

    warning_points = float(
        selected_row.get(
            "warning_points",
            0.0,
        )
    )

    # --------------------------------------------------------
    # Current final risk
    # --------------------------------------------------------

    if final_risk == "elevated":

        alerts.append(
            {
                "level": "critical",
                "title": "🚨 خطر مرتفع",
                "message": (
                    "الخطر النهائي مرتفع حاليًا. "
                    "يُنصح بإعادة تقييم الحالة قبل "
                    "المشاركة الكاملة."
                ),
            }
        )

    elif final_risk == "moderate":

        alerts.append(
            {
                "level": "warning",
                "title": "🟡 خطر متوسط",
                "message": (
                    "توجد مؤشرات خطر متوسطة. "
                    "يُنصح بمراقبة الحالة وتعديل "
                    "الجهد التدريبي عند الحاجة."
                ),
            }
        )

    # --------------------------------------------------------
    # Readiness
    # --------------------------------------------------------

    if readiness_score < 60:

        alerts.append(
            {
                "level": "critical",
                "title": "⚠️ جاهزية منخفضة",
                "message": (
                    f"جاهزية اللاعب الحالية "
                    f"{readiness_score:.1f}/100."
                ),
            }
        )

    elif readiness_score < 75:

        alerts.append(
            {
                "level": "warning",
                "title": "🟡 الجاهزية تحتاج متابعة",
                "message": (
                    f"جاهزية اللاعب "
                    f"{readiness_score:.1f}/100 "
                    "وتحتاج إلى متابعة قبل الحدث القادم."
                ),
            }
        )

    # --------------------------------------------------------
    # Sleep deviation
    # --------------------------------------------------------

    if sleep_deviation <= -1.0:

        alerts.append(
            {
                "level": "warning",
                "title": "😴 انخفاض النوم",
                "message": (
                    f"النوم أقل من خط الأساس "
                    f"بمقدار {abs(sleep_deviation):.1f} ساعة."
                ),
            }
        )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    if recovery_drop <= -10:

        alerts.append(
            {
                "level": "warning",
                "title": "🔋 انخفاض التعافي",
                "message": (
                    f"التعافي انخفض بنحو "
                    f"{abs(recovery_drop):.1f} نقطة "
                    "عن السجل المرجعي المستخدم."
                ),
            }
        )

    # --------------------------------------------------------
    # Reaction time
    # --------------------------------------------------------

    if reaction_deviation >= 10:

        alerts.append(
            {
                "level": "warning",
                "title": "⚡ تباطؤ الاستجابة",
                "message": (
                    f"زمن الاستجابة أعلى من خط الأساس "
                    f"بـ{reaction_deviation:.1f} ms."
                ),
            }
        )

    # --------------------------------------------------------
    # Training deviation
    # --------------------------------------------------------

    if training_deviation >= 30:

        alerts.append(
            {
                "level": "warning",
                "title": "📈 ارتفاع الجهد التدريبي",
                "message": (
                    f"الجهد التدريبي الحديث أعلى من "
                    f"خط الأساس بنحو {training_deviation:.1f}."
                ),
            }
        )

    # --------------------------------------------------------
    # Multiple warning indicators
    # --------------------------------------------------------

    if warning_points >= 3:

        alerts.append(
            {
                "level": "critical",
                "title": "🚨 تراكم مؤشرات التحذير",
                "message": (
                    f"تم رصد {warning_points:.0f} "
                    "مؤشرات تحذير في السجل الحالي."
                ),
            }
        )

    elif warning_points >= 2:

        alerts.append(
            {
                "level": "warning",
                "title": "🟡 عدة مؤشرات تحتاج متابعة",
                "message": (
                    f"تم رصد {warning_points:.0f} "
                    "مؤشرات تحذير."
                ),
            }
        )

    # --------------------------------------------------------
    # ML probability
    # --------------------------------------------------------

    if ml_probability >= 0.70:

        alerts.append(
            {
                "level": "critical",
                "title": "🤖 ML Risk مرتفع",
                "message": (
                    f"احتمال الخطر النموذجي "
                    f"{ml_probability * 100:.1f}%."
                ),
            }
        )

    elif ml_probability >= 0.40:

        alerts.append(
            {
                "level": "warning",
                "title": "🤖 ML Risk متوسط",
                "message": (
                    f"احتمال الخطر النموذجي "
                    f"{ml_probability * 100:.1f}%."
                ),
            }
        )

    # --------------------------------------------------------
    # Risk trajectory
    # --------------------------------------------------------

    if (
        not trajectory.empty
        and "predicted_probability"
        in trajectory.columns
    ):

        recent = (
            trajectory
            .sort_values("match_number")
            .tail(3)
        )

        if len(recent) >= 2:

            start_probability = float(
                recent.iloc[0][
                    "predicted_probability"
                ]
            )

            end_probability = float(
                recent.iloc[-1][
                    "predicted_probability"
                ]
            )

            change = (
                end_probability
                - start_probability
            )

            if change >= 0.20:

                alerts.append(
                    {
                        "level": "critical",
                        "title": "📈 تصاعد سريع في الخطر",
                        "message": (
                            "مسار الخطر يرتفع بوضوح "
                            "خلال آخر السجلات."
                        ),
                    }
                )

            elif change >= 0.05:

                alerts.append(
                    {
                        "level": "warning",
                        "title": "📈 الخطر في اتجاه تصاعدي",
                        "message": (
                            "هناك ارتفاع حديث في مسار "
                            "الخطر ويستحسن إعادة التقييم."
                        ),
                    }
                )

    # --------------------------------------------------------
    # Upcoming event
    # --------------------------------------------------------

    if current_event_result is not None:

        event_score = float(
            current_event_result.get(
                "event_readiness_score",
                readiness_score,
            )
        )

        if event_score < 60:

            alerts.append(
                {
                    "level": "critical",
                    "title": "🏆 جاهزية الحدث منخفضة",
                    "message": (
                        f"جاهزية الحدث القادم "
                        f"{event_score:.1f}/100."
                    ),
                }
            )

        elif event_score < 75:

            alerts.append(
                {
                    "level": "warning",
                    "title": "🏆 الحدث القادم يحتاج متابعة",
                    "message": (
                        f"جاهزية الحدث القادم "
                        f"{event_score:.1f}/100."
                    ),
                }
            )

    return alerts


def render_smart_alerts(
    alerts: list[dict[str, str]],
    role: str,
) -> None:
    """
    Display Smart Alerts.

    Coach sees the decision-support alerts explicitly.
    Player sees personal alerts only.
    """

    st.divider()

    st.subheader(
        "🔔 التنبيهات الذكية"
    )

    if not alerts:

        st.success(
            "🟢 لا توجد تنبيهات ذكية رئيسية في السجل الحالي."
        )

        return

    if role == "coach":

        st.caption(
            "تنبيهات تساعد المدرب على متابعة "
            "الحالة واتخاذ قرار إعادة التقييم."
        )

    else:

        st.caption(
            "تنبيهات شخصية تساعدك على متابعة "
            "جاهزيتك وتعافيك."
        )

    priority = {
        "critical": 0,
        "warning": 1,
        "info": 2,
    }

    alerts = sorted(
        alerts,
        key=lambda item: priority.get(
            item.get("level", "info"),
            3,
        ),
    )

    for alert in alerts:

        level = alert.get(
            "level",
            "info",
        )

        title = alert.get(
            "title",
            "تنبيه",
        )

        message = alert.get(
            "message",
            "",
        )

        if level == "critical":

            st.error(
                f"**{title}**\n\n"
                f"{message}"
            )

        elif level == "warning":

            st.warning(
                f"**{title}**\n\n"
                f"{message}"
            )

        else:

            st.info(
                f"**{title}**\n\n"
                f"{message}"
            )


# ============================================================
# Matches — Previous & Upcoming
# ============================================================

st.divider()

st.subheader(
    "🏟️ المباريات"
)

st.caption(
    "سجل المباريات السابقة والقادمة مع تفاصيل "
    "المسابقة والنتيجة والملعب والأهمية."
)

all_matches = get_matches()

past_matches = [
    match
    for match in all_matches
    if match.get("status") == "past"
]

upcoming_matches = [
    match
    for match in all_matches
    if match.get("status") == "upcoming"
]

past_tab, upcoming_tab = st.tabs(
    [
        "📚 المباريات السابقة",
        "📅 المباريات القادمة",
    ]
)

with past_tab:

    if not past_matches:

        st.info(
            "لا توجد مباريات سابقة مسجلة."
        )

    else:

        past_rows = []

        for match in past_matches:

            past_rows.append(
                {
                    "ID":
                        match["match_id"],
                    "التاريخ":
                        match_date_text(match),
                    "المسابقة":
                        match["competition"],
                    "نوع المباراة":
                        match["match_type"],
                    "المضيف":
                        match["team_home"],
                    "الضيف":
                        match["team_away"],
                    "النتيجة":
                        match_score_text(match),
                    "الملعب":
                        match["venue"]
                        or "غير مسجل",
                    "الأهمية":
                        match["importance"],
                }
            )

        st.dataframe(
            pd.DataFrame(
                past_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

with upcoming_tab:

    if current_user["role"] == "coach":

        # --------------------------------------------------------
        # Edit any existing match
        # --------------------------------------------------------

        all_match_ids = [
            match["match_id"]
            for match in all_matches
        ]

        if all_match_ids:

            with st.expander(
                "✏️ تعديل بيانات مباراة",
                expanded=False,
            ):

                selected_edit_id = st.selectbox(
                    "اختر المباراة",
                    all_match_ids,
                    key="edit_match_id",
                )

                selected_edit_match = next(
                    (
                        match
                        for match in all_matches
                        if match["match_id"]
                        == selected_edit_id
                    ),
                    None,
                )

                if selected_edit_match is not None:

                    edit_col1, edit_col2 = (
                        st.columns(2)
                    )

                    with edit_col1:

                        edit_date_value = (
                            pd.to_datetime(
                                selected_edit_match[
                                    "match_date"
                                ],
                                errors="coerce",
                            )
                        )

                        if pd.isna(edit_date_value):
                            edit_date_value = (
                                pd.Timestamp.now()
                                .date()
                            )
                        else:
                            edit_date_value = (
                                edit_date_value.date()
                            )

                        edit_match_date = st.date_input(
                            "تاريخ المباراة",
                            value=edit_date_value,
                            key="edit_match_date",
                        )

                        edit_status = st.selectbox(
                            "حالة المباراة",
                            [
                                "past",
                                "upcoming",
                            ],
                            index=(
                                0
                                if selected_edit_match[
                                    "status"
                                ]
                                == "past"
                                else 1
                            ),
                            format_func=lambda value: (
                                "سابقة"
                                if value == "past"
                                else "قادمة"
                            ),
                            key="edit_match_status",
                        )

                        edit_competition = st.text_input(
                            "المسابقة / البطولة",
                            value=(
                                ""
                                if selected_edit_match[
                                    "competition"
                                ]
                                == "غير مسجل"
                                else str(
                                    selected_edit_match[
                                        "competition"
                                    ]
                                )
                            ),
                            key="edit_competition",
                        )

                        edit_match_type = st.text_input(
                            "نوع المباراة",
                            value=str(
                                selected_edit_match[
                                    "match_type"
                                ]
                            ),
                            key="edit_match_type",
                        )

                    with edit_col2:

                        edit_team_home = st.text_input(
                            "الفريق المضيف",
                            value=(
                                ""
                                if selected_edit_match[
                                    "team_home"
                                ]
                                == "غير مسجل"
                                else str(
                                    selected_edit_match[
                                        "team_home"
                                    ]
                                )
                            ),
                            key="edit_team_home",
                        )

                        edit_team_away = st.text_input(
                            "الفريق الضيف",
                            value=(
                                ""
                                if selected_edit_match[
                                    "team_away"
                                ]
                                == "غير مسجل"
                                else str(
                                    selected_edit_match[
                                        "team_away"
                                    ]
                                )
                            ),
                            key="edit_team_away",
                        )

                        edit_venue = st.text_input(
                            "الملعب",
                            value=(
                                ""
                                if selected_edit_match[
                                    "venue"
                                ]
                                in (None, "غير مسجل")
                                else str(
                                    selected_edit_match[
                                        "venue"
                                    ]
                                )
                            ),
                            key="edit_venue",
                        )

                        edit_importance = st.slider(
                            "أهمية المباراة",
                            min_value=1,
                            max_value=5,
                            value=int(
                                selected_edit_match[
                                    "importance"
                                ]
                            ),
                            key="edit_importance",
                        )

                    score_col1, score_col2 = (
                        st.columns(2)
                    )

                    with score_col1:

                        edit_home_score = st.number_input(
                            "نتيجة الفريق المضيف",
                            min_value=0,
                            max_value=100,
                            value=int(
                                selected_edit_match[
                                    "home_score"
                                ]
                                if selected_edit_match[
                                    "home_score"
                                ] is not None
                                else 0
                            ),
                            disabled=(
                                edit_status
                                == "upcoming"
                            ),
                            key="edit_home_score",
                        )

                    with score_col2:

                        edit_away_score = st.number_input(
                            "نتيجة الفريق الضيف",
                            min_value=0,
                            max_value=100,
                            value=int(
                                selected_edit_match[
                                    "away_score"
                                ]
                                if selected_edit_match[
                                    "away_score"
                                ] is not None
                                else 0
                            ),
                            disabled=(
                                edit_status
                                == "upcoming"
                            ),
                            key="edit_away_score",
                        )

                    if st.button(
                        "💾 حفظ تعديلات المباراة",
                        key="save_match_edits",
                        use_container_width=True,
                        type="primary",
                    ):

                        try:

                            if not edit_team_home.strip():
                                raise ValueError(
                                    "يجب إدخال الفريق المضيف."
                                )

                            if not edit_team_away.strip():
                                raise ValueError(
                                    "يجب إدخال الفريق الضيف."
                                )

                            if not edit_competition.strip():
                                raise ValueError(
                                    "يجب إدخال المسابقة."
                                )

                            if not edit_match_type.strip():
                                raise ValueError(
                                    "يجب إدخال نوع المباراة."
                                )

                            update_match(
                                match_id=selected_edit_id,
                                match_date=(
                                    edit_match_date.isoformat()
                                ),
                                status=edit_status,
                                competition=(
                                    edit_competition.strip()
                                ),
                                match_type=(
                                    edit_match_type.strip()
                                ),
                                team_home=(
                                    edit_team_home.strip()
                                ),
                                team_away=(
                                    edit_team_away.strip()
                                ),
                                home_score=(
                                    None
                                    if edit_status
                                    == "upcoming"
                                    else int(
                                        edit_home_score
                                    )
                                ),
                                away_score=(
                                    None
                                    if edit_status
                                    == "upcoming"
                                    else int(
                                        edit_away_score
                                    )
                                ),
                                venue=(
                                    edit_venue.strip()
                                    if edit_venue.strip()
                                    else "غير مسجل"
                                ),
                                importance=int(
                                    edit_importance
                                ),
                                notes=(
                                    selected_edit_match.get(
                                        "notes",
                                        "",
                                    )
                                ),
                                created_by=(
                                    current_user[
                                        "username"
                                    ]
                                ),
                            )

                            st.success(
                                "✅ تم تحديث بيانات المباراة."
                            )

                            st.rerun()

                        except Exception as exc:

                            st.error(
                                f"تعذر تحديث المباراة: {exc}"
                            )

        with st.expander(
            "➕ إضافة / تحديث مباراة قادمة",
            expanded=False,
        ):

            match_form_col1, match_form_col2 = (
                st.columns(2)
            )

            with match_form_col1:

                new_match_id = st.text_input(
                    "معرف المباراة",
                    placeholder="E005",
                )

                new_match_date = st.date_input(
                    "تاريخ المباراة",
                )

                new_competition = st.text_input(
                    "المسابقة / البطولة",
                    placeholder="League / Cup / Final",
                )

                new_match_type = st.text_input(
                    "نوع المباراة",
                    placeholder="League Match",
                )

            with match_form_col2:

                new_team_home = st.text_input(
                    "الفريق المضيف",
                    placeholder="Team A",
                )

                new_team_away = st.text_input(
                    "الفريق الضيف",
                    placeholder="Team B",
                )

                new_venue = st.text_input(
                    "الملعب",
                    placeholder="Stadium",
                )

                new_importance = st.slider(
                    "أهمية المباراة",
                    min_value=1,
                    max_value=5,
                    value=3,
                )

            if st.button(
                "💾 حفظ المباراة",
                use_container_width=True,
                type="primary",
            ):

                try:

                    if not new_match_id.strip():
                        raise ValueError(
                            "يجب إدخال معرف المباراة."
                        )

                    if not new_competition.strip():
                        raise ValueError(
                            "يجب إدخال اسم المسابقة."
                        )

                    if not new_match_type.strip():
                        raise ValueError(
                            "يجب إدخال نوع المباراة."
                        )

                    if not new_team_home.strip():
                        raise ValueError(
                            "يجب إدخال الفريق المضيف."
                        )

                    if not new_team_away.strip():
                        raise ValueError(
                            "يجب إدخال الفريق الضيف."
                        )

                    save_match(
                        match_id=new_match_id,
                        match_date=(
                            new_match_date.isoformat()
                        ),
                        status="upcoming",
                        competition=(
                            new_competition
                        ),
                        match_type=(
                            new_match_type
                        ),
                        team_home=(
                            new_team_home
                        ),
                        team_away=(
                            new_team_away
                        ),
                        venue=(
                            new_venue
                            if new_venue.strip()
                            else "غير مسجل"
                        ),
                        importance=int(
                            new_importance
                        ),
                        created_by=(
                            current_user["username"]
                        ),
                    )

                    st.success(
                        "✅ تم حفظ المباراة القادمة."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"تعذر حفظ المباراة: {exc}"
                    )

    if not upcoming_matches:

        st.info(
            "لا توجد مباريات قادمة مسجلة."
        )

    else:

        upcoming_rows = []

        for match in upcoming_matches:

            event_result = (
                upcoming_match_event_result(
                    match,
                    analysis,
                )
            )

            upcoming_rows.append(
                {
                    "ID":
                        match["match_id"],
                    "التاريخ":
                        match_date_text(match),
                    "المسابقة":
                        match["competition"],
                    "نوع المباراة":
                        match["match_type"],
                    "المضيف":
                        match["team_home"],
                    "الضيف":
                        match["team_away"],
                    "الملعب":
                        match["venue"]
                        or "غير مسجل",
                    "الأهمية":
                        match["importance"],
                    "Event Readiness":
                        (
                            f"{event_result['event_readiness_score']:.1f}"
                            if event_result
                            else "—"
                        ),
                    "Risk":
                        (
                            risk_label(
                                event_result[
                                    "final_risk_level"
                                ]
                            )
                            if event_result
                            else "—"
                        ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                upcoming_rows
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Event Readiness
# ============================================================

st.divider()

st.subheader(
    "🏆 جاهزية الحدث القادم"
)

current_event_result = None
next_event = None

if upcoming_matches:

    future_matches = [
        match
        for match in upcoming_matches
        if match.get("match_date")
    ]

    today = pd.Timestamp.now().normalize()

    future_matches = [
        match
        for match in future_matches
        if pd.Timestamp(
            match["match_date"]
        ) >= today
    ]

    future_matches = sorted(
        future_matches,
        key=lambda match: match["match_date"],
    )

    if future_matches:

        next_event = future_matches[0]

        current_event_result = (
            upcoming_match_event_result(
                next_event,
                analysis,
            )
        )

        event_cols = st.columns(4)

        with event_cols[0]:

            st.metric(
                "المباراة القادمة",
                (
                    f"{next_event['team_home']} "
                    f"vs "
                    f"{next_event['team_away']}"
                ),
            )

        with event_cols[1]:

            st.metric(
                "التاريخ",
                match_date_text(
                    next_event
                ),
            )

        with event_cols[2]:

            st.metric(
                "Event Readiness",
                (
                    f"{current_event_result['event_readiness_score']:.1f}"
                    "/100"
                ),
            )

        with event_cols[3]:

            st.metric(
                "Current Risk",
                risk_label(
                    current_event_result[
                        "final_risk_level"
                    ]
                ),
            )

        st.caption(
            f"المسابقة: "
            f"{next_event['competition']} | "
            f"النوع: "
            f"{next_event['match_type']} | "
            f"الأهمية: "
            f"{next_event['importance']}/5 | "
            f"الملعب: "
            f"{next_event['venue'] or 'غير مسجل'}"
        )

        if (
            current_event_result[
                "status"
            ]
            == "high_risk"
        ):

            st.error(
                "🚨 لا يُنصح بالمشاركة بكامل الشدة "
                "قبل إعادة التقييم."
            )

        elif (
            current_event_result[
                "status"
            ]
            == "needs_reassessment"
        ):

            st.warning(
                "🟡 يحتاج اللاعب إلى إعادة تقييم "
                "وتعديل الجهد التدريبي قبل المباراة."
            )

        else:

            st.success(
                "🟢 الجاهزية الحالية مناسبة "
                "مع الاستمرار في المراقبة."
            )

        st.info(
            "💡 "
            + current_event_result[
                "recommendation"
            ]
        )

    else:

        st.info(
            "لا توجد مباراة قادمة بتاريخ مستقبلي."
        )

else:

    st.info(
        "لا توجد مباريات قادمة مسجلة."
    )


# ============================================================
# Smart Alerts — current state
# ============================================================

trajectory_for_alerts = (
    predictions[
        predictions["player_id"]
        == str(selected_player)
    ].copy()
    if not predictions.empty
    else pd.DataFrame()
)

smart_alerts = build_smart_alerts(
    analysis=analysis,
    selected_row=selected_row,
    trajectory=trajectory_for_alerts,
    current_event_result=current_event_result,
)

render_smart_alerts(
    smart_alerts,
    current_user["role"],
)


# ============================================================
# What-If Simulator
# ============================================================

st.divider()

st.subheader(
    "🧪 What-If Simulator"
)

st.caption(
    "جرّب تدخلًا تدريبيًا وشاهد تأثيره المتوقع "
    "على الجاهزية والخطر والحدث القادم."
)

scenario_options = [
    "خفض الجهد التدريبي 10%",
    "خفض الجهد التدريبي 20%",
    "يوم استشفاء",
    "زيادة النوم ساعة",
]

scenario = st.selectbox(
    "اختر السيناريو",
    scenario_options,
)

scenario_changes = (
    build_scenario_changes(
        selected_row,
        scenario,
    )
)

if scenario_changes:

    with st.expander(
        "🔍 التغييرات التي سيحاكيها النظام",
        expanded=False,
    ):

        st.json(
            scenario_changes
        )

    if st.button(
        "🔮 محاكاة السيناريو",
        use_container_width=True,
    ):

        with st.spinner(
            "جاري محاكاة السيناريو..."
        ):

            try:

                what_if_result = run_what_if(
                    selected_row,
                    scenario_changes,
                )

            except Exception as exc:

                st.error(
                    f"تعذر تشغيل المحاكاة: {exc}"
                )

            else:

                current_state = (
                    what_if_result[
                        "current"
                    ]
                )

                scenario_state = (
                    what_if_result[
                        "scenario"
                    ]
                )

                deltas = (
                    what_if_result[
                        "deltas"
                    ]
                )

                interpretation = (
                    what_if_result[
                        "interpretation"
                    ]
                )

                st.markdown(
                    '<div class="what-if-card">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "### الحالة الحالية → بعد التدخل"
                )

                comparison_cols = (
                    st.columns(3)
                )

                with comparison_cols[0]:

                    st.metric(
                        "Readiness",
                        (
                            f"{scenario_state['readiness_score']:.1f}"
                            "/100"
                        ),
                        (
                            f"{deltas['readiness_score']:+.1f}"
                        ),
                    )

                with comparison_cols[1]:

                    probability_change = (
                        deltas[
                            "ml_injury_probability"
                        ]
                        * 100
                    )

                    st.metric(
                        "ML Risk Score",
                        (
                            f"{scenario_state['ml_injury_probability'] * 100:.2f}%"
                        ),
                        (
                            f"{probability_change:+.2f} نقطة"
                        ),
                        delta_color="inverse",
                    )

                with comparison_cols[2]:

                    st.metric(
                        "Final Risk",
                        risk_label(
                            scenario_state[
                                "final_risk"
                            ]
                        ),
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                current_col, scenario_col = (
                    st.columns(2)
                )

                with current_col:

                    st.markdown(
                        "### 🔴 الحالة الحالية"
                    )

                    st.write(
                        f"**Readiness:** "
                        f"{current_state['readiness_score']:.1f}/100"
                    )

                    st.write(
                        f"**ML Risk:** "
                        f"{risk_label(current_state['ml_risk'])}"
                    )

                    st.write(
                        f"**Final Risk:** "
                        f"{risk_label(current_state['final_risk'])}"
                    )

                with scenario_col:

                    st.markdown(
                        "### 🟢 بعد التدخل"
                    )

                    st.write(
                        f"**Readiness:** "
                        f"{scenario_state['readiness_score']:.1f}/100"
                    )

                    st.write(
                        f"**ML Risk:** "
                        f"{risk_label(scenario_state['ml_risk'])}"
                    )

                    st.write(
                        f"**Final Risk:** "
                        f"{risk_label(scenario_state['final_risk'])}"
                    )

                if next_event is not None:

                    scenario_event_result = (
                        calculate_event_readiness(
                            event_name=str(
                                next_event[
                                    "match_type"
                                ]
                            ),
                            event_date=(
                                pd.Timestamp(
                                    next_event[
                                        "match_date"
                                    ]
                                ).date()
                            ),
                            event_importance=int(
                                next_event[
                                    "importance"
                                ]
                            ),
                            hybrid_result={
                                "readiness_score":
                                    scenario_state[
                                        "readiness_score"
                                    ],
                                "final_risk_level":
                                    scenario_state[
                                        "final_risk"
                                    ],
                            },
                        )
                    )

                    st.divider()

                    st.markdown(
                        "### 🏆 تأثير السيناريو على المباراة القادمة"
                    )

                    event_compare_cols = (
                        st.columns(2)
                    )

                    current_event_score = (
                        current_event_result[
                            "event_readiness_score"
                        ]
                        if current_event_result
                        else None
                    )

                    with event_compare_cols[0]:

                        if (
                            current_event_score
                            is not None
                        ):

                            st.metric(
                                "الحالة الحالية",
                                (
                                    f"{current_event_score:.1f}"
                                    "/100"
                                ),
                            )

                    with event_compare_cols[1]:

                        st.metric(
                            "بعد السيناريو",
                            (
                                f"{scenario_event_result['event_readiness_score']:.1f}"
                                "/100"
                            ),
                            (
                                f"{scenario_event_result['event_readiness_score'] - current_event_score:+.1f}"
                                if current_event_score
                                is not None
                                else None
                            ),
                        )

                st.success(
                    "💡 "
                    + interpretation
                )

else:

    st.info(
        "اختر سيناريو للمحاكاة."
    )


# ============================================================
# Selected player
# ============================================================

st.divider()

player_col, risk_col = (
    st.columns(
        [2, 1]
    )
)

with player_col:

    st.subheader(
        f"👤 {selected_row['name']}"
    )

    st.caption(
        f"Player ID: "
        f"{selected_row['player_id']} | "
        f"Match: "
        f"{selected_row['match_id']}"
    )

with risk_col:

    if final_risk == "elevated":

        st.error(
            f"الخطر النهائي: "
            f"{risk_label(final_risk)}"
        )

    elif final_risk == "moderate":

        st.warning(
            f"الخطر النهائي: "
            f"{risk_label(final_risk)}"
        )

    else:

        st.success(
            f"الخطر النهائي: "
            f"{risk_label(final_risk)}"
        )


# ============================================================
# Team Training — One Session for the Whole Team
# ============================================================

st.divider()
st.subheader("📅 تدريب الفريق والحضور")
st.caption(
    "ينشئ المدرب تدريب اليوم مرة واحدة للفريق، ثم يسجل حالة حضور كل لاعب. "
    "تظهر النتيجة تلقائيًا في صفحة اللاعب."
)

selected_player_record = get_player(str(selected_player)) or {}
current_team_name = (
    selected_player_record.get("team_name")
    or "AIRA Demo Team"
)

if current_user["role"] == "coach":

    with st.expander(
        "➕ إنشاء تدريب للفريق",
        expanded=False,
    ):

        team_form_col1, team_form_col2 = st.columns(2)

        with team_form_col1:

            team_session_name = st.text_input(
                "اسم التدريب",
                placeholder="تدريب تكتيكي / لياقة / مهارات",
                key="team_session_name",
            )

            team_training_type = st.selectbox(
                "نوع التدريب",
                [
                    "لياقة",
                    "مهارات",
                    "تكتيك",
                    "استشفاء",
                    "أخرى",
                ],
                key="team_training_type",
            )

            team_session_date = st.date_input(
                "تاريخ التدريب",
                key="team_session_date",
            )

            team_start_time = st.time_input(
                "وقت البداية",
                key="team_start_time",
            )

        with team_form_col2:

            team_end_time = st.time_input(
                "وقت النهاية",
                key="team_end_time",
            )

            team_duration = st.number_input(
                "المدة (دقيقة)",
                min_value=1.0,
                max_value=600.0,
                value=60.0,
                step=1.0,
                key="team_duration",
            )

            team_intensity = st.number_input(
                "شدة التدريب",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                key="team_intensity",
            )

            team_status = st.selectbox(
                "حالة التدريب",
                [
                    "scheduled",
                    "completed",
                    "cancelled",
                ],
                format_func=lambda value: {
                    "scheduled": "📅 مجدول",
                    "completed": "✅ مكتمل",
                    "cancelled": "⛔ ملغى",
                }[value],
                key="team_status",
            )

        team_notes = st.text_area(
            "ملاحظات التدريب",
            placeholder="ملاحظات عامة للفريق...",
            key="team_training_notes",
        )

        if st.button(
            "💾 حفظ تدريب الفريق",
            key="save_team_training",
            use_container_width=True,
            type="primary",
        ):

            try:

                if not team_session_name.strip():
                    raise ValueError(
                        "يجب إدخال اسم التدريب."
                    )

                if team_end_time < team_start_time:
                    raise ValueError(
                        "وقت النهاية يجب أن يكون بعد وقت البداية."
                    )

                training_session_id = add_team_training_session(
                    team_name=str(current_team_name),
                    session_name=team_session_name.strip(),
                    training_type=team_training_type,
                    session_date=team_session_date.isoformat(),
                    start_time=team_start_time.strftime("%H:%M"),
                    end_time=team_end_time.strftime("%H:%M"),
                    duration_min=float(team_duration),
                    intensity=float(team_intensity),
                    status=team_status,
                    notes=team_notes.strip() or None,
                    created_by=current_user["username"],
                )

                st.success(
                    f"✅ تم إنشاء تدريب الفريق #{training_session_id}. "
                    "تم تجهيز سجل حضور لجميع لاعبي الفريق."
                )
                st.rerun()

            except Exception as exc:
                st.error(
                    f"تعذر إنشاء تدريب الفريق: {exc}"
                )

    team_sessions = get_team_training_sessions(
        team_name=str(current_team_name)
    )

    if team_sessions:

        st.markdown("#### 📋 تدريبات الفريق")

        session_options = {
            (
                f"#{session['id']} — "
                f"{session['session_date']} — "
                f"{session['session_name']}"
            ): session["id"]
            for session in team_sessions
        }

        selected_team_session_label = st.selectbox(
            "اختر تدريبًا لتسجيل الحضور",
            list(session_options.keys()),
            key="team_session_selector",
        )

        selected_team_session_id = session_options[
            selected_team_session_label
        ]

        selected_team_session = next(
            session
            for session in team_sessions
            if session["id"] == selected_team_session_id
        )

        st.caption(
            f"الفريق: {selected_team_session['team_name']} | "
            f"النوع: {selected_team_session['training_type']} | "
            f"المدة: {selected_team_session['duration_min'] or '—'} دقيقة | "
            f"الحالة: {selected_team_session['status']}"
        )

        attendance_rows = get_team_training_attendance(
            team_training_session_id=selected_team_session_id
        )

        if attendance_rows:

            with st.form(
                "team_attendance_form"
            ):

                attendance_choices = {}

                for record in attendance_rows:

                    current_status = record[
                        "attendance_status"
                    ]

                    attendance_choices[
                        record["player_id"]
                    ] = st.selectbox(
                        (
                            f"{record['player_id']} — "
                            f"{record['full_name']}"
                        ),
                        [
                            "pending",
                            "present",
                            "absent",
                            "excused",
                        ],
                        index=[
                            "pending",
                            "present",
                            "absent",
                            "excused",
                        ].index(current_status),
                        format_func=lambda value: {
                            "pending": "🕓 لم يُسجل بعد",
                            "present": "✅ حاضر",
                            "absent": "❌ غائب",
                            "excused": "🟡 معتذر",
                        }[value],
                        key=(
                            f"attendance_{selected_team_session_id}_"
                            f"{record['player_id']}"
                        ),
                    )

                save_attendance = st.form_submit_button(
                    "💾 حفظ حضور الفريق",
                    use_container_width=True,
                )

            if save_attendance:

                try:

                    for player_id, status in attendance_choices.items():

                        set_team_training_attendance(
                            team_training_session_id=(
                                selected_team_session_id
                            ),
                            player_id=str(player_id),
                            attendance_status=status,
                            updated_by=current_user[
                                "username"
                            ],
                        )

                    st.success(
                        "✅ تم حفظ حضور الفريق. "
                        "وسيظهر لكل لاعب في صفحته الخاصة."
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"تعذر حفظ الحضور: {exc}"
                    )

        else:
            st.info(
                "لا يوجد لاعبون مرتبطون بهذا التدريب بعد."
            )

else:

    # Player does not need a separate team-training table here.
    # Team training will be shown once inside the Player Profile
    # -> Training Record tab below.
    st.caption(
        "🔒 تدريبات الفريق والحضور تظهر في سجل التدريب "
        "داخل ملف اللاعب."
    )


# ============================================================
# Player Profile — Contracts, Achievements, Training Record
# ============================================================

st.divider()

st.subheader(
    "📁 ملف اللاعب"
)

st.caption(
    "العقود والإنجازات وسجل التدريب المرتبط بهذا اللاعب."
)

if current_user["role"] == "player":
    st.info(
        "🔒 في حساب اللاعب، العقود والإنجازات وسجل التدريب "
        "للعرض فقط وتُحدّث من المدرب/الإدارة."
    )

contracts_tab, achievements_tab, training_tab = st.tabs(
    [
        "📄 العقود",
        "🏆 الإنجازات",
        "🏋️ سجل التدريب",
    ]
)

# ------------------------------------------------------------
# Contracts
# ------------------------------------------------------------

with contracts_tab:

    contracts = get_player_contracts(
        str(selected_player)
    )

    if current_user["role"] == "coach":

        with st.expander(
            "➕ إضافة عقد",
            expanded=False,
        ):

            contract_col1, contract_col2 = (
                st.columns(2)
            )

            with contract_col1:

                contract_name = st.text_input(
                    "اسم العقد",
                    placeholder="Professional Contract",
                )

                contract_start = st.date_input(
                    "بداية العقد",
                )

            with contract_col2:

                contract_end = st.date_input(
                    "نهاية العقد",
                )

                contract_status = st.selectbox(
                    "حالة العقد",
                    [
                        "active",
                        "completed",
                        "expired",
                        "pending",
                    ],
                )

            if st.button(
                "💾 حفظ العقد",
                key="save_contract",
                use_container_width=True,
            ):

                try:

                    if not contract_name.strip():
                        raise ValueError(
                            "يجب إدخال اسم العقد."
                        )

                    add_contract(
                        player_id=str(
                            selected_player
                        ),
                        contract_name=(
                            contract_name.strip()
                        ),
                        start_date=(
                            contract_start.isoformat()
                        ),
                        end_date=(
                            contract_end.isoformat()
                        ),
                        contract_status=(
                            contract_status
                        ),
                    )

                    st.success(
                        "✅ تم حفظ العقد."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"تعذر حفظ العقد: {exc}"
                    )

    if contracts:

        contract_rows = [
            {
                "العقد":
                    contract["contract_name"],
                "البداية":
                    contract["start_date"]
                    or "غير مسجل",
                "النهاية":
                    contract["end_date"]
                    or "غير مسجل",
                "الحالة":
                    contract["status"],
            }
            for contract in contracts
        ]

        st.dataframe(
            pd.DataFrame(
                contract_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "لا توجد عقود مسجلة لهذا اللاعب."
        )

# ------------------------------------------------------------
# Achievements
# ------------------------------------------------------------

with achievements_tab:

    achievements = get_player_achievements(
        str(selected_player)
    )

    if current_user["role"] == "coach":

        with st.expander(
            "➕ إضافة إنجاز",
            expanded=False,
        ):

            achievement_col1, achievement_col2 = (
                st.columns(2)
            )

            with achievement_col1:

                achievement_title = st.text_input(
                    "اسم الإنجاز",
                    placeholder="Best Player / Championship",
                    key="achievement_title",
                )

                achievement_date = st.date_input(
                    "تاريخ الإنجاز",
                    key="achievement_date",
                )

            with achievement_col2:

                achievement_description = (
                    st.text_area(
                        "الوصف",
                        placeholder=(
                            "تفاصيل مختصرة عن الإنجاز..."
                        ),
                        key="achievement_description",
                    )
                )

            if st.button(
                "💾 حفظ الإنجاز",
                key="save_achievement",
                use_container_width=True,
            ):

                try:

                    if not achievement_title.strip():
                        raise ValueError(
                            "يجب إدخال اسم الإنجاز."
                        )

                    add_achievement(
                        player_id=str(
                            selected_player
                        ),
                        title=(
                            achievement_title.strip()
                        ),
                        achievement_date=(
                            achievement_date.isoformat()
                        ),
                        description=(
                            achievement_description.strip()
                            or None
                        ),
                    )

                    st.success(
                        "✅ تم حفظ الإنجاز، وسيظهر تلقائيًا "
                        "في صفحة اللاعب."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"تعذر حفظ الإنجاز: {exc}"
                    )

    else:

        st.info(
            "🏆 الإنجازات تُضاف من المدرب وتظهر تلقائيًا "
            "في صفحة اللاعب."
        )

    if achievements:

        if current_user["role"] == "player":
            st.caption(
                "🔄 هذه الإنجازات مرتبطة بملفك وتُحدّث تلقائيًا "
                "عند إضافتها من المدرب."
            )

        achievement_rows = [
            {
                "الإنجاز":
                    achievement["title"],
                "التاريخ":
                    achievement["achievement_date"]
                    or "غير مسجل",
                "الوصف":
                    achievement["description"]
                    or "—",
            }
            for achievement in achievements
        ]

        st.dataframe(
            pd.DataFrame(
                achievement_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "لا توجد إنجازات مسجلة لهذا اللاعب."
        )

# ------------------------------------------------------------
# Training Record + Attendance
# ------------------------------------------------------------

with training_tab:

    if current_user["role"] == "coach":

        st.info(
            "🏋️ التدريب اليومي والحضور يُدار من المدرب. "
            "ما يسجله المدرب يظهر تلقائيًا في صفحة اللاعب."
        )

        with st.expander(
            "➕ إضافة تدريب يومي وتسجيل الحضور",
            expanded=False,
        ):

            training_col1, training_col2 = (
                st.columns(2)
            )

            with training_col1:

                session_name = st.text_input(
                    "اسم التدريب",
                    placeholder=(
                        "لياقة / مهارات / تكتيك / استشفاء"
                    ),
                    key="daily_session_name",
                )

                session_date = st.date_input(
                    "تاريخ التدريب",
                    key="daily_session_date",
                )

                duration_min = st.number_input(
                    "مدة التدريب (دقيقة)",
                    min_value=1.0,
                    max_value=600.0,
                    value=60.0,
                    step=1.0,
                    key="daily_duration",
                )

            with training_col2:

                intensity = st.number_input(
                    "شدة التدريب",
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                    key="daily_intensity",
                )

                attendance_status = st.selectbox(
                    "حالة الحضور",
                    [
                        "pending",
                        "present",
                        "absent",
                        "excused",
                        "cancelled",
                    ],
                    format_func=lambda value: {
                        "pending": "🕓 لم يُسجل بعد",
                        "present": "✅ حاضر",
                        "absent": "❌ غائب",
                        "excused": "🟡 معتذر",
                        "cancelled": "⛔ التدريب ملغى",
                    }[value],
                    key="daily_attendance",
                )

                training_notes = st.text_area(
                    "ملاحظات المدرب",
                    placeholder="ملاحظات عن التدريب أو الحضور...",
                    key="daily_training_notes",
                )

            if st.button(
                "💾 حفظ التدريب والحضور",
                key="save_daily_training",
                use_container_width=True,
                type="primary",
            ):

                try:

                    if not session_name.strip():
                        raise ValueError(
                            "يجب إدخال اسم التدريب."
                        )

                    training_id = add_training_session(
                        player_id=str(
                            selected_player
                        ),
                        session_name=(
                            session_name.strip()
                        ),
                        session_date=(
                            session_date.isoformat()
                        ),
                        duration_min=(
                            duration_min
                        ),
                        intensity=(
                            intensity
                        ),
                        notes=(
                            training_notes.strip()
                            or None
                        ),
                    )

                    add_training_attendance(
                        training_session_id=training_id,
                        player_id=str(
                            selected_player
                        ),
                        attendance_status=(
                            attendance_status
                        ),
                        attendance_note=(
                            training_notes.strip()
                            or None
                        ),
                        updated_by=(
                            current_user["username"]
                        ),
                    )

                    st.success(
                        "✅ تم حفظ التدريب والحضور، "
                        "وسيظهر تلقائيًا في صفحة اللاعب."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"تعذر حفظ التدريب: {exc}"
                    )

        st.markdown(
            "#### 📋 سجل التدريب والحضور"
        )

        training_records = (
            get_player_training_attendance(
                str(selected_player)
            )
        )

    else:

        st.info(
            "🔒 سجل التدريب والحضور للعرض فقط. "
            "يتم تحديثه من المدرب."
        )

        individual_training_records = (
            get_player_training_attendance(
                str(selected_player)
            )
        )

        team_training_records = (
            get_team_training_attendance(
                player_id=str(selected_player)
            )
        )

        # Canonical player-facing training history:
        # team sessions + legacy individual sessions.
        training_records = []

        for record in team_training_records:
            training_records.append(
                {
                    **record,
                    "record_source": "team",
                    "notes": (
                        record.get("notes")
                        or record.get("attendance_note")
                    ),
                }
            )

        existing_team_keys = {
            (
                str(record.get("session_date")),
                str(record.get("session_name")),
                str(record.get("training_session_id")),
            )
            for record in training_records
        }

        for record in individual_training_records:
            key = (
                str(record.get("session_date")),
                str(record.get("session_name")),
                str(record.get("training_session_id")),
            )

            if key not in existing_team_keys:
                training_records.append(
                    {
                        **record,
                        "record_source": "individual",
                    }
                )

    if training_records:

        training_rows = []

        for session in training_records:

            training_rows.append(
                {
                    "التدريب":
                        session["session_name"],
                    "النوع":
                        session.get("training_type")
                        or "تدريب",
                    "التاريخ":
                        session["session_date"],
                    "المدة":
                        (
                            f"{float(session['duration_min']):.0f} دقيقة"
                            if session.get("duration_min") is not None
                            else "—"
                        ),
                    "الشدة":
                        (
                            f"{float(session['intensity']):.1f}"
                            if session.get("intensity") is not None
                            else "—"
                        ),
                    "الحضور":
                        {
                            "pending":
                                "🕓 لم يُسجل",
                            "present":
                                "✅ حاضر",
                            "absent":
                                "❌ غائب",
                            "excused":
                                "🟡 معتذر",
                            "cancelled":
                                "⛔ ملغى",
                        }.get(
                            session.get(
                                "attendance_status"
                            ),
                            "🕓 لم يُسجل",
                        ),
                    "حالة التدريب":
                        {
                            "scheduled": "📅 مجدول",
                            "completed": "✅ مكتمل",
                            "cancelled": "⛔ ملغى",
                        }.get(
                            session.get(
                                "training_status"
                            ),
                            "—",
                        ),
                    "الملاحظات":
                        session.get("notes")
                        or session.get(
                            "attendance_note"
                        )
                        or "—",
                }
            )

        st.dataframe(
            pd.DataFrame(
                training_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "لا يوجد سجل تدريب محفوظ لهذا اللاعب بعد."
        )

# ------------------------------------------------------------
# END Training Record + Attendance
# ------------------------------------------------------------


# ============================================================
# Core metrics
# ============================================================

metric_cols = st.columns(4)

with metric_cols[0]:

    st.metric(
        "جاهزية اللاعب",
        f"{readiness_score:.1f}/100",
    )

with metric_cols[1]:

    st.metric(
        "Readiness Risk",
        risk_label(
            readiness_risk
        ),
    )

with metric_cols[2]:

    st.metric(
        "ML Risk",
        risk_label(
            ml_risk
        ),
    )

with metric_cols[3]:

    st.metric(
        "ACWR",
        f"{metrics.get('acwr', 0):.3f}",
    )


# ============================================================
# Historical Risk Trajectory
# ============================================================

st.divider()

st.subheader(
    "📈 مسار الخطر التاريخي"
)

trajectory = (
    predictions[
        predictions["player_id"]
        == str(selected_player)
    ].copy()
    if not predictions.empty
    else pd.DataFrame()
)

if trajectory.empty:

    st.info(
        "لا توجد بيانات Temporal ML لهذا اللاعب."
    )

else:

    trajectory["model_score_percent"] = (
        trajectory[
            "predicted_probability"
        ]
        * 100
    )

    fig_risk = px.line(
        trajectory,
        x="match_number",
        y="model_score_percent",
        markers=True,
    )

    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#cbd5e1"
        ),
        xaxis_title="رقم المباراة",
        yaxis_title="Model Score",
        yaxis_range=[0, 100],
        margin=dict(
            l=20,
            r=20,
            t=25,
            b=20,
        ),
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )

    latest_probability = float(
        trajectory.iloc[-1][
            "predicted_probability"
        ]
    )

    peak_probability = float(
        trajectory[
            "predicted_probability"
        ].max()
    )

    peak_match = int(
        trajectory.loc[
            trajectory[
                "predicted_probability"
            ].idxmax(),
            "match_number",
        ]
    )

    trajectory_cols = (
        st.columns(3)
    )

    with trajectory_cols[0]:

        st.metric(
            "الخطر في آخر سجل",
            risk_label(
                probability_to_risk(
                    latest_probability
                )
            ),
        )

    with trajectory_cols[1]:

        st.metric(
            "أعلى Model Score",
            f"{peak_probability * 100:.1f}%",
        )

    with trajectory_cols[2]:

        st.metric(
            "أعلى إشارة في",
            f"M{peak_match:02d}",
        )


# ============================================================
# Player Digital Twin
# ============================================================

st.divider()

st.subheader(
    "🧬 Player Digital Twin"
)

st.caption(
    "مقارنة حالة اللاعب الحالية بخط الأساس الخاص به "
    "لاكتشاف أي انحراف غير معتاد عن مستواه الطبيعي."
)


# ------------------------------------------------------------
# Baseline vs current values
# ------------------------------------------------------------

digital_twin_rows = [
    {
        "المؤشر": "النوم",
        "خط الأساس": float(
            selected_row["baseline_sleep"]
        ),
        "الحالة الحالية": float(
            selected_row["sleep_duration"]
        ),
        "الوحدة": "ساعة",
    },
    {
        "المؤشر": "زمن الاستجابة",
        "خط الأساس": float(
            selected_row["baseline_reaction_time"]
        ),
        "الحالة الحالية": float(
            selected_row["reaction_time_ms"]
        ),
        "الوحدة": "ms",
    },
    {
        "المؤشر": "الحمل التدريبي",
        "خط الأساس": float(
            selected_row["baseline_training_load"]
        ),
        "الحالة الحالية": float(
            selected_row["acute_load"]
        ),
        "الوحدة": "load",
    },
]

digital_twin_df = pd.DataFrame(
    digital_twin_rows
)

digital_twin_df["الانحراف"] = (
    digital_twin_df["الحالة الحالية"]
    - digital_twin_df["خط الأساس"]
)

st.dataframe(
    digital_twin_df[
        [
            "المؤشر",
            "خط الأساس",
            "الحالة الحالية",
            "الانحراف",
            "الوحدة",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)


# ------------------------------------------------------------
# Deviation summary
# ------------------------------------------------------------

twin_cols = st.columns(4)

with twin_cols[0]:
    sleep_deviation = float(
        selected_row["sleep_deviation"]
    )
    st.metric(
        "Sleep Deviation",
        f"{sleep_deviation:+.2f}",
    )

with twin_cols[1]:
    reaction_deviation = float(
        selected_row["reaction_time_deviation"]
    )
    st.metric(
        "Reaction Deviation",
        f"{reaction_deviation:+.2f}",
    )

with twin_cols[2]:
    recovery_drop = float(
        selected_row["recovery_drop"]
    )
    st.metric(
        "Recovery Change",
        f"{recovery_drop:+.2f}",
    )

with twin_cols[3]:
    training_deviation = float(
        selected_row["acute_load"]
        - selected_row["baseline_training_load"]
    )
    st.metric(
        "Training Deviation",
        f"{training_deviation:+.2f}",
    )


# ------------------------------------------------------------
# Explain the digital twin
# ------------------------------------------------------------

digital_twin_messages = []

if sleep_deviation < -0.5:
    digital_twin_messages.append(
        "النوم أقل من خط الأساس المعتاد للاعب."
    )
elif sleep_deviation > 0.5:
    digital_twin_messages.append(
        "النوم أعلى من خط الأساس المعتاد للاعب."
    )

if reaction_deviation > 5:
    digital_twin_messages.append(
        "زمن الاستجابة أبطأ من المستوى المعتاد للاعب."
    )
elif reaction_deviation < -5:
    digital_twin_messages.append(
        "زمن الاستجابة أفضل من خط الأساس المعتاد."
    )

if recovery_drop < -5:
    digital_twin_messages.append(
        "مستوى التعافي أقل من المعتاد."
    )
elif recovery_drop > 5:
    digital_twin_messages.append(
        "مستوى التعافي أعلى من المعتاد."
    )

if training_deviation > 30:
    digital_twin_messages.append(
        "الحمل الحديث أعلى من خط الأساس التدريبي."
    )
elif training_deviation < -30:
    digital_twin_messages.append(
        "الحمل الحديث أقل من خط الأساس التدريبي."
    )

if digital_twin_messages:
    st.warning(
        "🧠 **AI Digital Twin Insights**\n\n"
        + "\n".join(
            f"• {message}"
            for message in digital_twin_messages
        )
    )
else:
    st.success(
        "🟢 مؤشرات اللاعب الحالية قريبة من خط الأساس الشخصي."
    )


# ============================================================
# Training / Recovery + AI
# ============================================================

st.divider()

left_col, right_col = (
    st.columns(
        [2, 1]
    )
)

with left_col:

    st.subheader(
        "🏋️ االجهد التدريبي والتعافي"
    )

    timeline = (
        player_history[
            [
                "match_number",
                "session_load",
                "recovery_score",
            ]
        ]
        .copy()
    )

    timeline_long = timeline.melt(
        id_vars=[
            "match_number"
        ],
        value_vars=[
            "session_load",
            "recovery_score",
        ],
        var_name="metric",
        value_name="value",
    )

    timeline_long["metric"] = (
        timeline_long[
            "metric"
        ].map(
            {
                "session_load":
                    "الجهد التدريبي",
                "recovery_score":
                    "التعافي",
            }
        )
    )

    fig = px.line(
        timeline_long,
        x="match_number",
        y="value",
        color="metric",
        markers=True,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#cbd5e1"
        ),
        xaxis_title="رقم المباراة",
        yaxis_title="القيمة",
        margin=dict(
            l=20,
            r=20,
            t=25,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


with right_col:

    st.subheader(
        "🧠 تحليل الذكاء الاصطناعي"
    )

    st.write(
        f"**الخطر النهائي:** "
        f"{risk_label(final_risk)}"
    )

    st.write(
        f"**ML Risk:** "
        f"{risk_label(ml_risk)}"
    )

    st.write(
        f"**Readiness:** "
        f"{readiness_score:.1f}/100"
    )

    st.write(
        f"**Sleep Quality:** "
        f"{selected_row['sleep_quality']:.1f}/100"
    )

    st.markdown(
        "**العوامل المؤثرة:**"
    )

    if factors:

        for factor in factors:

            if isinstance(
                factor,
                dict,
            ):

                label = factor.get(
                    "label",
                    factor.get(
                        "feature",
                        "عامل",
                    ),
                )

                impact = factor.get(
                    "impact"
                )

                if impact is None:

                    st.write(
                        f"• {label}"
                    )

                else:

                    st.write(
                        f"• {label} "
                        f"({impact:+})"
                    )

            else:

                st.write(
                    f"• {factor}"
                )

    else:

        st.write(
            "• لا توجد عوامل تحذير رئيسية."
        )


# ============================================================
# Recommendation
# ============================================================

st.divider()

st.subheader(
    "💡 توصية النظام"
)

if final_risk == "elevated":

    st.error(
        recommendation
    )

elif final_risk == "moderate":

    st.warning(
        recommendation
    )

else:

    st.success(
        recommendation
    )


# ============================================================
# Detailed metrics
# ============================================================

with st.expander(
    "🔬 عرض المؤشرات التفصيلية"
):

    detail_cols = st.columns(4)

    with detail_cols[0]:

        st.metric(
            "Session Load",
            f"{metrics.get('session_load', 0):.2f}",
        )

        st.metric(
            "Acute Load",
            f"{metrics.get('acute_load', 0):.2f}",
        )

    with detail_cols[1]:

        st.metric(
            "Chronic Load",
            f"{metrics.get('chronic_load', 0):.2f}",
        )

        st.metric(
            "ACWR",
            f"{metrics.get('acwr', 0):.3f}",
        )

    with detail_cols[2]:

        st.metric(
            "Recovery",
            f"{subscores.get('recovery', 0):.1f}",
        )

        st.metric(
            "Sleep Subscore",
            f"{subscores.get('sleep', 0):.1f}",
        )

    with detail_cols[3]:

        st.metric(
            "Workload Balance",
            f"{subscores.get('workload_balance', 0):.1f}",
        )

        st.metric(
            "Reaction",
            f"{subscores.get('reaction', 0):.1f}",
        )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "⚠️ HARES AI  — نموذج أولي بحثي يعتمد حاليًا "
    "على بيانات تطوير اصطناعية. "
    "What-If وEvent Readiness أدوات دعم قرار أولية "
    "وليستا تصريحًا طبيًا بالمشاركة."
)