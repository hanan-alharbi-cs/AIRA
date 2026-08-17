from ai.features.load_features import (
    calculate_acwr,
    calculate_acute_load,
    calculate_chronic_load,
    calculate_session_load,
)

from ai.features.baseline_features import (
    calculate_sleep_deviation,
    calculate_reaction_time_deviation,
    calculate_training_load_deviation,
)

from ai.features.recovery_features import (
    calculate_recovery_subscore,
    calculate_sleep_subscore,
    calculate_reaction_subscore,
)

from ai.scoring.risk import get_risk_level

from ai.scoring.readiness_engine import calculate_readiness

from ai.scoring.early_warning import calculate_early_warning

from ai.explainability.rules_explainer import explain_readiness


def create_test_player() -> dict:
    """Return a deterministic player scenario for testing."""

    return {
        "player_id": "P001",
        "training_duration_min": 90,
        "rpe": 7,
        "sleep_duration": 5.0,
        "sleep_quality": 2,
        "reaction_time_ms": 335,
        "recovery_score": 58,
        "baseline_sleep": 8.0,
        "baseline_reaction_time_ms": 250.0,
        "baseline_training_load": 500.0,
        "acute_load": 650.0,
        "chronic_load": 500.0,
        "injury_context": False,
    }


# ---------------------------------------------------------
# Load feature tests
# ---------------------------------------------------------


def test_session_load():
    assert calculate_session_load(90, 7) == 630


def test_acute_load():
    loads = [100, 110, 120, 90, 130, 100, 150]

    assert calculate_acute_load(loads) == 800


def test_chronic_load():
    loads = [100] * 28

    assert calculate_chronic_load(loads) == 100


def test_acwr():
    assert calculate_acwr(650, 500) == 1.3


# ---------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------


def test_sleep_deviation():
    assert calculate_sleep_deviation(6, 8) == -0.25


def test_reaction_time_deviation():
    assert calculate_reaction_time_deviation(
        300,
        250,
    ) == 0.2


def test_training_load_deviation():
    assert calculate_training_load_deviation(
        120,
        100,
    ) == 0.2


# ---------------------------------------------------------
# Subscore tests
# ---------------------------------------------------------


def test_recovery_subscore():
    assert calculate_recovery_subscore(80) == 80


def test_sleep_subscore():
    score = calculate_sleep_subscore(
        sleep_duration=6,
        baseline_sleep=8,
        sleep_quality=3,
    )

    assert 0 <= score <= 100


def test_reaction_subscore():
    score = calculate_reaction_subscore(
        reaction_time_ms=300,
        baseline_reaction_time_ms=250,
    )

    assert score == 80


# ---------------------------------------------------------
# Risk tests
# ---------------------------------------------------------


def test_risk_level_low():
    assert get_risk_level(80) == "low"


def test_risk_level_moderate():
    assert get_risk_level(60) == "moderate"


def test_risk_level_elevated():
    assert get_risk_level(30) == "elevated"


# ---------------------------------------------------------
# Readiness Engine tests
# ---------------------------------------------------------


def test_readiness_score_range():
    player = create_test_player()

    result = calculate_readiness(player)

    assert 0 <= result["readiness_score"] <= 100


def test_readiness_result_contains_required_fields():
    player = create_test_player()

    result = calculate_readiness(player)

    required_fields = {
        "player_id",
        "readiness_score",
        "risk_level",
        "recommendation",
        "data_quality",
        "metrics",
        "subscores",
        "deviations",
    }

    assert required_fields.issubset(result.keys())


def test_readiness_is_deterministic():
    player = create_test_player()

    result_1 = calculate_readiness(player)
    result_2 = calculate_readiness(player)

    assert result_1 == result_2


# ---------------------------------------------------------
# Explainability tests
# ---------------------------------------------------------


def test_explainability_returns_top_three_factors():
    player = create_test_player()

    readiness_result = calculate_readiness(player)

    factors = explain_readiness(
        player,
        readiness_result,
    )

    assert len(factors) <= 3


def test_explainability_factor_structure():
    player = create_test_player()

    readiness_result = calculate_readiness(player)

    factors = explain_readiness(
        player,
        readiness_result,
    )

    for factor in factors:
        assert "feature" in factor
        assert "label" in factor
        assert "impact" in factor
        assert "direction" in factor


# ---------------------------------------------------------
# Early Warning tests
# ---------------------------------------------------------


def test_early_warning_low():
    result = calculate_early_warning(
        workload_spike=False,
        sleep_deviation_bad=False,
        recovery_drop=False,
        reaction_deviation_bad=False,
        injury_context=False,
    )

    assert result["warning_points"] == 0
    assert result["risk_level"] == "low"


def test_early_warning_moderate():
    result = calculate_early_warning(
        workload_spike=True,
        sleep_deviation_bad=True,
        recovery_drop=False,
        reaction_deviation_bad=False,
        injury_context=False,
    )

    assert result["warning_points"] == 2
    assert result["risk_level"] == "moderate"


def test_early_warning_elevated():
    result = calculate_early_warning(
        workload_spike=True,
        sleep_deviation_bad=True,
        recovery_drop=True,
        reaction_deviation_bad=True,
        injury_context=True,
    )

    assert result["warning_points"] == 5
    assert result["risk_level"] == "elevated"


def test_early_warning_signals():
    result = calculate_early_warning(
        workload_spike=True,
        sleep_deviation_bad=True,
        recovery_drop=True,
        reaction_deviation_bad=False,
        injury_context=False,
    )

    assert result["signals"] == [
        "workload_spike",
        "sleep_deviation",
        "recovery_drop",
    ]