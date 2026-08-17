def get_risk_level(
    readiness_score: float,
) -> str:
    """
    Convert readiness score into the prototype status band.

    0-39   -> elevated
    40-69  -> moderate
    70-100 -> low
    """

    if not 0 <= readiness_score <= 100:
        raise ValueError(
            "readiness_score must be between 0 and 100."
        )

    if readiness_score < 40:
        return "elevated"

    if readiness_score < 70:
        return "moderate"

    return "low"