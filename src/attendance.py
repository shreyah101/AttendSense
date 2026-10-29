"""Exact attendance arithmetic. Threshold arguments are percentages (1 to 100)."""
from dataclasses import dataclass
from fractions import Fraction


def validate(attended: int, conducted: int, remaining: int = 0, threshold: float = 75) -> Fraction:
    for value in (attended, conducted, remaining):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Class counts must be nonnegative integers.")
    if attended > conducted:
        raise ValueError("Attended classes cannot exceed conducted classes.")
    try:
        target = Fraction(str(threshold)) / 100
    except (ValueError, ZeroDivisionError):
        raise ValueError("Threshold must be between 1 and 100 percent.") from None
    if not Fraction(1, 100) <= target <= 1:
        raise ValueError("Threshold must be between 1 and 100 percent.")
    return target


def percentage(attended: int, conducted: int) -> float | None:
    validate(attended, conducted)
    return 100 * attended / conducted if conducted else None


def meets_target(attended: int, conducted: int, threshold: float = 75) -> bool | None:
    target = validate(attended, conducted, threshold=threshold)
    return Fraction(attended, conducted) >= target if conducted else None


def safe_bunks_now(attended: int, conducted: int, threshold: float = 75) -> int:
    target = validate(attended, conducted, threshold=threshold)
    return max(0, (Fraction(attended) / target).__floor__() - conducted)


@dataclass(frozen=True)
class Plan:
    required: int
    feasible: bool
    allowance: int | None
    minimum: float | None
    maximum: float | None


def plan(attended: int, conducted: int, remaining: int, threshold: float = 75) -> Plan:
    target = validate(attended, conducted, remaining, threshold)
    required = max(0, (target * (conducted + remaining) - attended).__ceil__())
    feasible = required <= remaining
    return Plan(required, feasible, remaining - required if feasible else None,
                percentage(attended, conducted + remaining),
                percentage(attended + remaining, conducted + remaining))


def projected(attended: int, conducted: int, remaining: int, future_attended: int) -> float | None:
    validate(attended, conducted, remaining)
    if isinstance(future_attended, bool) or not isinstance(future_attended, int) or not 0 <= future_attended <= remaining:
        raise ValueError("Planned attendance must be an integer between zero and remaining classes.")
    return percentage(attended + future_attended, conducted + remaining)


def certain_risk(attended: int, conducted: int, remaining: int, threshold: float = 75) -> float | None:
    target = validate(attended, conducted, remaining, threshold)
    total = conducted + remaining
    if total == 0:
        return None  # No classes: attendance is undefined.
    if Fraction(attended, total) >= target:
        return 0.0
    if Fraction(attended + remaining, total) < target:
        return 1.0
    return None
