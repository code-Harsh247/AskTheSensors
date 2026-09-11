"""Deterministic reasoning operators over the activity timeline (docs/TASKS.md
task 2B.2). Every number, interval, and verdict in an answer is computed here
in Python from the timeline; nothing downstream may invent one.

Negative findings ("never happened", "not prolonged") cite every observed
interval as the evidence that was examined, so they are grounded too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ats.aggregate import Interval, Timeline
from ats.evidence import describe, summarize, summarize_each
from ats.routing import OperatorCall
from ats.serialize import format_seconds as fs
from ats.signal import GRAVITY, MAJORITY, STILL_ACC_STD, STILL_GYRO_ENERGY, gravity_ok, recording_gravity
from ats.vocab import ACTIVE, PROLONGED_S, SEDENTARY, VIGOROUS, display

# Two totals closer than one window hop cannot be told apart at the
# timeline's resolution (docs/TASKS.md §0: 2.0 s hop), so they count as equal.
TIE_TOLERANCE_S = 2.0

Span = tuple[float, float]
Windows = Sequence[dict[str, Any]]


@dataclass(frozen=True)
class Finding:
    answer: str
    activity_event: str
    cited: tuple[Span, ...]
    explanation: str


def abstain(reason: str) -> Finding:
    return Finding("N/A", "N/A", (), reason)


def _spans(intervals: Sequence[Interval]) -> tuple[Span, ...]:
    return tuple(iv.as_tuple() for iv in intervals)


def _phrase(activity: str) -> str:
    return display(activity).lower()


def _bounds(iv: Interval) -> str:
    return f"{fs(iv.t_start)}-{fs(iv.t_end)} s"


def _no_data() -> Finding:
    return abstain("The recording contains no usable windows.")


def _absent(activity: str, timeline: Timeline, answer: str) -> Finding:
    observed = timeline.intervals
    if not observed:
        return _no_data()
    covered = sum(iv.duration for iv in observed)
    return Finding(
        answer,
        display(activity),
        _spans(observed),
        f"{display(activity)} does not appear in any of the {len(observed)} observed interval(s), "
        f"which together cover {fs(covered)} s of the recording; all of them are cited as the evidence examined.",
    )


def _at(timeline: Timeline, t: float) -> Interval | Finding:
    interval = timeline.interval_at(t)
    if interval is not None:
        return interval
    gap = timeline.gap_at(t)
    if gap is not None:
        return abstain(
            f"No sensor data exists between {fs(gap[0])} and {fs(gap[1])} s, so the activity at "
            f"{fs(t)} s cannot be determined from the signal."
        )
    span = timeline.span()
    if span is None:
        return _no_data()
    return abstain(f"{fs(t)} s lies outside the recorded span, {fs(span[0])} to {fs(span[1])} s.")


def _unsupported_time(call: OperatorCall) -> Finding:
    return abstain(
        f"The question restricts a {call.op} question to a particular time or threshold, which is not "
        "supported yet; answering over the whole recording instead would answer a different question."
    )


def _identify(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if call.time_s is None:
        activity = timeline.dominant_activity()
        if activity is None:
            return _no_data()
        intervals = timeline.intervals_of(activity)
        return Finding(
            display(activity),
            display(activity),
            _spans(intervals),
            f"{display(activity)} accounts for more of the recording than any other activity: "
            f"{fs(timeline.total_duration(activity))} s across {len(intervals)} interval(s). "
            + describe(summarize(windows, _spans(intervals))),
        )

    at = _at(timeline, call.time_s)
    if isinstance(at, Finding):
        return at
    return Finding(
        display(at.activity),
        display(at.activity),
        (at.as_tuple(),),
        f"{fs(call.time_s)} s falls inside the interval {_bounds(at)}, classified as {_phrase(at.activity)} "
        f"(mean confidence {at.mean_confidence:.2f}). " + describe(summarize(windows, (at.as_tuple(),))),
    )


def _verify(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    target = call.activities[0]
    if call.time_s is not None:
        at = _at(timeline, call.time_s)
        if isinstance(at, Finding):
            return at
        matches = at.activity == target
        return Finding(
            "Yes" if matches else "No",
            display(target),
            (at.as_tuple(),),
            f"At {fs(call.time_s)} s the user was {_phrase(at.activity)} (interval {_bounds(at)}, mean confidence "
            f"{at.mean_confidence:.2f}), which {'matches' if matches else 'is not'} {_phrase(target)}. "
            + describe(summarize(windows, (at.as_tuple(),))),
        )

    intervals = timeline.intervals_of(target)
    if not intervals:
        return _absent(target, timeline, "No")
    return Finding(
        "Yes",
        display(target),
        _spans(intervals),
        f"{display(target)} occurs in {len(intervals)} interval(s) totalling "
        f"{fs(timeline.total_duration(target))} s. " + describe(summarize(windows, _spans(intervals))),
    )


def _duration(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if call.time_s is not None:
        return _unsupported_time(call)
    activity = call.activities[0]
    intervals = timeline.intervals_of(activity)
    if not intervals:
        return _absent(activity, timeline, "0 seconds")
    total = timeline.total_duration(activity)
    parts = ", ".join(fs(iv.duration) for iv in intervals)
    return Finding(
        f"{fs(total)} seconds",
        display(activity),
        _spans(intervals),
        f"{display(activity)} was detected in {len(intervals)} separate interval(s) of {parts} s, "
        f"which sum to {fs(total)} s. " + describe(summarize(windows, _spans(intervals))),
    )


def _count(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if call.time_s is not None:
        return _unsupported_time(call)
    activity = call.activities[0]
    intervals = timeline.intervals_of(activity)
    if not intervals:
        return _absent(activity, timeline, "0 bouts")
    n = len(intervals)
    answer = "1 bout" if n == 1 else f"{n} separate bouts"
    return Finding(
        answer,
        display(activity),
        _spans(intervals),
        f"{display(activity)} occurs as {n} bout(s), each bounded by a different activity or by a gap in "
        f"the recording. " + describe(summarize(windows, _spans(intervals))),
    )


def _compare(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if call.time_s is not None:
        return _unsupported_time(call)
    a, b = call.activities[:2]
    total_a, total_b = timeline.total_duration(a), timeline.total_duration(b)
    cited = tuple(sorted(_spans(timeline.intervals_of(a)) + _spans(timeline.intervals_of(b))))
    if not cited:
        cited = _spans(timeline.intervals)
    if abs(total_a - total_b) <= TIE_TOLERANCE_S:
        verdict = "Equal"
        tail = "the difference is within one window hop, so they are equal at the timeline's resolution."
    else:
        winner = a if total_a > total_b else b
        verdict = display(winner)
        tail = f"{_phrase(winner)} exceeds the other by {fs(abs(total_a - total_b))} s."
    return Finding(
        verdict,
        f"{display(a)}, {display(b)}",
        cited,
        f"{display(a)} totalled {fs(total_a)} s across {timeline.count(a)} interval(s) and {_phrase(b)} "
        f"{fs(total_b)} s across {timeline.count(b)}; {tail}",
    )


def _onset(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    activity = call.activities[0]
    intervals = timeline.intervals_of(activity)
    if not intervals:
        return _absent(activity, timeline, "No")
    first = intervals[0]
    index = timeline.intervals.index(first)
    context = f"The first {_phrase(activity)} interval runs {_bounds(first)}."
    current = summarize(windows, (first.as_tuple(),))
    if index == 0:
        context += " It is the first interval in the recording, so the activity may already have been under way when recording began."
    else:
        previous = timeline.intervals[index - 1]
        if abs(previous.t_end - first.t_start) > 1e-6:
            context += f" It follows a gap in the recording that ends at {fs(first.t_start)} s, so the true onset may lie inside that gap."
        else:
            before = summarize(windows, (previous.as_tuple(),))
            context += f" It follows {_phrase(previous.activity)} ({_bounds(previous)})"
            if before is not None and current is not None:
                context += (
                    f"; across the transition the accelerometer-magnitude standard deviation goes from "
                    f"{before.acc_mag_std:.2f} to {current.acc_mag_std:.2f} m/s^2 and gyroscope energy from "
                    f"{before.gyro_energy:.3g} to {current.gyro_energy:.3g}."
                )
            else:
                context += "."
    return Finding(
        f"Yes, {_phrase(activity)} began at {fs(first.t_start)} seconds",
        f"Onset of {_phrase(activity)}",
        (first.as_tuple(),),
        f"{context} {describe(current)}",
    )


def _ground(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if not call.activities:
        return abstain("The question does not name an activity to locate in the signal.")
    activity = call.activities[0]
    intervals = timeline.intervals_of(activity)
    if not intervals:
        return _absent(activity, timeline, f"{display(activity)} not observed")
    confidence = sum(iv.mean_confidence for iv in intervals) / len(intervals)
    return Finding(
        display(activity),
        display(activity),
        _spans(intervals),
        f"{display(activity)} is supported by {len(intervals)} interval(s) totalling "
        f"{fs(timeline.total_duration(activity))} s, with mean classifier confidence {confidence:.2f}. "
        + describe(summarize(windows, _spans(intervals))),
    )


_STILL_RULE = (
    f"accelerometer-magnitude standard deviation at most {STILL_ACC_STD} m/s^2 and gyroscope energy at most "
    f"{STILL_GYRO_ENERGY}, as still as 95% of lying-down windows on the calibration subject"
)


def _moving(intervals: Sequence[Interval], windows: Windows) -> tuple[list[Interval], int]:
    """The intervals whose signal is mostly moving, and how many were set
    aside because it is mostly still: a movement claim cannot rest on a still
    signal, whatever the classifier called it. When the recording's units are
    off, stillness cannot be judged and nothing is set aside."""
    if not gravity_ok(windows):
        return list(intervals), 0
    summaries = summarize_each(windows, _spans(intervals))
    moving = [iv for iv, s in zip(intervals, summaries) if s is not None and s.still_share < MAJORITY]
    return moving, len(intervals) - len(moving)


def _sustained_stillness(timeline: Timeline, windows: Windows) -> Finding:
    """Rest judged from the signal alone: the longest run of adjacent
    intervals whose windows are mostly still, whatever posture the classifier
    gave each of them."""
    if not timeline.intervals:
        return _no_data()
    if not gravity_ok(windows):
        return abstain(
            f"This recording's median acceleration magnitude is {recording_gravity(windows):.2f} m/s^2, far from "
            f"gravity ({GRAVITY} m/s^2), so its units look wrong and stillness cannot be judged from the signal."
        )
    summaries = summarize_each(windows, _spans(timeline.intervals))
    stretches: list[list[Interval]] = []
    current: list[Interval] = []
    for interval, summary in zip(timeline.intervals, summaries):
        if summary is not None and summary.still_share >= MAJORITY:
            if current and abs(current[-1].t_end - interval.t_start) <= 1e-6:
                current.append(interval)
            else:
                if current:
                    stretches.append(current)
                current = [interval]
        elif current:
            stretches.append(current)
            current = []
    if current:
        stretches.append(current)

    if not stretches:
        return Finding(
            "Likely no",
            "Sustained stillness",
            _spans(timeline.intervals),
            f"None of the {len(timeline.intervals)} observed intervals shows a mostly still signal ({_STILL_RULE}); "
            "all are cited as the evidence examined.",
        )

    def length(stretch: list[Interval]) -> float:
        return stretch[-1].t_end - stretch[0].t_start

    longest = max(stretches, key=length)
    qualifying = [s for s in stretches if length(s) >= PROLONGED_S]
    cited = tuple(iv.as_tuple() for stretch in (qualifying or [longest]) for iv in stretch)
    postures = sorted({_phrase(iv.activity) for iv in longest})
    return Finding(
        "Likely yes" if qualifying else "Likely no",
        "Sustained stillness",
        cited,
        f"The longest stretch of mostly still signal runs {fs(longest[0].t_start)}-{fs(longest[-1].t_end)} s "
        f"({fs(length(longest))} s), {'at or above' if qualifying else 'below'} the {fs(PROLONGED_S)} s taken as "
        f"prolonged; the classifier called it {', '.join(postures)}. Stillness is judged from the signal "
        f"({_STILL_RULE}), so it holds whichever resting posture the classifier chose. "
        + describe(summarize(windows, cited)),
    )


def _prolonged(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if not call.activities:
        return _sustained_stillness(timeline, windows)
    activity = call.activities[0]
    intervals = timeline.intervals_of(activity)
    if not intervals:
        return _absent(activity, timeline, "Likely no")
    qualifying = [iv for iv in intervals if iv.duration >= PROLONGED_S]
    longest = max(intervals, key=lambda iv: iv.duration)
    cited = _spans(qualifying or intervals)
    return Finding(
        "Likely yes" if qualifying else "Likely no",
        f"Prolonged {_phrase(activity)}",
        cited,
        f"The longest continuous {_phrase(activity)} stretch runs {_bounds(longest)} ({fs(longest.duration)} s), "
        f"{'at or above' if qualifying else 'below'} the {fs(PROLONGED_S)} s taken as a prolonged period. "
        + describe(summarize(windows, cited)),
    )


def _wheeled(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    labelled = timeline.intervals_of("BICYCLING")
    moving, set_aside = _moving(labelled, windows)
    if not moving:
        if not labelled:
            found = _absent("BICYCLING", timeline, "No")
            return Finding(found.answer, "No wheeled or pedal-based movement observed", found.cited, found.explanation)
        return Finding(
            "No",
            "No wheeled or pedal-based movement observed",
            _spans(timeline.intervals),
            f"{len(labelled)} interval(s) were classified as bicycling, but the signal in each is mostly still "
            f"({_STILL_RULE}), which cannot support a claim of pedalling; every observed interval is cited as the "
            "evidence examined.",
        )
    note = f" {set_aside} bicycling interval(s) with a mostly still signal were set aside." if set_aside else ""
    return Finding(
        "Yes",
        "Consistent with cycling",
        _spans(moving),
        f"{len(moving)} interval(s) totalling {fs(sum(iv.duration for iv in moving))} s were classified as "
        f"bicycling and show a moving signal.{note} " + describe(summarize(windows, _spans(moving))),
    )


def _balance(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    sedentary = sum(timeline.total_duration(a) for a in SEDENTARY)
    active = sum(timeline.total_duration(a) for a in ACTIVE)
    if sedentary == 0 and active == 0:
        return abstain("The recording contains neither sedentary nor active intervals to weigh.")
    at_rest = sedentary > active
    group = SEDENTARY if at_rest else ACTIVE
    cited = _spans([iv for iv in timeline.intervals if iv.activity in group])
    light = timeline.total_duration("STANDING_MOVING")
    return Finding(
        "Mostly at rest" if at_rest else "Mostly active",
        "Overall activity balance",
        cited,
        f"Sedentary activities (lying down, sitting, standing in place) total {fs(sedentary)} s and active ones "
        f"(walking, running, bicycling) total {fs(active)} s; standing and moving ({fs(light)} s) is counted "
        f"in neither.",
    )


def _movement_extreme(lowest: bool) -> Callable[[OperatorCall, Timeline, Windows], Finding]:
    def operator(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
        summaries = summarize_each(windows, _spans(timeline.intervals))
        scored = [(iv, s) for iv, s in zip(timeline.intervals, summaries) if s is not None]
        if not scored:
            return _no_data()
        if lowest:
            chosen, stats = min(scored, key=lambda p: (p[1].acc_mag_std, -p[0].duration))
        else:
            chosen, stats = max(scored, key=lambda p: (p[1].acc_mag_std, p[0].duration))
        others = [s.acc_mag_std for iv, s in scored if iv is not chosen]
        comparison = (
            f", against {sum(others) / len(others):.2f} averaged over the other {len(others)}" if others else ""
        )
        word = "lowest" if lowest else "highest"
        return Finding(
            display(chosen.activity),
            f"{word.capitalize()}-movement stretch",
            (chosen.as_tuple(),),
            f"Of the {len(scored)} observed intervals, {_bounds(chosen)} ({_phrase(chosen.activity)}) has the {word} "
            f"mean accelerometer-magnitude standard deviation, {stats.acc_mag_std:.2f} m/s^2{comparison}. "
            + describe(stats),
        )

    return operator


def _strenuous(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    if call.time_s is not None:
        at = _at(timeline, call.time_s)
        if isinstance(at, Finding):
            return at
        stats = summarize(windows, (at.as_tuple(),))
        moving = stats is not None and stats.still_share < MAJORITY
        vigorous = at.activity in VIGOROUS and moving
        if vigorous:
            verdict = "vigorous activity with a moving signal"
        elif at.activity in VIGOROUS:
            verdict = f"vigorous by label, but its signal is mostly still ({_STILL_RULE}), which cannot support a claim of exertion"
        else:
            verdict = "not vigorous activity (running or bicycling)"
        return Finding(
            "Yes" if vigorous else "No",
            "Strenuous activity",
            (at.as_tuple(),),
            f"At {fs(call.time_s)} s the user was {_phrase(at.activity)} ({_bounds(at)}), which is {verdict}. "
            + describe(stats),
        )
    labelled = [iv for iv in timeline.intervals if iv.activity in VIGOROUS]
    moving, set_aside = _moving(labelled, windows)
    if not moving:
        if not timeline.intervals:
            return _no_data()
        extra = (
            f" {len(labelled)} running or bicycling interval(s) were set aside because their signal is mostly still "
            f"({_STILL_RULE})."
            if labelled
            else ""
        )
        return Finding(
            "No",
            "Strenuous activity",
            _spans(timeline.intervals),
            f"None of the {len(timeline.intervals)} observed intervals is running or bicycling with a moving signal; "
            f"all are cited as the evidence examined.{extra}",
        )
    note = f" {set_aside} running or bicycling interval(s) with a mostly still signal were set aside." if set_aside else ""
    return Finding(
        "Yes",
        "Strenuous activity",
        _spans(moving),
        f"{len(moving)} interval(s) of running or bicycling with a moving signal total "
        f"{fs(sum(iv.duration for iv in moving))} s.{note} " + describe(summarize(windows, _spans(moving))),
    )


_OPEN_WORLD: dict[str, Callable[[OperatorCall, Timeline, Windows], Finding]] = {
    "prolonged": _prolonged,
    "wheeled": _wheeled,
    "activity_balance": _balance,
    "least_movement": _movement_extreme(lowest=True),
    "most_movement": _movement_extreme(lowest=False),
    "strenuous": _strenuous,
}


def _open_world(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    handler = _OPEN_WORLD.get(call.predicate or "")
    if handler is None:
        return abstain(
            "This question does not match any reasoning pattern the system supports, and answering it would "
            "mean guessing beyond the recorded evidence."
        )
    return handler(call, timeline, windows)


_OPERATORS: dict[str, Callable[[OperatorCall, Timeline, Windows], Finding]] = {
    "identify": _identify,
    "verify": _verify,
    "duration": _duration,
    "count": _count,
    "compare": _compare,
    "onset": _onset,
    "ground": _ground,
    "open_world": _open_world,
}


def execute(call: OperatorCall, timeline: Timeline, windows: Windows) -> Finding:
    return _OPERATORS[call.op](call, timeline, windows)
