
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

MLB_STATS = "https://statsapi.mlb.com/api/v1"
LIVE_FEED = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

SAVANT_CSV = (
    "https://baseballsavant.mlb.com/statcast_search/csv?"
    "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&"
    "hfGT=R%7CPO%7CS%7C=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&"
    "opponent=&pitcher_throws=&batter_stands=&hfSA=&"
    "game_date_gt={start_dt}&game_date_lt={end_dt}&team=&position=&hfRO=&"
    "home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&"
    "group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&"
    "sort_order=desc&min_abs=0&type=details"
)

BIP_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "force_out",
    "grounded_into_double_play", "field_error", "double_play",
    "fielders_choice", "fielders_choice_out", "sac_fly", "sac_bunt",
    "triple_play"
}

PITCH_GROUPS = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curve",
    "KC": "Knuckle curve", "CH": "Changeup", "FS": "Splitter",
    "SV": "Slurve", "KN": "Knuckleball", "EP": "Eephus"
}

PARK_FACTORS = {
    "COL": 1.24, "CIN": 1.15, "PHI": 1.12, "NYY": 1.11, "BOS": 1.10,
    "ATL": 1.07, "MIL": 1.06, "HOU": 1.05, "ARI": 1.05, "CHC": 1.04,
    "LAA": 1.02, "KC": 1.00, "TEX": 1.00, "WSH": 0.99, "BAL": 0.99,
    "LAD": 0.98, "SD": 0.97, "STL": 0.97, "DET": 0.96, "PIT": 0.96,
    "SF": 0.94, "SEA": 0.90, "OAK": 0.96, "ATH": 0.96, "CLE": 0.95,
    "MIN": 0.98, "CWS": 1.00, "NYM": 0.97, "MIA": 0.94, "TB": 0.96,
    "TOR": 1.02
}

DEFAULT_WEIGHTS = {
    "contact_quality": 0.30,
    "pitcher_vulnerability": 0.25,
    "pitch_mix_matchup": 0.15,
    "park_environment": 0.15,
    "regression_due": 0.10,
    "market_value": 0.05,
}

# These are used only to validate roster eligibility. Confirmed batting orders
# always take precedence because a player listed in the official lineup is
# eligible for that game.
INACTIVE_TRANSACTION_TERMS = (
    "optioned", "reassigned", "injured list", "disabled list",
    "designated for assignment", "released", "suspended",
    "restricted list", "bereavement list", "paternity list",
    "family medical emergency list", "outrighted", "transferred to the 60-day",
    "placed on the 10-day", "placed on the 15-day", "placed on the 60-day",
)
ACTIVE_TRANSACTION_TERMS = (
    "recalled", "activated", "reinstated", "selected the contract",
    "contract selected", "purchased the contract", "returned from",
    "added to the active roster",
)


@dataclass
class GameContext:
    game_pk: int
    game_date: str
    away: str
    home: str
    venue: str
    status: str
    away_pitcher_id: int | None
    away_pitcher_name: str | None
    home_pitcher_id: int | None
    home_pitcher_name: str | None


def get_json(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed: {url}") from last_exc


def schedule_for_date(game_date: str) -> list[GameContext]:
    payload = get_json(
        f"{MLB_STATS}/schedule",
        {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team,venue",
        },
    )
    games: list[GameContext] = []
    for d in payload.get("dates", []):
        for g in d.get("games", []):
            teams = g["teams"]
            away_prob = teams["away"].get("probablePitcher") or {}
            home_prob = teams["home"].get("probablePitcher") or {}
            games.append(
                GameContext(
                    game_pk=g["gamePk"],
                    game_date=g["gameDate"],
                    away=teams["away"]["team"]["abbreviation"],
                    home=teams["home"]["team"]["abbreviation"],
                    venue=g.get("venue", {}).get("name", ""),
                    status=g.get("status", {}).get("detailedState", ""),
                    away_pitcher_id=away_prob.get("id"),
                    away_pitcher_name=away_prob.get("fullName"),
                    home_pitcher_id=home_prob.get("id"),
                    home_pitcher_name=home_prob.get("fullName"),
                )
            )
    return games


def game_lineups_and_rosters(
    game_pk: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Return confirmed lineups and game-level eligible hitter rosters.

    The live boxscore frequently exposes the game roster before the batting
    order is posted. That roster is preferred over the broader team endpoint
    because it reflects eligibility for the selected game.
    """
    payload = get_json(LIVE_FEED.format(game_pk=game_pk))
    box = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})
    lineups: list[list[dict]] = []
    rosters: list[list[dict]] = []

    for side in ("away", "home"):
        team_box = box.get(side, {}) or {}
        order = [int(pid) for pid in (team_box.get("battingOrder") or [])]
        players = team_box.get("players") or {}
        by_id: dict[int, dict] = {}

        for player_data in players.values():
            person = player_data.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            pid = int(pid)
            position = player_data.get("position") or {}
            position_type = str(position.get("type", ""))
            status = player_data.get("status") or {}
            status_text = " ".join(
                str(status.get(key, "")) for key in ("code", "description")
            ).lower()

            # Never discard an official batting-order player. Otherwise remove
            # explicit inactive statuses and pitcher-only entries.
            if pid not in order:
                if any(term in status_text for term in (
                    "injured", "disabled", "inactive", "suspended",
                    "restricted", "minors", "optioned",
                )):
                    continue
                if position_type == "Pitcher":
                    continue

            by_id[pid] = {
                "player_id": pid,
                "player": person.get("fullName", str(pid)),
                "lineup_spot": np.nan,
                "position": position.get("abbreviation"),
                "roster_source": "game_boxscore",
            }

        lineup: list[dict] = []
        for slot, pid in enumerate(order, start=1):
            row = by_id.get(pid)
            if row is None:
                player_data = players.get(f"ID{pid}", {}) or {}
                person = player_data.get("person") or {}
                position = player_data.get("position") or {}
                row = {
                    "player_id": pid,
                    "player": person.get("fullName", str(pid)),
                    "lineup_spot": slot,
                    "position": position.get("abbreviation"),
                    "roster_source": "confirmed_lineup",
                }
            else:
                row = dict(row)
                row["lineup_spot"] = slot
                row["roster_source"] = "confirmed_lineup"
            lineup.append(row)

        lineups.append(lineup)
        rosters.append(list(by_id.values()))

    return lineups[0], lineups[1], rosters[0], rosters[1]


def confirmed_lineup(game_pk: int) -> tuple[list[dict], list[dict]]:
    """Compatibility wrapper returning only the posted batting orders."""
    away, home, _, _ = game_lineups_and_rosters(game_pk)
    return away, home


def _entry_status_is_active(entry: dict) -> bool:
    status = entry.get("status") or {}
    text = " ".join(
        str(status.get(key, "")) for key in ("code", "description")
    ).lower()
    if not text.strip():
        return True
    return not any(term in text for term in (
        "injured", "disabled", "inactive", "suspended", "restricted",
        "minors", "optioned", "designated", "released",
    ))


def _people_current_team(person_ids: list[int]) -> dict[int, dict[str, Any]]:
    """Bulk player metadata used to catch optioned/minor-league players."""
    ids = sorted({int(pid) for pid in person_ids})
    if not ids:
        return {}
    try:
        payload = get_json(
            f"{MLB_STATS}/people",
            {
                "personIds": ",".join(str(pid) for pid in ids),
                "hydrate": "currentTeam",
            },
        )
    except Exception:
        return {}

    result: dict[int, dict[str, Any]] = {}
    for person in payload.get("people", []) or []:
        pid = person.get("id")
        if pid is None:
            continue
        current_team = person.get("currentTeam") or {}
        result[int(pid)] = {
            "active": person.get("active"),
            "current_team_id": current_team.get("id"),
        }
    return result


def _recent_transaction_states(
    team_id: int,
    game_date: str,
    person_ids: list[int],
    lookback_days: int = 90,
) -> dict[int, bool]:
    """Return latest known active/inactive state from MLB transactions.

    True means a recent activation/recall; False means a recent option, IL,
    DFA, release, suspension, or other inactive transaction. Unknown players
    are omitted so API gaps do not incorrectly remove them.
    """
    ids = {int(pid) for pid in person_ids}
    if not ids:
        return {}
    end_date = pd.Timestamp(game_date).date()
    start_date = end_date - timedelta(days=lookback_days)
    try:
        payload = get_json(
            f"{MLB_STATS}/transactions",
            {
                "teamId": int(team_id),
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "sportId": 1,
            },
        )
    except Exception:
        return {}

    latest: dict[int, tuple[pd.Timestamp, bool]] = {}
    for tx in payload.get("transactions", []) or []:
        person = tx.get("person") or {}
        pid = person.get("id")
        if pid is None or int(pid) not in ids:
            continue
        text = " ".join(
            str(tx.get(key, ""))
            for key in ("typeCode", "typeDesc", "description")
        ).lower()

        state: bool | None = None
        if any(term in text for term in ACTIVE_TRANSACTION_TERMS):
            state = True
        elif any(term in text for term in INACTIVE_TRANSACTION_TERMS):
            state = False
        if state is None:
            continue

        tx_date = pd.to_datetime(
            tx.get("effectiveDate") or tx.get("date"), errors="coerce"
        )
        if pd.isna(tx_date):
            tx_date = pd.Timestamp.min
        current = latest.get(int(pid))
        if current is None or tx_date >= current[0]:
            latest[int(pid)] = (tx_date, state)

    return {pid: state for pid, (_, state) in latest.items()}


def fallback_roster(team_id: int, game_date: str) -> list[dict]:
    """Validated active MLB hitter roster for games without a posted lineup."""
    payload = get_json(
        f"{MLB_STATS}/teams/{team_id}/roster",
        {
            "rosterType": "active",
            "date": game_date,
            "hydrate": "person(currentTeam)",
        },
    )
    entries = [
        x for x in (payload.get("roster", []) or [])
        if x.get("position", {}).get("type") != "Pitcher"
        and _entry_status_is_active(x)
    ]
    if not entries:
        return []

    person_ids = [int(x["person"]["id"]) for x in entries]

    # Current-team and transaction checks are appropriate for a live slate.
    # Historical backtests continue to rely on the date-specific roster API.
    selected_date = date.fromisoformat(game_date)
    near_current = abs((selected_date - date.today()).days) <= 2
    people = _people_current_team(person_ids) if near_current else {}
    transactions = (
        _recent_transaction_states(team_id, game_date, person_ids)
        if near_current else {}
    )

    roster: list[dict] = []
    for entry in entries:
        pid = int(entry["person"]["id"])
        meta = people.get(pid, {})
        current_team_id = meta.get("current_team_id")
        explicitly_active = meta.get("active")

        if explicitly_active is False:
            continue
        if current_team_id is not None and int(current_team_id) != int(team_id):
            continue
        if transactions.get(pid) is False:
            continue

        roster.append({
            "player_id": pid,
            "player": entry["person"].get("fullName", str(pid)),
            "lineup_spot": np.nan,
            "position": entry.get("position", {}).get("abbreviation"),
            "roster_source": "validated_active_roster",
        })
    return roster

def team_id_map() -> dict[str, int]:
    payload = get_json(f"{MLB_STATS}/teams", {"sportId": 1})
    return {
        x["abbreviation"]: int(x["id"])
        for x in payload.get("teams", [])
    }


def historical_bvp_for_pitcher(
    batter_ids: list[int],
    pitcher_id: int | None,
) -> dict[int, dict[str, float]]:
    """Fetch career batter-vs-pitcher totals in one request per starter.

    BvP is retained as a small, sample-regressed adjustment. It supplements,
    rather than replaces, the last-10-game Statcast profile.
    """
    if not pitcher_id or not batter_ids:
        return {}
    try:
        payload = get_json(
            f"{MLB_STATS}/people",
            {
                "personIds": ",".join(
                    str(int(pid)) for pid in sorted(set(batter_ids))
                ),
                "hydrate": (
                    "stats(group=[hitting],type=[vsPlayer],"
                    f"opposingPlayerId={int(pitcher_id)},sportId=1)"
                ),
            },
        )
    except Exception:
        return {}

    result: dict[int, dict[str, float]] = {}
    for person in payload.get("people", []) or []:
        pid = person.get("id")
        if pid is None:
            continue

        candidates: list[dict[str, Any]] = []
        for block in person.get("stats", []) or []:
            for split in block.get("splits", []) or []:
                stat = split.get("stat") or {}
                if stat:
                    candidates.append(stat)

        # The hydration can expose season and total splits. The largest PA
        # sample is the career matchup and is the appropriate historical input.
        def number(value: Any) -> float:
            parsed = pd.to_numeric(value, errors="coerce")
            return 0.0 if pd.isna(parsed) else float(parsed)

        def pa_value(stat: dict[str, Any]) -> float:
            return number(stat.get("plateAppearances"))

        stat = max(candidates, key=pa_value) if candidates else {}
        pa = number(stat.get("plateAppearances"))
        ab = number(stat.get("atBats"))
        hits = number(stat.get("hits"))
        hr = number(stat.get("homeRuns"))
        doubles = number(stat.get("doubles"))
        triples = number(stat.get("triples"))
        walks = number(stat.get("baseOnBalls"))
        hbp = number(stat.get("hitByPitch"))
        total_bases = hits + doubles + 2 * triples + 3 * hr
        avg = hits / ab if ab else np.nan
        slg = total_bases / ab if ab else np.nan
        obp_den = ab + walks + hbp
        obp = (hits + walks + hbp) / obp_den if obp_den else np.nan
        result[int(pid)] = {
            "BvP_PA": pa,
            "BvP_AB": ab,
            "BvP_H": hits,
            "BvP_HR": hr,
            "BvP_AVG": avg,
            "BvP_OBP": obp,
            "BvP_SLG": slg,
        }
    return result


def _coerce_statcast_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "game_pk", "batter", "pitcher", "launch_speed", "launch_angle",
        "hit_distance_sc", "hc_x", "hc_y", "bat_score", "post_bat_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _fetch_savant_day(day: str, probable_pitcher_ids: set[int]) -> pd.DataFrame:
    """
    Fetch one day directly from Baseball Savant and immediately discard
    unnecessary pitch rows. We retain:
      - terminal plate-appearance rows,
      - all measured batted balls,
      - all pitches thrown by today's probable starters.
    """
    url = SAVANT_CSV.format(start_dt=day, end_dt=day)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
            "AppleWebKit/605.1.15 Safari/604.1"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    text = response.text
    if not text.strip():
        return pd.DataFrame()
    if text.lstrip().lower().startswith("<!doctype html") or "<html" in text[:300].lower():
        raise RuntimeError(
            f"Baseball Savant returned an HTML page instead of CSV for {day}."
        )

    daily = pd.read_csv(io.StringIO(text), low_memory=False)
    daily.columns = [str(c).strip() for c in daily.columns]

    needed = [
        "game_date", "game_pk", "batter", "pitcher", "stand", "p_throws", "events",
        "pitch_type", "launch_speed", "launch_angle", "hit_distance_sc",
        "hc_x", "hc_y", "bat_score", "post_bat_score"
    ]
    for col in needed:
        if col not in daily.columns:
            daily[col] = np.nan
    daily = daily[needed].copy()
    daily = _coerce_statcast_types(daily)

    terminal = daily["events"].notna()
    batted_ball = daily["launch_speed"].notna() & daily["launch_angle"].notna()
    starter_pitch = daily["pitcher"].isin(probable_pitcher_ids)
    daily = daily.loc[terminal | batted_ball | starter_pitch].copy()
    return daily


def pull_statcast(
    start_dt: str,
    end_dt: str,
    probable_pitcher_ids: set[int] | None = None,
) -> pd.DataFrame:
    """
    Direct Baseball Savant CSV ingestion.

    The query is split into one-day requests because Savant caps large result
    sets. Each day is filtered before concatenation, keeping memory stable on
    Streamlit Community Cloud.
    """
    pitcher_ids = probable_pitcher_ids or set()
    days = pd.date_range(start=start_dt, end=end_dt, freq="D")
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for idx, timestamp in enumerate(days, start=1):
        day = timestamp.strftime("%Y-%m-%d")
        print(f"Statcast day {idx}/{len(days)}: {day}", flush=True)
        try:
            daily = _fetch_savant_day(day, pitcher_ids)
        except Exception as exc:
            failures.append(f"{day}: {type(exc).__name__}: {exc}")
            continue
        if not daily.empty:
            frames.append(daily)
        gc.collect()

    if not frames:
        detail = "\n".join(failures[-5:])
        raise RuntimeError(
            "Baseball Savant returned no usable Statcast rows."
            + (f"\nRecent request errors:\n{detail}" if detail else "")
        )

    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    if len(failures) > max(3, len(days) // 4):
        detail = "\n".join(failures[-8:])
        raise RuntimeError(
            f"Too many Baseball Savant date requests failed "
            f"({len(failures)} of {len(days)}).\n{detail}"
        )

    return combined


def batting_side(row: pd.Series) -> str:
    stand = str(row.get("stand", "")).upper()
    return stand if stand in {"L", "R"} else "U"


def is_bbe(df: pd.DataFrame) -> pd.Series:
    return (
        df["launch_speed"].notna()
        & df["launch_angle"].notna()
        & df["events"].fillna("").isin(BIP_EVENTS)
    )


def is_barrel_row(ev: float, la: float) -> bool:
    """
    Public barrel approximation using the MLB barrel window:
    minimum 98 mph; launch-angle window expands with EV.
    This is intentionally labeled an approximation.
    """
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return False
    low = max(8, 26 - (ev - 98))
    high = min(50, 30 + 2 * (ev - 98))
    return low <= la <= high


def expected_hr_proxy(ev: float, la: float, distance: float | None) -> float:
    """
    Transparent custom xHR proxy, not MLB's official xHR.
    Uses EV, launch angle, and projected distance when available.
    """
    if pd.isna(ev) or pd.isna(la):
        return 0.0
    ev_component = 1 / (1 + math.exp(-(ev - 101.5) / 3.0))
    angle_component = math.exp(-((la - 28.0) ** 2) / (2 * 10.0 ** 2))
    if distance is None or pd.isna(distance):
        dist_component = 0.45
    else:
        dist_component = 1 / (1 + math.exp(-(distance - 385.0) / 16.0))
    return float(np.clip(0.45 * ev_component + 0.30 * angle_component + 0.25 * dist_component, 0, 1))


def last_n_games_for_batter(df: pd.DataFrame, batter_id: int, n: int = 10) -> pd.DataFrame:
    p = df[df["batter"] == batter_id].copy()
    dates = sorted(p["game_date"].dropna().dt.normalize().unique(), reverse=True)[:n]
    return p[p["game_date"].dt.normalize().isin(dates)].copy()


def count_runs_rbi_hits(p: pd.DataFrame) -> dict[str, float]:
    pa = p[p["events"].notna()].copy()
    hits = pa["events"].isin(["single", "double", "triple", "home_run"]).sum()
    hrs = (pa["events"] == "home_run").sum()
    doubles = (pa["events"] == "double").sum()
    triples = (pa["events"] == "triple").sum()
    walks = pa["events"].isin(["walk", "intent_walk", "hit_by_pitch"]).sum()
    ab = (~pa["events"].isin([
        "walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "catcher_interf"
    ])).sum()
    avg = hits / ab if ab else np.nan
    tb = hits + doubles + 2 * triples + 3 * hrs
    rbi = pd.to_numeric(pa.get("post_bat_score", 0), errors="coerce").fillna(0).sub(
        pd.to_numeric(pa.get("bat_score", 0), errors="coerce").fillna(0)
    ).clip(lower=0).sum()
    # Runs scored cannot be perfectly reconstructed from batter-only rows.
    # Count from runner movements when available; otherwise leave blank.
    runs = np.nan
    return {
        "G": int(pa["game_date"].dt.normalize().nunique()),
        "PA": int(len(pa)),
        "AB": int(ab),
        "AVG": avg,
        "H": int(hits),
        "HR": int(hrs),
        "R": runs,
        "RBI": float(rbi),
        "TB": int(tb),
        "BB_HBP": int(walks),
    }


def batted_ball_metrics(p: pd.DataFrame) -> dict[str, float]:
    b = p[is_bbe(p)].copy()
    if b.empty:
        return {k: np.nan for k in [
            "BBE", "Avg_EV", "EV90", "Max_EV", "HH_95", "HH_pct",
            "EV_100_plus", "EV_100_plus_outs", "Barrels_approx", "Barrel_pct_approx",
            "Avg_LA", "SweetSpot", "SweetSpot_pct", "PullAir", "PullAir_pct",
            "Fly_350_plus", "Fly_375_plus", "Out_380_400", "Near_HR",
            "xHR_proxy", "xHR_minus_HR"
        ]}
    b["launch_speed"] = pd.to_numeric(b["launch_speed"], errors="coerce")
    b["launch_angle"] = pd.to_numeric(b["launch_angle"], errors="coerce")
    b["hit_distance_sc"] = pd.to_numeric(b.get("hit_distance_sc"), errors="coerce")
    b["is_hit"] = b["events"].isin(["single", "double", "triple", "home_run"])
    b["is_out"] = ~b["is_hit"]
    b["barrel_approx"] = [
        is_barrel_row(ev, la) for ev, la in zip(b["launch_speed"], b["launch_angle"])
    ]
    b["xhr_proxy"] = [
        expected_hr_proxy(ev, la, dist)
        for ev, la, dist in zip(b["launch_speed"], b["launch_angle"], b["hit_distance_sc"])
    ]
    sweet = b["launch_angle"].between(8, 32, inclusive="both")
    air = b["launch_angle"] >= 10
    pull = (
        ((b["stand"] == "R") & (b["hc_x"] < 125))
        | ((b["stand"] == "L") & (b["hc_x"] > 125))
    )
    pull_air = air & pull
    fly_350 = (b["launch_angle"] >= 15) & (b["hit_distance_sc"] >= 350)
    fly_375 = (b["launch_angle"] >= 15) & (b["hit_distance_sc"] >= 375)
    out_380_400 = b["is_out"] & b["hit_distance_sc"].between(380, 400, inclusive="both")
    near_hr = (
        (b["is_out"] & (b["hit_distance_sc"] >= 375))
        | (b["is_out"] & (b["launch_speed"] >= 100) & b["launch_angle"].between(18, 36))
    )
    actual_hr = int((b["events"] == "home_run").sum())
    return {
        "BBE": int(len(b)),
        "Avg_EV": float(b["launch_speed"].mean()),
        "EV90": float(b["launch_speed"].quantile(0.90)),
        "Max_EV": float(b["launch_speed"].max()),
        "HH_95": int((b["launch_speed"] >= 95).sum()),
        "HH_pct": float((b["launch_speed"] >= 95).mean()),
        "EV_100_plus": int((b["launch_speed"] >= 100).sum()),
        "EV_100_plus_outs": int(((b["launch_speed"] >= 100) & b["is_out"]).sum()),
        "Barrels_approx": int(b["barrel_approx"].sum()),
        "Barrel_pct_approx": float(b["barrel_approx"].mean()),
        "Avg_LA": float(b["launch_angle"].mean()),
        "SweetSpot": int(sweet.sum()),
        "SweetSpot_pct": float(sweet.mean()),
        "PullAir": int(pull_air.sum()),
        "PullAir_pct": float(pull_air.mean()),
        "Fly_350_plus": int(fly_350.sum()),
        "Fly_375_plus": int(fly_375.sum()),
        "Out_380_400": int(out_380_400.sum()),
        "Near_HR": int(near_hr.sum()),
        "xHR_proxy": float(b["xhr_proxy"].sum()),
        "xHR_minus_HR": float(b["xhr_proxy"].sum() - actual_hr),
    }


def pitcher_vulnerability(df: pd.DataFrame, pitcher_id: int | None, batter_stand: str) -> dict[str, float]:
    if not pitcher_id:
        return {
            "Pitcher_BBE": np.nan, "Pitcher_HR": np.nan, "Pitcher_HR_pct": np.nan,
            "Pitcher_HH_pct": np.nan, "Pitcher_Barrel_pct_approx": np.nan,
            "Pitcher_Avg_EV": np.nan,
        }
    p = df[df["pitcher"] == pitcher_id].copy()
    if batter_stand in {"L", "R"}:
        p = p[p["stand"] == batter_stand]
    b = p[is_bbe(p)].copy()
    if b.empty:
        return {
            "Pitcher_BBE": 0, "Pitcher_HR": 0, "Pitcher_HR_pct": np.nan,
            "Pitcher_HH_pct": np.nan, "Pitcher_Barrel_pct_approx": np.nan,
            "Pitcher_Avg_EV": np.nan,
        }
    b["barrel_approx"] = [
        is_barrel_row(ev, la)
        for ev, la in zip(
            pd.to_numeric(b["launch_speed"], errors="coerce"),
            pd.to_numeric(b["launch_angle"], errors="coerce"),
        )
    ]
    return {
        "Pitcher_BBE": int(len(b)),
        "Pitcher_HR": int((b["events"] == "home_run").sum()),
        "Pitcher_HR_pct": float((b["events"] == "home_run").mean()),
        "Pitcher_HH_pct": float((pd.to_numeric(b["launch_speed"], errors="coerce") >= 95).mean()),
        "Pitcher_Barrel_pct_approx": float(b["barrel_approx"].mean()),
        "Pitcher_Avg_EV": float(pd.to_numeric(b["launch_speed"], errors="coerce").mean()),
    }


def pitch_mix_matchup_score(batter_rows: pd.DataFrame, pitcher_rows: pd.DataFrame) -> float:
    """
    Score 0-100 using batter damage on the pitcher's most-used pitch types.
    This is a transparent matchup score, not a proprietary projection.
    """
    pitcher_usage = pitcher_rows["pitch_type"].value_counts(normalize=True).head(4)
    if pitcher_usage.empty:
        return 50.0
    scores = []
    weights = []
    for pitch_type, usage in pitcher_usage.items():
        b = batter_rows[(batter_rows["pitch_type"] == pitch_type) & is_bbe(batter_rows)].copy()
        if b.empty:
            score = 50.0
        else:
            ev = pd.to_numeric(b["launch_speed"], errors="coerce").mean()
            hh = (pd.to_numeric(b["launch_speed"], errors="coerce") >= 95).mean()
            score = np.clip((ev - 82) * 4.0 + hh * 35, 0, 100)
        scores.append(score)
        weights.append(float(usage))
    return float(np.average(scores, weights=weights))


def normalize_series(s: pd.Series, low: float = 0, high: float = 100) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() <= 1 or s.max() == s.min():
        return pd.Series(np.where(s.notna(), 50.0, np.nan), index=s.index)
    return low + (s - s.min()) * (high - low) / (s.max() - s.min())


def add_model_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    contact_raw = (
        out["Avg_EV"].fillna(out["Avg_EV"].median()) * 0.18
        + out["EV90"].fillna(out["EV90"].median()) * 0.16
        + out["Max_EV"].fillna(out["Max_EV"].median()) * 0.10
        + out["HH_pct"].fillna(0) * 100 * 0.16
        + out["Barrel_pct_approx"].fillna(0) * 100 * 0.16
        + out["PullAir_pct"].fillna(0) * 100 * 0.10
        + out["Fly_375_plus"].fillna(0) * 3.5
        + out["Near_HR"].fillna(0) * 2.5
    )
    pitcher_raw = (
        out["Pitcher_HR_pct"].fillna(0) * 100 * 0.35
        + out["Pitcher_HH_pct"].fillna(0) * 100 * 0.30
        + out["Pitcher_Barrel_pct_approx"].fillna(0) * 100 * 0.25
        + out["Pitcher_Avg_EV"].fillna(85) * 0.10
    )
    due_raw = (
        out["xHR_minus_HR"].fillna(0) * 20
        + out["EV_100_plus_outs"].fillna(0) * 3
        + out["Out_380_400"].fillna(0) * 5
    )
    out["Contact_Score"] = normalize_series(contact_raw)
    out["Pitcher_Vuln_Score"] = normalize_series(pitcher_raw)
    out["Pitch_Mix_Score"] = out["Pitch_Mix_Score"].fillna(50).clip(0, 100)
    out["Park_Env_Score"] = normalize_series(out["Park_Factor"] * out["Weather_Factor"])
    out["Due_Score"] = normalize_series(due_raw)
    out["Market_Value_Score"] = out["Market_Value_Score"].fillna(50).clip(0, 100)

    out["Model_Score"] = (
        out["Contact_Score"] * weights["contact_quality"]
        + out["Pitcher_Vuln_Score"] * weights["pitcher_vulnerability"]
        + out["Pitch_Mix_Score"] * weights["pitch_mix_matchup"]
        + out["Park_Env_Score"] * weights["park_environment"]
        + out["Due_Score"] * weights["regression_due"]
        + out["Market_Value_Score"] * weights["market_value"]
    )
    out["Qualifying_Power_Signals"] = (
        (out["Barrels_approx"].fillna(0) >= 2).astype(int)
        + (out["Max_EV"].fillna(0) >= 105).astype(int)
        + (out["EV_100_plus"].fillna(0) >= 2).astype(int)
        + (out["Fly_375_plus"].fillna(0) >= 1).astype(int)
        + (out["xHR_minus_HR"].fillna(0) > 0.35).astype(int)
        + (out["PullAir_pct"].fillna(0) >= 0.20).astype(int)
    )
    out["Core_HR_Eligible"] = (
        (pd.to_numeric(out["Model_Score"], errors="coerce").fillna(0) >= 60)
        & (pd.to_numeric(out["Qualifying_Power_Signals"], errors="coerce").fillna(0) >= 3)
        & out["opposing_pitcher_id"].notna()
    )
    return out.sort_values(["Model_Score", "Qualifying_Power_Signals"], ascending=False)


def _fixed_scale(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if high <= low:
        return pd.Series(50.0, index=values.index)
    return ((values - low) / (high - low) * 100.0).clip(0, 100)


def add_v4_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transparent V4 engine:
      50% recent power (last 10 games)
      25% pitcher leak by batter side
      15% due indicators
      10% environment
      plus a sample-regressed historical BvP adjustment capped at +/-5 points
    """
    out = df.copy()

    def n(col: str, default: float = 0.0) -> pd.Series:
        if col not in out.columns:
            return pd.Series(default, index=out.index, dtype="float64")
        return pd.to_numeric(out[col], errors="coerce").fillna(default)

    # Recent power profile.
    recent_hr = n("HR")
    barrels = n("Barrels_approx")
    ev100 = n("EV_100_plus")
    deep375 = n("Fly_375_plus")
    deep400 = n("Out_380_400")
    hh = n("HH_pct")
    pull_air = n("PullAir_pct")
    sweet = n("SweetSpot_pct")
    ev90 = n("EV90", 95.0)
    max_ev = n("Max_EV", 95.0)

    out["Recent_Power"] = (
        _fixed_scale(recent_hr, 0, 5) * 0.18
        + _fixed_scale(barrels, 0, 7) * 0.20
        + _fixed_scale(ev100, 1, 16) * 0.14
        + _fixed_scale(deep375, 0, 7) * 0.12
        + _fixed_scale(deep400, 0, 4) * 0.05
        + _fixed_scale(hh, 0.25, 0.65) * 0.10
        + _fixed_scale(pull_air, 0.08, 0.38) * 0.09
        + _fixed_scale(sweet, 0.18, 0.48) * 0.05
        + _fixed_scale(ev90, 96, 111) * 0.04
        + _fixed_scale(max_ev, 101, 116) * 0.03
    ).clip(0, 100)

    # HR-quality contact indicators inspired by the uploaded cheat sheet.
    launch_angle = n("Avg_LA")
    out["TANKS"] = (
        (barrels * 0.60 + n("EV_100_plus") * 0.25 + deep375 * 0.15)
        .round()
        .astype(int)
    )
    out["Porch_Shots"] = (
        (n("Near_HR") + n("Out_380_400") + (pull_air >= 0.20).astype(int))
        .round()
        .astype(int)
    )

    # Pitcher leak with strong sample regression.
    p_bbe = n("Pitcher_BBE")
    p_hr = n("Pitcher_HR_pct")
    p_barrel = n("Pitcher_Barrel_pct_approx")
    p_hh = n("Pitcher_HH_pct")
    p_ev = n("Pitcher_Avg_EV", 87.0)

    reliability = (p_bbe / 50.0).clip(0, 1)
    raw_pitcher = (
        _fixed_scale(p_hr, 0.01, 0.09) * 0.40
        + _fixed_scale(p_barrel, 0.03, 0.16) * 0.25
        + _fixed_scale(p_hh, 0.28, 0.55) * 0.20
        + _fixed_scale(p_ev, 84, 93) * 0.15
    )
    out["Pitcher_Leak"] = (50 + (raw_pitcher - 50) * reliability).clip(0, 100)
    out["Pitcher_Sample"] = p_bbe.astype(int)
    out["Pitcher_Sample_Flag"] = ""
    out.loc[p_bbe < 25, "Pitcher_Sample_Flag"] = "⚠ SMALL SAMPLE"

    # Due meter.
    due_raw = (
        _fixed_scale(n("EV_100_plus_outs"), 0, 8) * 0.30
        + _fixed_scale(n("Out_380_400"), 0, 4) * 0.25
        + _fixed_scale(n("Near_HR"), 0, 5) * 0.25
        + _fixed_scale(n("xHR_minus_HR"), 0, 2.5) * 0.20
    )
    out["Due_Score_V4"] = due_raw.clip(0, 100)
    out["Due_Meter"] = "🟡 NEUTRAL"
    out.loc[(due_raw >= 65) & (recent_hr <= 2), "Due_Meter"] = "🟢 DUE"
    out.loc[(recent_hr >= 3), "Due_Meter"] = "🔥 PRODUCING"

    # Environment capped so it cannot dominate.
    park = n("Park_Factor", 1.0)
    weather = n("Weather_Factor", 1.0)
    out["Environment"] = (
        _fixed_scale(park, 0.90, 1.18) * 0.70
        + _fixed_scale(weather, 0.94, 1.06) * 0.30
    ).clip(0, 100)

    # Historical BvP. Small samples are strongly regressed to neutral.
    bvp_pa = n("BvP_PA")
    bvp_hr = n("BvP_HR")
    bvp_avg = n("BvP_AVG", 0.250)
    bvp_slg = n("BvP_SLG", 0.400)
    bvp_hr_rate = (bvp_hr / bvp_pa.replace(0, np.nan)).fillna(0)
    bvp_raw = (
        _fixed_scale(bvp_hr_rate, 0.00, 0.12) * 0.40
        + _fixed_scale(bvp_slg, 0.250, 0.850) * 0.35
        + _fixed_scale(bvp_avg, 0.150, 0.400) * 0.25
    )
    bvp_reliability = (bvp_pa / 30.0).clip(0, 1)
    out["BvP_Score"] = (50 + (bvp_raw - 50) * bvp_reliability).clip(0, 100)
    out["BvP_Adjustment"] = ((out["BvP_Score"] - 50) * 0.10).clip(-5, 5)
    out["BvP_Sample_Flag"] = ""
    out.loc[bvp_pa == 0, "BvP_Sample_Flag"] = "NO HISTORY"
    out.loc[(bvp_pa > 0) & (bvp_pa < 10), "BvP_Sample_Flag"] = "⚠ SMALL BvP"

    # Preserve the original V4 weighting and apply BvP only as a bounded overlay.
    out["Base_Power_Index"] = (
        out["Recent_Power"] * 0.50
        + out["Pitcher_Leak"] * 0.25
        + out["Due_Score_V4"] * 0.15
        + out["Environment"] * 0.10
    ).clip(0, 100)
    out["Power_Index"] = (out["Base_Power_Index"] + out["BvP_Adjustment"]).clip(0, 100)

    # Recalibrated display-only estimate: roughly 4%-30%, with a 58 index
    # near 24%. This changes the display scale, not the underlying rankings.
    likelihood = 0.035 + 0.265 / (1 + np.exp(-(out["Power_Index"] - 47.0) / 9.0))
    out["HR_Likelihood_pct"] = (likelihood * 100).round().astype(int)
    out["HR_Likelihood"] = out["HR_Likelihood_pct"].astype(str) + "%"

    # Conviction and best market.
    out["Conviction"] = "HR LEAN"
    out.loc[out["Power_Index"] >= 62, "Conviction"] = "🔥 SOLID"
    out.loc[out["Power_Index"] >= 72, "Conviction"] = "🔥🔥 FIRED"
    out.loc[(out["Due_Score_V4"] >= 70) & (out["Power_Index"] < 72), "Conviction"] = "🎯 DUE"

    out["Best_Look"] = "HR LEAN"
    out.loc[(out["Power_Index"] >= 68) & (out["Recent_Power"] >= 65), "Best_Look"] = "HR"
    out.loc[(out["Due_Score_V4"] >= 70) & (out["Recent_Power"] >= 50), "Best_Look"] = "LONGSHOT HR"
    out.loc[(pull_air < 0.12) & (n("AVG") >= 0.275), "Best_Look"] = "2+ TB / HRR"
    out.loc[(out["Recent_Power"] < 35) & (out["Pitcher_Leak"] < 45), "Best_Look"] = "SKIP HR"

    # Player display + platoon.
    batter_side = out.get("batter_profile_side", pd.Series("U", index=out.index)).fillna("U").astype(str).str.upper()
    pitcher_side = out.get("pitcher_throws", pd.Series("U", index=out.index)).fillna("U").astype(str).str.upper()
    out["Platoon_Marker"] = ""
    out.loc[batter_side.eq("S"), "Platoon_Marker"] = "**"
    opposite = ((batter_side.eq("L") & pitcher_side.eq("R")) | (batter_side.eq("R") & pitcher_side.eq("L")))
    out.loc[opposite & ~batter_side.eq("S"), "Platoon_Marker"] = "*"

    out["Hot_Symbol"] = ""
    out.loc[out["Recent_Power"].between(52, 64.999), "Hot_Symbol"] = "🔥"
    out.loc[out["Recent_Power"].between(65, 77.999), "Hot_Symbol"] = "🔥🔥"
    out.loc[out["Recent_Power"] >= 78, "Hot_Symbol"] = "🔥🔥🔥"
    out["Player_Display"] = out["player"].fillna("").astype(str) + out["Platoon_Marker"]

    # Rank overall and within each game.
    out = out.sort_values(
        ["Power_Index", "Recent_Power", "Pitcher_Leak"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    out["Overall_Rank"] = np.arange(1, len(out) + 1)

    game_keys = [c for c in ["game_pk"] if c in out.columns]
    if game_keys:
        out["Game_Rank"] = (
            out.groupby(game_keys, dropna=False)["Power_Index"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        out["Game_Rank"] = out["Overall_Rank"]

    out["Top_4_Per_Game"] = out["Game_Rank"] <= 4
    out["Top_10_Overall"] = out["Overall_Rank"] <= 10

    return out


def people_handedness(person_ids: list[int | None]) -> dict[int, dict[str, str]]:
    """Return MLB-listed batting and throwing sides. Empty output is a safe fallback."""
    ids = sorted({int(pid) for pid in person_ids if pid is not None})
    if not ids:
        return {}
    try:
        payload = get_json(
            f"{MLB_STATS}/people",
            {"personIds": ",".join(str(pid) for pid in ids)},
        )
    except Exception:
        return {}

    result: dict[int, dict[str, str]] = {}
    for person in payload.get("people", []):
        pid = person.get("id")
        if pid is None:
            continue
        bat = str((person.get("batSide") or {}).get("code", "U")).upper()
        throw = str((person.get("pitchHand") or {}).get("code", "U")).upper()
        result[int(pid)] = {
            "bat": bat if bat in {"L", "R", "S"} else "U",
            "throw": throw if throw in {"L", "R"} else "U",
        }
    return result


def add_display_symbols(board: pd.DataFrame) -> pd.DataFrame:
    """
    Add visual-only fields. No score, sorting, Core HR, or ranking value changes.
    """
    out = board.copy()

    recent_hr = pd.to_numeric(out.get("HR"), errors="coerce").fillna(0)
    barrels = pd.to_numeric(out.get("Barrels_approx"), errors="coerce").fillna(0)
    ev100 = pd.to_numeric(out.get("EV_100_plus"), errors="coerce").fillna(0)
    deep_375 = pd.to_numeric(out.get("Fly_375_plus"), errors="coerce").fillna(0)
    near_hr = pd.to_numeric(out.get("Near_HR"), errors="coerce").fillna(0)

    out["Hot_Points"] = (
        (recent_hr >= 1).astype(int)
        + (recent_hr >= 3).astype(int)
        + (barrels >= 3).astype(int)
        + (barrels >= 5).astype(int)
        + (ev100 >= 8).astype(int)
        + (deep_375 >= 3).astype(int)
        + (near_hr >= 2).astype(int)
    )

    out["Hot_Symbol"] = ""
    out.loc[out["Hot_Points"].between(2, 3), "Hot_Symbol"] = "🔥"
    out.loc[out["Hot_Points"].between(4, 5), "Hot_Symbol"] = "🔥🔥"
    out.loc[out["Hot_Points"] >= 6, "Hot_Symbol"] = "🔥🔥🔥"

    batter_side = (
        out.get("batter_profile_side", pd.Series("U", index=out.index))
        .fillna("U").astype(str).str.upper()
    )
    pitcher_side = (
        out.get("pitcher_throws", pd.Series("U", index=out.index))
        .fillna("U").astype(str).str.upper()
    )

    out["Platoon_Marker"] = ""
    out.loc[batter_side.eq("S"), "Platoon_Marker"] = "**"
    opposite = (
        (batter_side.eq("L") & pitcher_side.eq("R"))
        | (batter_side.eq("R") & pitcher_side.eq("L"))
    )
    out.loc[opposite & ~batter_side.eq("S"), "Platoon_Marker"] = "*"

    out["Player_Display"] = (
        out["player"].fillna("").astype(str)
        + out["Platoon_Marker"]
        + " "
        + out["Hot_Symbol"]
    ).str.strip()

    # Display-only "BEST MATCHUP": the best top-six fit within each offense.
    pitch_fit = pd.to_numeric(out.get("Pitch_Mix_Score"), errors="coerce").fillna(50)
    pitcher_vuln = pd.to_numeric(
        out.get("Pitcher_Vuln_Score"), errors="coerce"
    ).fillna(50)
    out["Matchup_Display_Score"] = (pitch_fit * 0.55 + pitcher_vuln * 0.45).clip(0, 100)

    keys = [
        c for c in ["game_pk", "team", "opposing_pitcher_id"]
        if c in out.columns
    ]
    if keys:
        out["Matchup_Display_Rank"] = (
            out.groupby(keys, dropna=False)["Matchup_Display_Score"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )
    else:
        out["Matchup_Display_Rank"] = pd.Series(
            range(1, len(out) + 1), index=out.index, dtype="Int64"
        )

    out["Matchup_Label"] = ""
    out.loc[
        (out["Matchup_Display_Rank"] == 1)
        & (out["Matchup_Display_Score"] >= 65),
        "Matchup_Label",
    ] = "BEST MATCHUP"

    return out


STADIUM_WEATHER = {
    "ARI": {"lat": 33.4453, "lon": -112.0667, "tz": "America/Phoenix", "roof": True},
    "ATL": {"lat": 33.8908, "lon": -84.4678, "tz": "America/New_York", "roof": False},
    "BAL": {"lat": 39.2839, "lon": -76.6217, "tz": "America/New_York", "roof": False},
    "BOS": {"lat": 42.3467, "lon": -71.0972, "tz": "America/New_York", "roof": False},
    "CHC": {"lat": 41.9484, "lon": -87.6553, "tz": "America/Chicago", "roof": False},
    "CWS": {"lat": 41.8299, "lon": -87.6338, "tz": "America/Chicago", "roof": False},
    "CIN": {"lat": 39.0979, "lon": -84.5082, "tz": "America/New_York", "roof": False},
    "CLE": {"lat": 41.4962, "lon": -81.6852, "tz": "America/New_York", "roof": False},
    "COL": {"lat": 39.7559, "lon": -104.9942, "tz": "America/Denver", "roof": False},
    "DET": {"lat": 42.3390, "lon": -83.0485, "tz": "America/New_York", "roof": False},
    "HOU": {"lat": 29.7573, "lon": -95.3555, "tz": "America/Chicago", "roof": True},
    "KC": {"lat": 39.0517, "lon": -94.4803, "tz": "America/Chicago", "roof": False},
    "LAA": {"lat": 33.8003, "lon": -117.8827, "tz": "America/Los_Angeles", "roof": False},
    "LAD": {"lat": 34.0739, "lon": -118.2400, "tz": "America/Los_Angeles", "roof": False},
    "MIA": {"lat": 25.7781, "lon": -80.2196, "tz": "America/New_York", "roof": True},
    "MIL": {"lat": 43.0280, "lon": -87.9712, "tz": "America/Chicago", "roof": True},
    "MIN": {"lat": 44.9817, "lon": -93.2776, "tz": "America/Chicago", "roof": False},
    "NYM": {"lat": 40.7571, "lon": -73.8458, "tz": "America/New_York", "roof": False},
    "NYY": {"lat": 40.8296, "lon": -73.9262, "tz": "America/New_York", "roof": False},
    "ATH": {"lat": 38.5618, "lon": -121.4997, "tz": "America/Los_Angeles", "roof": False},
    "PHI": {"lat": 39.9061, "lon": -75.1665, "tz": "America/New_York", "roof": False},
    "PIT": {"lat": 40.4469, "lon": -80.0057, "tz": "America/New_York", "roof": False},
    "SD": {"lat": 32.7076, "lon": -117.1570, "tz": "America/Los_Angeles", "roof": False},
    "SEA": {"lat": 47.5914, "lon": -122.3325, "tz": "America/Los_Angeles", "roof": True},
    "SF": {"lat": 37.7786, "lon": -122.3893, "tz": "America/Los_Angeles", "roof": False},
    "STL": {"lat": 38.6226, "lon": -90.1928, "tz": "America/Chicago", "roof": False},
    "TB": {"lat": 27.7682, "lon": -82.6534, "tz": "America/New_York", "roof": True},
    "TEX": {"lat": 32.7473, "lon": -97.0847, "tz": "America/Chicago", "roof": True},
    "TOR": {"lat": 43.6414, "lon": -79.3894, "tz": "America/Toronto", "roof": True},
    "WSH": {"lat": 38.8730, "lon": -77.0074, "tz": "America/New_York", "roof": False},
}


def fetch_weather_display(team: str, game_time_utc: str | None) -> dict[str, object]:
    """
    Fetch first-pitch weather from Open-Meteo for display only.
    Returns neutral/unknown values on failure. Never changes model scoring.
    """
    meta = STADIUM_WEATHER.get(str(team).upper())
    fallback = {
        "Weather_Status": "Unknown",
        "Temperature_F": None,
        "Humidity_pct": None,
        "Wind_mph": None,
        "Wind_Direction_deg": None,
        "Rain_Probability_pct": None,
        "Roof_Type": "Retractable" if meta and meta.get("roof") else "Open Air",
        "Roof_Status": "Retractable Roof" if meta and meta.get("roof") else "Open Air",
        "Weather_Impact": "Unknown",
        "Weather_Arrow": "—",
        "Weather_Source": "Unavailable",
    }
    if not meta or not game_time_utc:
        return fallback

    try:
        game_dt = pd.to_datetime(game_time_utc, utc=True)
        local_dt = game_dt.tz_convert(meta["tz"])
        forecast_date = local_dt.strftime("%Y-%m-%d")

        params = {
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "wind_speed_10m",
                "wind_direction_10m",
            ]),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": meta["tz"],
            "start_date": forecast_date,
            "end_date": forecast_date,
        }
        payload = get_json("https://api.open-meteo.com/v1/forecast", params)
        hourly = payload.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        if len(times) == 0:
            return fallback

        target = local_dt.tz_localize(None)
        idx = int(abs(times - target).argmin())

        temp = float(hourly["temperature_2m"][idx])
        humidity = float(hourly["relative_humidity_2m"][idx])
        rain = float(hourly["precipitation_probability"][idx])
        wind = float(hourly["wind_speed_10m"][idx])
        wind_dir = float(hourly["wind_direction_10m"][idx])

        # Conservative display-only interpretation.
        if meta.get("roof"):
            impact = "Retractable Roof"
            arrow = "—"
            roof_status = "Retractable Roof"
        else:
            score = 0
            if temp >= 82:
                score += 1
            elif temp <= 58:
                score -= 1
            if wind >= 12:
                score += 1  # Wind direction shown separately; no score effect.
            if rain >= 45:
                score -= 1

            if score >= 2:
                impact, arrow = "Favorable", "⬆"
            elif score <= -1:
                impact, arrow = "Unfavorable", "⬇"
            else:
                impact, arrow = "Neutral", "—"
            roof_status = "Open Air"

        status = "Forecast"
        if rain >= 60:
            status = "High Delay Risk"
        elif rain >= 35:
            status = "Delay Watch"

        return {
            "Weather_Status": status,
            "Temperature_F": round(temp, 1),
            "Humidity_pct": round(humidity, 0),
            "Wind_mph": round(wind, 1),
            "Wind_Direction_deg": round(wind_dir, 0),
            "Rain_Probability_pct": round(rain, 0),
            "Roof_Type": "Retractable" if meta.get("roof") else "Open Air",
            "Roof_Status": roof_status,
            "Weather_Impact": impact,
            "Weather_Arrow": arrow,
            "Weather_Source": "Open-Meteo forecast",
        }
    except Exception:
        return fallback


def add_weather_display(board: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather columns by game. This is display-only and does not recalculate
    Model_Score, Core_HR_Eligible, or rankings.
    """
    out = board.copy()
    if out.empty:
        return out

    game_key_cols = [c for c in ["game_pk", "home_team", "game_time_utc"] if c in out.columns]
    if not game_key_cols:
        for key, value in fetch_weather_display("", None).items():
            out[key] = value
        return out

    records: list[dict[str, object]] = []
    unique_games = out[game_key_cols].drop_duplicates()

    for _, game in unique_games.iterrows():
        home_team = str(game.get("home_team", "") or "")
        game_time = game.get("game_time_utc")
        weather = fetch_weather_display(home_team, game_time)
        record = {c: game.get(c) for c in game_key_cols}
        record.update(weather)
        records.append(record)

    weather_df = pd.DataFrame(records)
    return out.merge(weather_df, on=game_key_cols, how="left")


def add_hr_likelihood_display(board: pd.DataFrame) -> pd.DataFrame:
    """
    Add a display-only model-estimated HR likelihood percentage.

    This is derived from the existing V3.1 Model Score using a bounded logistic
    display transform. It does not alter Model_Score, ranking, Core HR status,
    or any other scanner calculation.
    """
    out = board.copy()
    score = pd.to_numeric(out.get("Model_Score"), errors="coerce").fillna(50.0)

    # Bounded approximately from 5% to 35%.
    likelihood = 0.05 + 0.30 / (1.0 + np.exp(-(score - 65.0) / 8.0))
    out["HR_Likelihood_pct"] = (likelihood * 100.0).round(0).astype(int)
    out["HR_Likelihood"] = out["HR_Likelihood_pct"].map(
        lambda value: f"{value:d}%"
    )
    return out


def load_optional_inputs(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run(game_date: str, output_dir: Path, lookback_days: int = 32, include_unconfirmed: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    games = schedule_for_date(game_date)
    ids = team_id_map()

    start = (pd.Timestamp(game_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = game_date
    probable_pitcher_ids = {
        int(pid)
        for game in games
        for pid in (game.away_pitcher_id, game.home_pitcher_id)
        if pid is not None
    }
    print(f"Pulling Statcast {start} through {end}...", flush=True)
    sc = pull_statcast(start, end, probable_pitcher_ids)
    if sc.empty:
        raise RuntimeError("No Statcast data returned.")

    # Preserve the original pitcher matchup window at 32 calendar days.
    # The wider source download is used to reconstruct each hitter's
    # 10 most recent games.
    pitcher_start = pd.Timestamp(game_date) - pd.Timedelta(days=32)
    season_sc = sc[sc["game_date"] >= pitcher_start].copy()

    env = load_optional_inputs(Path("environment_inputs.csv"))
    odds = load_optional_inputs(Path("odds_inputs.csv"))
    weights = DEFAULT_WEIGHTS.copy()
    weight_path = Path("weights_hr.json")
    if weight_path.exists():
        weights.update(json.loads(weight_path.read_text()))

    rows = []
    for game in games:
        try:
            away_lineup, home_lineup, away_game_roster, home_game_roster = (
                game_lineups_and_rosters(game.game_pk)
            )
        except Exception:
            away_lineup, home_lineup = [], []
            away_game_roster, home_game_roster = [], []

        # Confirmed lineups remain optional. When unavailable, prefer the
        # game-level roster; only then use the validated team active roster.
        if not away_lineup:
            away_lineup = away_game_roster or fallback_roster(ids[game.away], game_date)
        if not home_lineup:
            home_lineup = home_game_roster or fallback_roster(ids[game.home], game_date)

        game_person_ids = [
            *(p.get("player_id") for p in away_lineup),
            *(p.get("player_id") for p in home_lineup),
            game.away_pitcher_id,
            game.home_pitcher_id,
        ]
        hand_map = people_handedness(game_person_ids)
        away_bvp = historical_bvp_for_pitcher(
            [int(p["player_id"]) for p in away_lineup], game.home_pitcher_id
        )
        home_bvp = historical_bvp_for_pitcher(
            [int(p["player_id"]) for p in home_lineup], game.away_pitcher_id
        )

        for side, lineup, team, opp, pitcher_id, pitcher_name, park_team in [
            ("away", away_lineup, game.away, game.home, game.home_pitcher_id, game.home_pitcher_name, game.home),
            ("home", home_lineup, game.home, game.away, game.away_pitcher_id, game.away_pitcher_name, game.home),
        ]:
            if pitcher_id is None:
                continue
            bvp_map = away_bvp if side == "away" else home_bvp
            for hitter in lineup:
                pid = hitter["player_id"]
                p20 = last_n_games_for_batter(sc, pid, 10)
                if p20.empty:
                    continue
                stand_mode = p20["stand"].dropna().mode()
                stand = stand_mode.iloc[0] if not stand_mode.empty else "U"

                listed_bat_side = hand_map.get(pid, {}).get("bat", "U")
                pitcher_throws = hand_map.get(pitcher_id, {}).get("throw", "U")
                if pitcher_throws not in {"L", "R"} and pitcher_id:
                    p_hand_rows = season_sc[season_sc["pitcher"] == pitcher_id]
                    if "p_throws" in p_hand_rows.columns:
                        p_hand_mode = p_hand_rows["p_throws"].dropna().astype(str).str.upper().mode()
                        if not p_hand_mode.empty and p_hand_mode.iloc[0] in {"L", "R"}:
                            pitcher_throws = p_hand_mode.iloc[0]

                production = count_runs_rbi_hits(p20)
                contact = batted_ball_metrics(p20)
                pv = pitcher_vulnerability(season_sc, pitcher_id, stand)
                batter_season = season_sc[season_sc["batter"] == pid]
                pitcher_season = season_sc[season_sc["pitcher"] == pitcher_id] if pitcher_id else pd.DataFrame()
                pitch_score = pitch_mix_matchup_score(batter_season, pitcher_season) if pitcher_id else 50.0

                park_factor = PARK_FACTORS.get(park_team, 1.0)
                weather_factor = 1.0
                roof_status = ""
                temp_f = np.nan
                wind_out_mph = np.nan
                if not env.empty and "game_pk" in env.columns:
                    m = env[env["game_pk"] == game.game_pk]
                    if not m.empty:
                        rec = m.iloc[0]
                        weather_factor = float(rec.get("weather_factor", 1.0))
                        roof_status = rec.get("roof_status", "")
                        temp_f = rec.get("temp_f", np.nan)
                        wind_out_mph = rec.get("wind_out_mph", np.nan)

                hr_odds = np.nan
                market_value = 50.0
                if not odds.empty and "player_id" in odds.columns:
                    m = odds[odds["player_id"] == pid]
                    if not m.empty:
                        hr_odds = pd.to_numeric(m.iloc[0].get("hr_odds_american"), errors="coerce")
                        if pd.notna(hr_odds):
                            implied = 100 / (hr_odds + 100) if hr_odds > 0 else (-hr_odds) / ((-hr_odds) + 100)
                            # Initial market score: longer price is better, but capped.
                            market_value = float(np.clip((0.18 - implied) * 500 + 50, 0, 100))

                row = {
                    "game_pk": game.game_pk,
                    "home_team": game.home,
                    "game_time_utc": game.game_date,
                    "game_date": game_date,
                    "status": game.status,
                    "team": team,
                    "opponent": opp,
                    "home_away": side,
                    "venue": game.venue,
                    "player_id": pid,
                    "player": hitter["player"],
                    "lineup_spot": hitter["lineup_spot"],
                    "position": hitter["position"],
                    "roster_source": hitter.get("roster_source", "unknown"),
                    "bat_side": stand,
                    "batter_profile_side": listed_bat_side,
                    "pitcher_throws": pitcher_throws,
                    "opposing_pitcher_id": pitcher_id,
                    "opposing_pitcher": pitcher_name,
                    "Park_Factor": park_factor,
                    "Weather_Factor": weather_factor,
                    "roof_status": roof_status,
                    "temp_f": temp_f,
                    "wind_out_mph": wind_out_mph,
                    "HR_Odds_American": hr_odds,
                    "Market_Value_Score": market_value,
                    "Pitch_Mix_Score": pitch_score,
                    **bvp_map.get(pid, {
                        "BvP_PA": 0, "BvP_AB": 0, "BvP_H": 0,
                        "BvP_HR": 0, "BvP_AVG": np.nan,
                        "BvP_OBP": np.nan, "BvP_SLG": np.nan,
                    }),
                    **production,
                    **contact,
                    **pv,
                }
                rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        raise RuntimeError(
            "No hitter rows were created. Confirm that opposing pitchers "
            "are listed for the selected slate."
        )
    board = add_weather_display(board)
    board = add_v4_scores(board)
    gc.collect()

    pct_cols = [
        "HH_pct", "Barrel_pct_approx", "SweetSpot_pct", "PullAir_pct",
        "Pitcher_HR_pct", "Pitcher_HH_pct", "Pitcher_Barrel_pct_approx"
    ]
    for c in pct_cols:
        if c in board:
            board[c] = pd.to_numeric(board[c], errors="coerce") * 100

    ordered = [
        "Overall_Rank", "Game_Rank", "Player_Display", "HR_Likelihood",
        "Best_Look", "Conviction", "Hot_Symbol", "Due_Meter",
        "opposing_pitcher", "Pitcher_Sample_Flag", "Pitcher_Sample",
        "Power_Index", "Base_Power_Index", "Recent_Power", "Pitcher_Leak",
        "BvP_Score", "BvP_Adjustment", "BvP_PA", "BvP_AB", "BvP_H",
        "BvP_HR", "BvP_AVG", "BvP_OBP", "BvP_SLG", "BvP_Sample_Flag",
        "Due_Score_V4", "Environment", "TANKS", "Porch_Shots",
        "Top_10_Overall", "Top_4_Per_Game", "team", "opponent", "venue",
        "lineup_spot", "roster_source",
        "batter_profile_side", "pitcher_throws",
        "HR", "Barrels_approx", "EV_100_plus", "EV_100_plus_outs",
        "Fly_375_plus", "Out_380_400", "Near_HR", "xHR_minus_HR",
        "Avg_EV", "EV90", "Max_EV", "HH_pct", "Barrel_pct_approx",
        "PullAir_pct", "SweetSpot_pct", "Pitch_Mix_Score",
        "Pitcher_BBE", "Pitcher_HR", "Pitcher_HR_pct",
        "Pitcher_HH_pct", "Pitcher_Barrel_pct_approx", "Pitcher_Avg_EV",
        "Park_Factor", "Weather_Factor", "Weather_Arrow",
        "Weather_Impact", "Temperature_F", "Wind_mph",
        "Rain_Probability_pct", "Roof_Status", "status", "game_pk",
        "player_id", "opposing_pitcher_id",
    ]
    ordered = [c for c in ordered if c in board.columns]
    board = board[ordered]

    csv_path = output_dir / f"outlaw_v4_{game_date}.csv"
    xlsx_path = output_dir / f"outlaw_v4_{game_date}.xlsx"
    board.to_csv(csv_path, index=False)

    top10 = board[board["Top_10_Overall"] == True].head(10)
    top4 = board[board["Top_4_Per_Game"] == True]
    due = board[board["Due_Meter"] == "🟢 DUE"].head(25)
    pitchers = (
        board.groupby(["opposing_pitcher"], dropna=False)
        .agg(
            Pitcher_Leak=("Pitcher_Leak", "max"),
            Pitcher_Sample=("Pitcher_Sample", "max"),
            Top_Hitter=("Player_Display", "first"),
        )
        .sort_values("Pitcher_Leak", ascending=False)
        .reset_index()
        .head(20)
    )

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        board.to_excel(writer, sheet_name="Full Board", index=False)
        top10.to_excel(writer, sheet_name="Top 10", index=False)
        top4.to_excel(writer, sheet_name="Top 4 Per Game", index=False)
        due.to_excel(writer, sheet_name="Due Hitters", index=False)
        pitchers.to_excel(writer, sheet_name="Pitchers To Attack", index=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Outlaw HR Scanner V5.1 — last 10 games + BvP")
    parser.add_argument("--date", default=str(date.today()), help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--lookback-days", type=int, default=32)
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Compatibility flag; active rosters are now scanned automatically.",
    )
    args = parser.parse_args()
    run(
        game_date=args.date,
        output_dir=Path(args.output_dir),
        lookback_days=args.lookback_days,
        include_unconfirmed=args.include_unconfirmed,
    )


if __name__ == "__main__":
    main()
