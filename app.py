from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Outlaw HR Scanner V5.2",
    page_icon="💣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {padding-top: .7rem; padding-left: .6rem; padding-right: .6rem;}
.stButton > button {width: 100%; min-height: 3rem; font-size: 1.05rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: .5rem; border-radius: .6rem;}
@media (max-width: 700px) {
  h1 {font-size: 1.55rem !important;}
  .block-container {padding-left: .45rem; padding-right: .45rem;}
}
</style>
""", unsafe_allow_html=True)

st.title("💣 Outlaw HR Scanner V5.2")
st.caption(
    "Last 10 games: 35% recent form | 35% pitcher matchup | "
    "25% near-HR | 5% park/weather | career BvP display-only"
)
st.caption("* = platoon advantage | ** = switch hitter | ⚠ = small pitcher sample")

scan_date = st.date_input("Slate date", value=date.today())

if "error" not in st.session_state:
    st.session_state.error = ""

if st.button("Run V5.2 Scan", type="primary", use_container_width=True):
    st.session_state.error = ""
    cmd = [
        sys.executable, str(APP_DIR / "scanner.py"),
        "--date", scan_date.isoformat(),
        "--output-dir", str(OUTPUT_DIR),
        "--lookback-days", "40",
    ]
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    with st.status("Running V5.2 scanner...", expanded=True) as status:
        st.write("Downloading recent Statcast data...")
        st.write("Building last-10 hitter profiles...")
        st.write("Validating active rosters and calculating recent form, pitcher matchup, near-HR, environment, and display-only BvP...")
        try:
            p = subprocess.run(
                cmd, cwd=APP_DIR, env=env, capture_output=True,
                text=True, timeout=600
            )
            if p.returncode:
                status.update(label="Scan failed", state="error")
                st.session_state.error = (p.stderr or p.stdout or "Unknown error")[-7000:]
            else:
                status.update(label="Scan complete", state="complete")
                if p.stdout:
                    st.code(p.stdout[-2000:])
        except Exception as exc:
            status.update(label="Scan failed", state="error")
            st.session_state.error = f"{type(exc).__name__}: {exc}"

if st.session_state.error:
    st.error("The scan did not complete.")
    st.code(st.session_state.error)

csv_path = OUTPUT_DIR / f"outlaw_v4_{scan_date.isoformat()}.csv"
xlsx_path = OUTPUT_DIR / f"outlaw_v4_{scan_date.isoformat()}.xlsx"

if csv_path.exists():
    board = pd.read_csv(csv_path)

    c1, c2, c3 = st.columns(3)
    c1.metric("Players", len(board))
    c2.metric("Games", board["game_pk"].nunique() if "game_pk" in board else "—")
    c3.metric("Top Index", f"{board['Power_Index'].max():.1f}")

    core_cols = [
        "Player_Display", "HR_Likelihood", "Best_Look", "Conviction",
        "Hot_Symbol", "Due_Meter", "opposing_pitcher",
        "Pitcher_Sample_Flag", "BvP_Sample_Flag", "Power_Index",
        "Recent_Power", "Pitcher_Matchup_Score", "Pitcher_Leak",
        "Near_HR_Score", "BvP_Score",
        "BvP_PA", "BvP_HR", "TANKS", "Porch_Shots", "team", "opponent",
    ]
    core_cols = [c for c in core_cols if c in board.columns]

    tabs = st.tabs([
        "⭐ Top 10", "🎮 Top 4/Game", "🪅 Pitchers",
        "🎯 Due", "🔥 Hot", "📋 Full Board"
    ])

    with tabs[0]:
        top10 = board[board["Top_10_Overall"] == True].head(10)
        st.dataframe(top10[core_cols], hide_index=True, use_container_width=True, height=470)

    with tabs[1]:
        top4 = board[board["Top_4_Per_Game"] == True].copy()
        for game_pk, game in top4.groupby("game_pk", sort=False):
            teams = f"{game['team'].iloc[0]} / {game['opponent'].iloc[0]}"
            with st.expander(teams, expanded=False):
                st.dataframe(
                    game.sort_values("Game_Rank")[core_cols],
                    hide_index=True, use_container_width=True
                )

    with tabs[2]:
        pitchers = (
            board.groupby("opposing_pitcher", dropna=False)
            .agg(
                Pitcher_Leak=("Pitcher_Leak", "max"),
                Sample=("Pitcher_Sample", "max"),
                Top_Hitter=("Player_Display", "first"),
                Team=("team", "first"),
            )
            .sort_values("Pitcher_Leak", ascending=False)
            .reset_index()
            .head(20)
        )
        st.dataframe(pitchers, hide_index=True, use_container_width=True, height=650)

    with tabs[3]:
        due = board[board["Due_Meter"] == "🟢 DUE"].sort_values(
            ["Due_Score_V4", "Power_Index"], ascending=False
        )
        st.dataframe(due[core_cols].head(30), hide_index=True, use_container_width=True, height=650)

    with tabs[4]:
        hot = board.sort_values(["Recent_Power", "TANKS"], ascending=False)
        st.dataframe(hot[core_cols].head(30), hide_index=True, use_container_width=True, height=650)

    with tabs[5]:
        detail_cols = [
            "Overall_Rank", "Game_Rank", "Player_Display", "HR_Likelihood",
            "Best_Look", "Conviction", "Due_Meter", "opposing_pitcher",
            "Pitcher_Sample_Flag", "Power_Index", "Recent_Power",
            "Pitcher_Matchup_Score", "Pitcher_Leak", "Near_HR_Score",
            "BvP_Score", "BvP_PA",
            "BvP_HR", "BvP_AVG", "BvP_SLG", "BvP_Sample_Flag",
            "Due_Score_V4", "Environment", "TANKS",
            "Porch_Shots", "HR", "Barrels_approx", "EV_100_plus",
            "EV_100_plus_outs", "Fly_350_plus_outs",
            "Fly_375_plus_outs", "Out_380_400",
            "Near_HR", "xHR_minus_HR", "PullAir_pct", "Pitcher_HR_pct",
            "Pitcher_Barrel_pct_approx", "Park_Factor", "Weather_Impact",
        ]
        detail_cols = [c for c in detail_cols if c in board.columns]
        st.dataframe(board[detail_cols], hide_index=True, use_container_width=True, height=720)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download CSV", csv_path.read_bytes(), csv_path.name,
            "text/csv", use_container_width=True
        )
    if xlsx_path.exists():
        with d2:
            st.download_button(
                "Download Excel", xlsx_path.read_bytes(), xlsx_path.name,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
else:
    st.info("Choose the slate date and tap **Run V5.2 Scan**.")
