# Outlaw HR Scanner V4

## Purpose
Find the strongest home-run candidates using the most recent 10 games.

## Formula
- 50% recent hitter power
- 25% pitcher home-run leak by batter side
- 15% due indicators
- 10% park/weather environment

## Main boards
- Top 10 overall
- Top 4 per game
- Pitchers to attack
- Due hitters
- Hottest hitters
- Full board

## Upload
Upload these files to the root of the `v4-rebuild` branch:

- app.py
- scanner.py
- requirements.txt
- config.json
- README.md

Then configure a Streamlit test app to use:
- Branch: `v4-rebuild`
- Main file: `app.py`

The HRR scanner is not included and is not changed.
