@echo off
title AdaMule Cyber Defense War Room
echo =======================================================
echo   LAUNCHING ADAMULE CYBER DEFENSE WAR ROOM
echo =======================================================
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python scripts\launch_war_room.py --port 8080
) else (
    python scripts\launch_war_room.py --port 8080
)
pause
