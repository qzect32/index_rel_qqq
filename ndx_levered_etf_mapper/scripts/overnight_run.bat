@echo off
setlocal
cd /d %~dp0\..

set STAMP=%DATE:~-4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set STAMP=%STAMP: =0%

if not exist data\logs mkdir data\logs
set LOG=data\logs\overnight_%STAMP%.log

echo Running overnight checks... > %LOG%

python scripts\mh_doctor.py >> %LOG% 2>&1
python -m py_compile app\streamlit_app.py scripts\decisions_listener.py >> %LOG% 2>&1
python scripts\todo_status_gen.py >> %LOG% 2>&1
python scripts\todo_mega_sprint_gen.py >> %LOG% 2>&1
python scripts\todo_mega_status_gen.py >> %LOG% 2>&1
pytest -q >> %LOG% 2>&1
python scripts\make_debug_bundle.py >> %LOG% 2>&1
git status --porcelain >> %LOG% 2>&1

echo %LOG%
endlocal
