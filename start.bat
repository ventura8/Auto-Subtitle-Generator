@echo off
set PATH=%~dp0venv\ffmpeg\bin;%PATH%
call venv\Scripts\activate
python auto_subtitle.py %*
pause
