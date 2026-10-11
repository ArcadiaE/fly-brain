@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动果蝇脑实时控制台，浏览器打开 http://127.0.0.1:8766/
start "" http://127.0.0.1:8766/
.venv\Scripts\python.exe experiments\live_server.py --port 8766
pause
