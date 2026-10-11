@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 生成离线回放页面（糖味觉实验 1 秒），完成后在 experiments\out\ 下
.venv\Scripts\python.exe experiments\export_3d.py --exp sugar --t 1.0
start "" experiments\out\brain3d_sugar.html
pause
