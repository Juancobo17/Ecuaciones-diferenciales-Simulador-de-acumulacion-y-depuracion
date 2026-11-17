@echo off
chcp 65001 >nul
echo Instalando librerias necesarias (numpy y matplotlib)...
python -m pip install --quiet numpy matplotlib
echo Abriendo el simulador...
python "%~dp0simulador_memoria.py"
pause
