@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
set PORT=8501
set "IPTMP=%TEMP%\_pid_simlab_ip.txt"

echo ============================================================
echo   PID_SimLab  -  启动仿真平台（电脑 + 手机均可访问）
echo ============================================================
echo.

rem ---- 自动识别本机局域网 IP ----
set "LANIP="
python -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('10.255.255.255',1));print(s.getsockname()[0]);s.close()" > "%IPTMP%" 2>nul
if exist "%IPTMP%" set /p LANIP=<"%IPTMP%"
del "%IPTMP%" >nul 2>nul
if "%LANIP%"=="" set "LANIP=127.0.0.1"

echo   本机访问 : http://localhost:%PORT%
echo   手机访问 : http://%LANIP%:%PORT%
echo.
echo   手机需连接与电脑【同一个 Wi-Fi】。
echo   若手机打不开，请用【管理员身份】运行一次下面这行，放行端口：
echo     netsh advfirewall firewall add rule name="PID_SimLab 8501" dir=in action=allow protocol=TCP localport=%PORT%
echo.
echo ------------------------------------------------------------
echo   服务启动后请勿关闭本窗口（关掉即停止服务）
echo ------------------------------------------------------------
echo.
python -m streamlit run app.py --server.headless=true --server.port=%PORT% --server.address=0.0.0.0 --browser.gatherUsageStats=false
pause
endlocal
