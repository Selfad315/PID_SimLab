@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
set PORT=8501
set "IPTMP=%TEMP%\_pid_simlab_ip.txt"

echo ============================================================
echo   PID_SimLab  -  启动服务并用 Edge 打开
echo ============================================================
echo.

rem ---- 自动识别本机局域网 IP ----
set "LANIP="
python -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('10.255.255.255',1));print(s.getsockname()[0]);s.close()" > "%IPTMP%" 2>nul
if exist "%IPTMP%" set /p LANIP=<"%IPTMP%"
del "%IPTMP%" >nul 2>nul
if "%LANIP%"=="" set "LANIP=127.0.0.1"

echo [1/3] 检查服务是否已在运行 ...
powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri 'http://localhost:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 3; if($r.StatusCode -eq 200){exit 0}else{exit 1}}catch{exit 1}"
if errorlevel 1 goto startserver
echo       服务已在运行。
goto waitready

:startserver
echo       未运行，正在后台启动服务 ...
start "PID_SimLab Service" /min cmd /c "python -m streamlit run app.py --server.headless=true --server.port=%PORT% --server.address=0.0.0.0 --browser.gatherUsageStats=false"

:waitready
echo [2/3] 等待服务就绪 ...
set /a n=0
:waitloop
timeout /t 2 /nobreak >nul
set /a n+=1
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri 'http://localhost:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 3 | Out-Null; exit 0}catch{exit 1}"
if not errorlevel 1 goto ready
if %n% lss 25 goto waitloop
echo       服务启动超时。请先确认已安装依赖:
echo           pip install -r requirements.txt
pause
exit /b 1

:ready
echo       服务就绪
echo.
echo [3/3] 用 Edge 打开本机页面 ...
start msedge "http://localhost:%PORT%"
if errorlevel 1 start "" "http://localhost:%PORT%"

echo.
echo ============================================================
echo   本机访问 : http://localhost:%PORT%
echo   手机访问 : http://%LANIP%:%PORT%
echo ============================================================
echo.
echo   手机需连接与电脑【同一个 Wi-Fi】。
echo   若手机打不开，请用【管理员身份】运行一次下面这行，放行端口：
echo     netsh advfirewall firewall add rule name="PID_SimLab 8501" dir=in action=allow protocol=TCP localport=%PORT%
echo.
echo   停止服务：关闭标题为 PID_SimLab Service 的最小化窗口
echo ------------------------------------------------------------
timeout /t 25 /nobreak >nul
endlocal
