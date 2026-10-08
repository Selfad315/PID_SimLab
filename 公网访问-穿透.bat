@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
set PORT=8501
set CF=%~dp0cloudflared.exe

echo ============================================================
echo   PID_SimLab  -  公网访问（Cloudflare 免费隧道）
echo ============================================================
echo   本方案会把本地服务映射成一个 https 公网地址，
echo   手机用 4G/5G 或任意 WiFi 都能打开。
echo ============================================================
echo.

if exist "%CF%" goto havecf
echo   [1/4] 未检测到 cloudflared，正在下载（约 30 MB）...
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '%CF%' -UseBasicParsing}catch{exit 1}"
if not exist "%CF%" goto dlfail
echo         下载完成。

:havecf
echo   [2/4] 检查本地仿真服务 ...
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri 'http://localhost:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 3 | Out-Null; exit 0}catch{exit 1}"
if errorlevel 1 goto startsrv
echo         服务已在运行。
goto waitready

:startsrv
echo         未运行，正在后台启动 ...
start "PID_SimLab Service" /min cmd /c "python -m streamlit run app.py --server.headless=true --server.port=%PORT% --server.address=0.0.0.0 --browser.gatherUsageStats=false"

:waitready
echo   [3/4] 等待服务就绪 ...
set /a n=0
:waitloop
timeout /t 2 /nobreak >nul
set /a n+=1
powershell -NoProfile -Command "try{Invoke-WebRequest -Uri 'http://localhost:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 3 | Out-Null; exit 0}catch{exit 1}"
if not errorlevel 1 goto tunnel
if %n% lss 25 goto waitloop
echo         服务启动超时，请先确认依赖已安装: pip install -r requirements.txt
pause
exit /b 1

:tunnel
echo   [4/4] 创建公网隧道 ...
echo.
echo ============================================================
echo   稍等 5-10 秒，下面会出现一行 https://xxxx.trycloudflare.com
echo   那就是公网地址 —— 手机用任何网络都能打开它！
echo.
echo   * 这个窗口不能关，关掉公网访问立即失效
echo   * 每次运行生成的地址都不同，重新运行请看新地址
echo ============================================================
echo.
"%CF%" tunnel --url http://localhost:%PORT%
pause
exit /b 0

:dlfail
echo.
echo   !! cloudflared 下载失败（可能是网络受限）
echo   请手动下载后放到本文件夹并改名为 cloudflared.exe：
echo   https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
echo.
pause
exit /b 1
