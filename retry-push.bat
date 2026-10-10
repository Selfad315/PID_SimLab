@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

echo ============================================================
echo   重试推送到 GitHub（最多 5 次，针对网络不稳定做了优化）
echo ============================================================
echo.

rem 清掉可能干扰的代理环境变量
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="

rem 传输参数调优：HTTP/1.1 + 大缓冲，对不稳定网络更友好
git config --global http.version HTTP/1.1 >nul 2>nul
git config --global http.postBuffer 524288000 >nul 2>nul
echo   已应用网络传输优化参数（HTTP/1.1 + 500MB 缓冲）
echo.

set /a n=0

:retry
set /a n+=1
echo ------------------------------------------------------------
echo   第 %n% 次尝试推送 ...
echo ------------------------------------------------------------
git push
if not errorlevel 1 goto success
if %n% geq 5 goto failed
echo.
echo   本次失败，等 6 秒后自动重试 ...
timeout /t 6 /nobreak >nul
echo.
goto retry

:success
echo.
echo ============================================================
echo   [成功] 推送完成！
echo.
echo   下一步：打开 https://share.streamlit.io
echo   你的应用会在 1~3 分钟内自动重新部署
echo ============================================================
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   5 次都失败了。这不是代码问题，是网络连不上 GitHub。
echo.
echo   可以按顺序试这几个办法：
echo.
echo   【办法 1】换网络（最简单有效）
echo       用手机开个热点，电脑连上热点后，再运行本脚本一次。
echo       手机网络的线路经常和宽带不一样，能绕开阻断。
echo.
echo   【办法 2】启动代理工具
echo       你电脑上配置过 127.0.0.1:7897 这个代理端口（Clash 类工具）。
echo       如果装了这类软件，先启动它，然后执行下面两行：
echo           git config --global http.proxy http://127.0.0.1:7897
echo           git config --global https.proxy http://127.0.0.1:7897
echo       再运行本脚本。推送成功后，执行下面两行关掉代理配置：
echo           git config --global --unset http.proxy
echo           git config --global --unset https.proxy
echo.
echo   【办法 3】隔一段时间再试
echo       GitHub 在国内是间歇性可达的，说不定十分钟后就通了。
echo.
echo   重要提示：线上程序目前是正常的（关键修复早就推上去了），
echo             本脚本只是把「性能优化」推上去属于锦上添花，
echo             推不上去也不影响你使用和演示。
echo ============================================================
pause
exit /b 1