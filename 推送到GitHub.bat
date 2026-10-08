@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

echo ============================================================
echo   把 PID_SimLab 推送到 GitHub
echo ============================================================
echo.

if not exist ".git" goto nogit

set "GHUSER="
set /p "GHUSER=请输入你的 GitHub 用户名（例如 Selfad315）: "
if "%GHUSER%"=="" goto nouser

set "REPO=PID_SimLab"
set "URL=https://github.com/%GHUSER%/%REPO%.git"

echo.
echo   目标仓库: %URL%
echo.
echo   推送前请先在浏览器里创建好这个仓库:
echo     1. 打开 https://github.com/new
echo     2. Repository name 填  %REPO%
echo     3. 选择 Public
echo     4. 不要勾选 Add a README / .gitignore / license
echo     5. 点 Create repository
echo.
pause

git remote remove origin >nul 2>nul
git remote add origin "%URL%"
git branch -M main

echo.
echo 正在推送 ...（若弹出浏览器，请选择用 GitHub 账号登录并授权）
echo.
git push -u origin main

if errorlevel 1 goto pushfail

echo.
echo ============================================================
echo   推送成功！
echo   代码地址: https://github.com/%GHUSER%/%REPO%
echo.
echo   下一步：打开 https://share.streamlit.io 部署成公网网站
echo ============================================================
pause
exit /b 0

:nogit
echo [错误] 当前目录不是 git 仓库，请确认文件完整。
pause
exit /b 1

:nouser
echo [错误] 用户名不能为空。
pause
exit /b 1

:pushfail
echo.
echo ============================================================
echo   [推送失败] 常见原因：
echo     - 仓库还没在 GitHub 上创建
echo     - 用户名拼写错误
echo     - 仓库名不是 %REPO%
echo   请把上面的错误信息截图发给我
echo ============================================================
pause
exit /b 1