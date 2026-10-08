@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal

echo ============================================================
echo   把本地最新代码推送到 GitHub
echo   （Streamlit Cloud 检测到后会 1~3 分钟内自动重新部署）
echo ============================================================
echo.

git add -A
git -c core.quotepath=false diff --cached --quiet
if errorlevel 1 goto commit

echo   没有检测到新的改动，直接推送当前提交。
goto push

:commit
echo   正在提交改动 ...
git -c core.quotepath=false commit -m "更新：修复图表元素ID冲突 / 补齐图表标题 / 收紧依赖版本"
if errorlevel 1 goto failed

:push
echo.
echo   正在推送到 GitHub ...
git push
if errorlevel 1 goto failed

echo.
echo ============================================================
echo   [成功] 已推送！
echo.
echo   下一步：打开 https://share.streamlit.io
echo   在你的应用页面点 Manage app，可以看重新部署的进度
echo   一般 1~3 分钟后刷新网页就生效了
echo ============================================================
pause
exit /b 0

:failed
echo.
echo   [失败] 请把上面的错误信息截图发给助手。
pause
exit /b 1