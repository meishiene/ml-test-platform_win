@echo off
rem ============================================================================
rem  大模型测试平台 · Windows 一键启动
rem
rem  双击本文件即可启动；也可以在命令行里追加参数覆盖默认值：
rem      start.bat --port 9000            换端口
rem      start.bat --host 0.0.0.0         允许局域网访问（首次会弹防火墙授权）
rem      start.bat --no-browser            不自动打开浏览器
rem
rem  脚本会自动完成：定位 Python → 创建 .venv → 按 requirements.txt 补齐依赖
rem  → 加载 .env → 释放被占用的端口 → 启动 Web 服务并打开浏览器。
rem  真正的启动逻辑在 scripts\start_web.py 里，与 macOS/Linux 的 start.sh 共用一份实现。
rem ============================================================================
chcp 65001 >nul
setlocal enableextensions
cd /d "%~dp0"

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

rem 国内网络下载依赖缓慢时，取消下面一行的注释改用镜像源：
rem set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

echo.
echo ==== 大模型测试平台 · Windows 一键启动 ====
echo.

rem ---- 1/4 定位可用的 Python（需要 3.10 及以上，推荐 3.12+）----
set "PY_CMD="
call :probe "py -3"
if not defined PY_CMD call :probe "py"
if not defined PY_CMD call :probe "python"
if not defined PY_CMD call :probe "python3"
if not defined PY_CMD goto :no_python
for /f "delims=" %%v in ('%PY_CMD% -V 2^>^&1') do echo [1/4] 已找到 %%v

rem ---- 2/4 准备虚拟环境 ----
if not exist "%VENV_PY%" goto :make_venv
"%VENV_PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :broken_venv
echo [2/4] 复用已有虚拟环境：%VENV_DIR%
goto :deps

:make_venv
echo [2/4] 正在创建虚拟环境：%VENV_DIR%
%PY_CMD% -m venv "%VENV_DIR%"
if errorlevel 1 goto :venv_failed
goto :deps

rem ---- 3/4 补齐依赖（首次运行需要几分钟）----
:deps
"%VENV_PY%" -c "import httpx,openai,jsonschema,pytest,xdist,PIL" >nul 2>nul
if not errorlevel 1 goto :start
echo [3/4] 正在安装/补齐项目依赖，首次运行需要几分钟……
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r "%ROOT_DIR%requirements.txt"
if errorlevel 1 goto :deps_failed
goto :start

rem ---- 4/4 启动 Web 服务（Ctrl+C 停止）----
:start
echo [4/4] 正在启动 Web 服务……
echo.
"%VENV_PY%" "%ROOT_DIR%scripts\start_web.py" --open-browser %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo [提示] 服务异常退出，退出码 %EXIT_CODE%，请查看上方日志。
if "%EXIT_CODE%"=="0" echo 服务已正常停止。
goto :done

rem =============================== 子过程与错误分支 ===============================
:probe
rem 依次探测候选解释器：能用且版本 >= 3.10 才写入 PY_CMD。
set "CANDIDATE=%~1"
%CANDIDATE% -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PY_CMD=%CANDIDATE%"
exit /b 0

:no_python
echo [错误] 未找到可用的 Python 3（需要 3.10 及以上，推荐 3.12+）。
echo        请到 https://www.python.org/downloads/windows/ 安装，
echo        安装时务必勾选 "Add python.exe to PATH" 与 "py launcher"，然后重新运行本脚本。
goto :fail

:broken_venv
echo [错误] 已有虚拟环境不可用：%VENV_PY%
echo        通常是这个 .venv 由其它 Python 版本创建，或从别的机器（macOS/Linux）拷贝而来。
echo        请删除目录 %VENV_DIR% 后重新运行本脚本。
goto :fail

:venv_failed
echo [错误] 虚拟环境创建失败，请检查磁盘空间与目录权限后重试。
goto :fail

:deps_failed
echo [错误] 依赖安装失败。请检查网络后重试；国内网络可取消脚本顶部的 PIP_INDEX_URL 注释改用镜像源。
goto :fail

:fail
echo.
echo 启动失败，请按上面的提示处理后重试。
echo.
pause
endlocal
exit /b 1

:done
echo.
pause
endlocal
exit /b 0
