@echo off
rem morie Console - opens a terminal with the bundled morie already on PATH.
rem Installed alongside the morie bundle and launched from the Start Menu
rem shortcut, so a user never has to know what PowerShell or PATH is.
title morie Console
set "PATH=%~dp0;%PATH%"
echo.
echo   morie is ready. Some things to try:
echo.
echo     morie --help
echo     morie list-modules
echo     morie tutorial
echo.
cmd /k
