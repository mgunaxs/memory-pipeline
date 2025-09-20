@echo off
REM Create symlinks for python and pip to point to Python 3.12
mklink "C:\Windows\python.exe" "D:\system(Dont delete)\Python312\python.exe"
mklink "C:\Windows\pip.exe" "D:\system(Dont delete)\Python312\Scripts\pip.exe"
echo Python 3.12 set as default
pause