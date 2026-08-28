@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure-ssh.ps1" -PublicKeyPath "%~dp0authorized_keys"
