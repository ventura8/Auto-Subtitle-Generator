param(
    [Parameter(Mandatory = $true)]
    [string]$PublicKeyPath
)

$ErrorActionPreference = "Stop"
$capabilityName = "OpenSSH.Server~~~~0.0.1.0"
if ((Get-WindowsCapability -Online -Name $capabilityName).State -ne "Installed") {
    Add-WindowsCapability -Online -Name $capabilityName
}

$authorizedKeysPath = Join-Path $env:ProgramData "ssh\administrators_authorized_keys"
$publicKey = (Get-Content -Raw -Path $PublicKeyPath).Trim()
if (-not $publicKey) {
    throw "The SSH public key is empty."
}

Set-Content -Path $authorizedKeysPath -Value $publicKey -Encoding ascii
icacls.exe $authorizedKeysPath /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
Set-Service -Name sshd -StartupType Automatic
Start-Service -Name sshd
