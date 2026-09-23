package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestBashPathIsAbsoluteAndExists(t *testing.T) {
	if runtime.GOOS == "windows" {
		// bashPath is only consulted on non-Windows, and its "/bin/bash"
		// fallback is not an absolute path by Windows' rules.
		t.Skip("bashPath is not used on Windows")
	}
	got := bashPath()
	if !filepath.IsAbs(got) {
		t.Fatalf("bashPath() = %q, want an absolute path (a PATH lookup is hijackable)", got)
	}
	if _, err := os.Stat(got); err != nil {
		t.Fatalf("bashPath() = %q, which does not exist: %v", got, err)
	}
	t.Logf("bashPath() = %s", got)
}

func TestPowerShellPathUsesSystemRoot(t *testing.T) {
	t.Setenv("SystemRoot", `C:\Windows`)
	want := filepath.Join(`C:\Windows`, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
	got := powerShellPath()
	if got != want {
		t.Fatalf("powerShellPath() = %q, want %q", got, want)
	}
	t.Logf("powerShellPath() = %s", got)
}

func TestPowerShellPathFallsBackWhenSystemRootUnset(t *testing.T) {
	t.Setenv("SystemRoot", "")
	want := filepath.Join(`C:\Windows`, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
	got := powerShellPath()
	if got != want {
		t.Fatalf("powerShellPath() with SystemRoot unset = %q, want the %q fallback", got, want)
	}
	t.Logf("fallback powerShellPath() = %s", got)
}
