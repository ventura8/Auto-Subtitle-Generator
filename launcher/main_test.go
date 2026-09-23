package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBashPathIsAbsoluteAndExists(t *testing.T) {
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
	got := powerShellPath()
	if !strings.HasSuffix(got, "powershell.exe") {
		t.Fatalf("powerShellPath() = %q, want it to end in powershell.exe", got)
	}
	if !strings.Contains(got, "System32") || !strings.Contains(got, "WindowsPowerShell") {
		t.Fatalf("powerShellPath() = %q, want the System32 interpreter location", got)
	}
	if got == "powershell.exe" {
		t.Fatal("powerShellPath() returned a bare name, which resolves through PATH")
	}
	t.Logf("powerShellPath() = %s", got)
}

func TestPowerShellPathFallsBackWhenSystemRootUnset(t *testing.T) {
	t.Setenv("SystemRoot", "")
	got := powerShellPath()
	if !strings.Contains(got, "Windows") {
		t.Fatalf("powerShellPath() = %q, want the C:\\Windows fallback", got)
	}
	t.Logf("fallback powerShellPath() = %s", got)
}
