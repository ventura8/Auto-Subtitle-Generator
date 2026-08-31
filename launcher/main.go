package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func getExecutableDir() (string, error) {
	exePath, err := os.Executable()
	if err != nil {
		return "", err
	}
	exePath, err = filepath.EvalSymlinks(exePath)
	if err != nil {
		return "", err
	}
	return filepath.Dir(exePath), nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func findVenvPython(baseDir string) string {
	candidates := []string{
		filepath.Join(baseDir, ".venv", "Scripts", "python.exe"),
		filepath.Join(baseDir, ".venv", "bin", "python"),
		filepath.Join(baseDir, ".venv", "bin", "python3"),
		filepath.Join(baseDir, ".venv", "bin", "python3.12"),
	}

	for _, candidate := range candidates {
		if fileExists(candidate) {
			return candidate
		}
	}
	return ""
}

func runAutoInstall(baseDir string) error {
	fmt.Println("==================================================================")
	fmt.Println("Auto-Subtitle-Generator: Virtual environment not found.")
	fmt.Println("Starting automated environment and dependency installation...")
	fmt.Println("==================================================================")

	var cmd *exec.Cmd

	if runtime.GOOS == "windows" {
		psScript := filepath.Join(baseDir, "install_dependencies.ps1")
		if !fileExists(psScript) {
			return fmt.Errorf("installer script not found: %s", psScript)
		}
		cmd = exec.Command("powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", psScript)
	} else {
		shScript := filepath.Join(baseDir, "install_dependencies.sh")
		if !fileExists(shScript) {
			return fmt.Errorf("installer script not found: %s", shScript)
		}
		cmd = exec.Command("bash", shScript)
	}

	cmd.Dir = baseDir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}

func main() {
	baseDir, err := getExecutableDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error determining executable directory: %v\n", err)
		os.Exit(1)
	}

	venvPy := findVenvPython(baseDir)
	if venvPy == "" {
		if installErr := runAutoInstall(baseDir); installErr != nil {
			fmt.Fprintf(os.Stderr, "Installation failed: %v\n", installErr)
			os.Exit(1)
		}

		venvPy = findVenvPython(baseDir)
		if venvPy == "" {
			fmt.Fprintf(os.Stderr, "Error: Environment setup completed but virtualenv python was not found in %s\n", baseDir)
			os.Exit(1)
		}
	}

	mainScript := filepath.Join(baseDir, "auto_subtitle.py")
	if !fileExists(mainScript) {
		fmt.Fprintf(os.Stderr, "Error: Main script not found at %s\n", mainScript)
		os.Exit(1)
	}

	args := append([]string{mainScript}, os.Args[1:]...)
	cmd := exec.Command(venvPy, args...)
	cmd.Dir = baseDir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "Execution error: %v\n", err)
		os.Exit(1)
	}
}
