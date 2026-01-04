
import subprocess
import os
import sys
import glob

def run_e2e_test():
    print("--- Running End-to-End Test ---")
    
    # Find the video file in input folder
    input_dir = os.path.join(os.getcwd(), "input")
    video_files = glob.glob(os.path.join(input_dir, "*.mkv")) + glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not video_files:
        print("No video file found in input directory.")
        return
        
    video_path = video_files[0]
    print(f"Using video: {video_path}")
    
    # Command to run (limiting to 1 file for now?)
    # auto_subtitle.py handles single file argument
    cmd = [sys.executable, "auto_subtitle.py", video_path]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("FAILED: Process exited with error code.")
    else:
        print("SUCCESS: Process completed.")
        
    # Check log file for GPU usage
    log_file = "subtitle_gen.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            content = f.read()
            if "GPU (FP16) Verified" in content:
                print("VERIFIED: GPU was used for translation.")
            elif "GPU Failed" in content or "CUDA not detected" in content:
                print("FAILURE: GPU was NOT used for translation.")
            else:
                print("UNKNOWN: Could not determine GPU usage from logs.")
    else:
        print("FAILURE: Log file not found.")

if __name__ == "__main__":
    run_e2e_test()
