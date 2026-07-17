import os
import shutil
import subprocess
import sys
import getpass
import dotenv

dotenv.load_dotenv()

import stat

def run_cmd(args, cwd=None, shell=False):
    """Run a shell command and print outputs, raising exception on failure."""
    print(f"Running: {' '.join(args) if isinstance(args, list) else args}")
    result = subprocess.run(args, cwd=cwd, shell=shell, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

def safe_rmtree(path):
    """Recursively delete a directory, clearing read-only flags (needed on Windows for .git files)."""
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            try:
                os.chmod(filename, stat.S_IWRITE)
                os.remove(filename)
            except Exception as e:
                pass
        for name in dirs:
            dirname = os.path.join(root, name)
            try:
                os.chmod(dirname, stat.S_IWRITE)
                os.rmdir(dirname)
            except Exception as e:
                pass
    try:
        os.chmod(path, stat.S_IWRITE)
        os.rmdir(path)
    except Exception as e:
        pass

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dist_hf = os.path.join(root_dir, "dist_hf")
    
    # 1. Retrieve credentials
    hf_token = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME")
    hf_space_name = os.environ.get("HF_SPACE_NAME")
    
    print("=== Hugging Face Local Deployment Script ===")
    if not hf_token:
        print("HF_TOKEN environment variable not found.")
        hf_token = getpass.getpass("Enter your Hugging Face Write Token: ").strip()
    if not hf_username:
        hf_username = input("Enter your Hugging Face Username (e.g. MJ41739): ").strip()
    if not hf_space_name:
        hf_space_name = input("Enter your Hugging Face Space Name (e.g. crack-detection-api): ").strip()
        
    if not hf_token or not hf_username or not hf_space_name:
        print("Error: Username, Space Name, and Token are required.")
        sys.exit(1)

    # 2. Build Frontend
    print("\n--- Step 1: Building Frontend ---")
    frontend_dir = os.path.join(root_dir, "frontend")
    try:
        # Check npm command is available
        run_cmd(["npm", "run", "build"], cwd=frontend_dir, shell=True)
    except Exception as e:
        print(f"Failed to build frontend: {e}")
        print("Please ensure Node.js/npm is installed and try again.")
        sys.exit(1)

    # 3. Prepare Deployment Directory
    print("\n--- Step 2: Preparing Deployment Assets ---")
    if os.path.exists(dist_hf):
        print("Cleaning up old dist_hf folder...")
        safe_rmtree(dist_hf)
    os.makedirs(dist_hf)
    
    # Copy backend, models, utils
    shutil.copytree(os.path.join(root_dir, "backend"), os.path.join(dist_hf, "backend"))
    shutil.copytree(os.path.join(root_dir, "models"), os.path.join(dist_hf, "models"))
    shutil.copytree(os.path.join(root_dir, "utils"), os.path.join(dist_hf, "utils"))
    
    # Copy checkpoints
    checkpoints_src = os.path.join(root_dir, "checkpoints")
    checkpoints_dest = os.path.join(dist_hf, "checkpoints")
    os.makedirs(checkpoints_dest)
    for f in ["custom_cnn_quantized.onnx", "custom_cnn.onnx", "custom_cnn_best.pth"]:
        src_file = os.path.join(checkpoints_src, f)
        if os.path.exists(src_file):
            shutil.copy2(src_file, checkpoints_dest)
            print(f"Copied model checkpoint: {f}")
            
    # Copy frontend build
    shutil.copytree(os.path.join(root_dir, "frontend", "dist"), os.path.join(dist_hf, "frontend", "dist"))
    
    # Copy root config files
    shutil.copy2(os.path.join(root_dir, "Dockerfile"), dist_hf)
    shutil.copy2(os.path.join(root_dir, "requirements.txt"), dist_hf)
    shutil.copy2(os.path.join(root_dir, "README_HF.md"), os.path.join(dist_hf, "README.md"))
    
    # 4. Git Push to Hugging Face
    print("\n--- Step 3: Pushing to Hugging Face Spaces ---")
    try:
        # Git Init
        run_cmd(["git", "init"], cwd=dist_hf)
        
        # Git LFS
        try:
            run_cmd(["git", "lfs", "install"], cwd=dist_hf)
            run_cmd(["git", "lfs", "track", "*.onnx"], cwd=dist_hf)
            run_cmd(["git", "lfs", "track", "*.pth"], cwd=dist_hf)
            run_cmd(["git", "add", ".gitattributes"], cwd=dist_hf)
        except Exception:
            print("Warning: git-lfs commands failed. Ensure git-lfs is installed on your path if uploading large files.")
            
        run_cmd(["git", "checkout", "-b", "main"], cwd=dist_hf)
        run_cmd(["git", "config", "user.email", f"{hf_username}@users.noreply.huggingface.co"], cwd=dist_hf)
        run_cmd(["git", "config", "user.name", hf_username], cwd=dist_hf)
        
        run_cmd(["git", "add", "-A"], cwd=dist_hf)
        run_cmd(["git", "commit", "-m", "deploy: local update to spaces container"], cwd=dist_hf)
        
        # Add remote and push
        remote_url = f"https://{hf_username}:{hf_token}@huggingface.co/spaces/{hf_username}/{hf_space_name}"
        run_cmd(["git", "remote", "add", "hf", remote_url], cwd=dist_hf)
        run_cmd(["git", "push", "--force", "hf", "main"], cwd=dist_hf)
        
        print("\n=== SUCCESS: Deployed directly to Hugging Face Spaces! ===")
    except Exception as e:
        print(f"\nDeployment failed during push phase: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
