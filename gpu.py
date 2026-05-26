import os
import sys
import subprocess

def check_system_gpu():
    print("=== 1. Checking System GPU (via nvidia-smi) ===")
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout)
            return True
        else:
            print("nvidia-smi returned an error:")
            print(res.stderr)
            return False
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
        return False

def check_pytorch_cuda():
    print("=== 2. Checking PyTorch CUDA Support ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available in PyTorch: {cuda_available}")
        if cuda_available:
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Active Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Active Device Capability: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("CUDA is NOT available in PyTorch. This is likely because a CPU-only build is installed.")
            return False
    except ImportError:
        print("PyTorch is not installed in the current environment.")
        return False

def setup_cuda_dlls():
    """
    On Windows, ctranslate2 (used by faster-whisper) needs CUDA runtime libraries (cuBLAS, cuDNN).
    If they are installed in site-packages (via nvidia-*-cu12 packages), we add their bin directories to PATH.
    """
    print("=== 3. Configuring CUDA DLL Paths ===")
    venv_path = sys.prefix
    site_packages = os.path.join(venv_path, "Lib", "site-packages")
    if not os.path.exists(site_packages):
        print(f"Could not locate site-packages directory at {site_packages}")
        return
    
    nvidia_dirs = [
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
        os.path.join(site_packages, "nvidia", "cuda_cupti", "bin"),
        os.path.join(site_packages, "nvidia", "cufft", "bin"),
        os.path.join(site_packages, "nvidia", "curand", "bin"),
        os.path.join(site_packages, "nvidia", "cusolver", "bin"),
        os.path.join(site_packages, "nvidia", "cusparse", "bin"),
        os.path.join(site_packages, "nvidia", "nccl", "bin"),
        os.path.join(site_packages, "nvidia", "nvtx", "bin"),
    ]
    
    added_paths = []
    for d in nvidia_dirs:
        if os.path.exists(d):
            if d not in os.environ["PATH"]:
                os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
                added_paths.append(d)
                
    if added_paths:
        print("Successfully added the following NVIDIA package directories to PATH:")
        for p in added_paths:
            print(f"  - {p}")
    else:
        print("No Python-packaged NVIDIA DLL directories found in site-packages.")
        print("Ensure you have installed packages like `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`.")

def test_faster_whisper_gpu():
    print("=== 4. Testing Faster-Whisper on GPU ===")
    setup_cuda_dlls()
    try:
        from faster_whisper import WhisperModel
        print("Attempting to load small Whisper model on GPU (float16)...")
        # Use a small model for testing
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        print("SUCCESS: Faster-Whisper loaded successfully on GPU!")
        return True
    except Exception as e:
        print(f"FAILED to load Faster-Whisper on GPU: {e}")
        print("\nPossible solutions:")
        print("1. If it says 'Could not load library ...', you might need to install CUDA Toolkit 12.x and cuDNN manually, or copy DLLs.")
        print("2. Ensure PyTorch is compiled with CUDA support.")
        return False

def check_env_file():
    print("=== 5. Checking Backend/.env Configuration ===")
    env_path = os.path.join("Backend", ".env")
    if not os.path.exists(env_path):
        print(f"Backend/.env file not found at {env_path}")
        return
        
    with open(env_path, "r") as f:
        lines = f.readlines()
        
    use_gpu_found = False
    for line in lines:
        if line.strip().startswith("USE_GPU"):
            print(f"Current setting: {line.strip()}")
            use_gpu_found = True
            break
            
    if not use_gpu_found:
        print("USE_GPU setting not found in Backend/.env.")

def update_env_to_gpu():
    env_path = os.path.join("Backend", ".env")
    if not os.path.exists(env_path):
        print(f"Backend/.env file not found at {env_path}")
        return
        
    with open(env_path, "r") as f:
        content = f.read()
        
    if "USE_GPU=false" in content:
        content = content.replace("USE_GPU=false", "USE_GPU=true")
        with open(env_path, "w") as f:
            f.write(content)
        print("Updated Backend/.env to set USE_GPU=true.")
    elif "USE_GPU=true" in content:
        print("Backend/.env is already configured with USE_GPU=true.")
    else:
        # Append it
        with open(env_path, "a") as f:
            f.write("\n# GPU Configuration\nUSE_GPU=true\n")
        print("Appended USE_GPU=true to Backend/.env.")

if __name__ == "__main__":
    has_gpu = check_system_gpu()
    if not has_gpu:
        print("\n[WARNING] No NVIDIA GPU detected. You cannot run this project on GPU.")
        sys.exit(1)
        
    has_torch_cuda = check_pytorch_cuda()
    
    if not has_torch_cuda:
        print("\n[INSTRUCTION] Please run the following command to reinstall PyTorch with CUDA support:")
        print("  venv\\Scripts\\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
        sys.exit(1)
        
    whisper_gpu = test_faster_whisper_gpu()
    
    check_env_file()
    if whisper_gpu:
        print("\nYour system and virtual environment are fully ready for GPU acceleration!")
        confirm = input("Would you like to configure Backend/.env to use GPU? (y/n): ").strip().lower()
        if confirm == 'y':
            update_env_to_gpu()
    else:
        print("\nGPU is available in PyTorch, but Faster-Whisper failed to run on GPU.")
