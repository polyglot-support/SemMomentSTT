import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print("\nCUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    CUDA cores: {props.multi_processor_count * 64}")  # Approximate for most GPUs
    else:
        print("\nPossible issues:")
        print("1. CUDA toolkit not installed")
        print("2. PyTorch not installed with CUDA support")
        print("3. Incompatible CUDA version")
        print("4. GPU drivers not properly installed")
        
        print("\nTry reinstalling PyTorch with:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    check_cuda()
