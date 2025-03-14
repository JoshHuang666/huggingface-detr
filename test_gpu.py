import torch

def test_gpu():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        # Create a random tensor and transfer it to GPU
        tensor = torch.rand(3, 3).cuda()
        print("Random Tensor:\n", tensor)
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    test_gpu()