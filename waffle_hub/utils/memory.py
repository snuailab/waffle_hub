import torch


def device_context(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):  # for torch 2.0 versions
                # https://github.com/pytorch/pytorch/issues/95668
                torch._C._cuda_clearCublasWorkspaces()
            # Memory free
            torch.cuda.empty_cache()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                # https://github.com/pytorch/pytorch/issues/95668
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        return result

    return wrapper
