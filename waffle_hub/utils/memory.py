import torch


def use_cuda(device):
    """For gpu memory decorator"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.init()  # for thread
            func(*args, **kwargs)
            if torch.cuda.is_available() and device != "cpu":
                # Memory free
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()

        return wrapper

    return decorator
