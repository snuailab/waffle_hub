from abc import abstractmethod
from pathlib import Path

from tabulate import tabulate

from waffle_hub import get_backends


class BaseHub:
    def __init__(self):
        pass

    @classmethod
    def get_available_backends(cls) -> str:
        """Available backends"""
        backends = get_backends()

        table_data = []
        for name, versions in backends.items():
            for i, version in enumerate(versions):
                table_data.append([name if i == 0 else "", version])

        table = str(
            tabulate(
                table_data,
                headers=["Backend", "Version"],
                tablefmt="simple_outline",
            )
        )
        return table

    @classmethod
    def is_available_backend(cls, name: str, version: str) -> bool:
        """Check if backend is available

        Args:
            name (str): backend name
            version (str): backend version

        Returns:
            bool: is available?
        """
        backends = get_backends()

        return (name in backends) and (version in backends[name])

    @property
    @abstractmethod
    def result_dir(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def log_file(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def remove(self):
        raise NotImplementedError

    @abstractmethod
    def get_progress(self):
        raise NotImplementedError

    @abstractmethod
    def get_status(self):
        raise NotImplementedError

    @abstractmethod
    def get_results(self):
        raise NotImplementedError

    @abstractmethod
    def get_log(self):
        raise NotImplementedError
