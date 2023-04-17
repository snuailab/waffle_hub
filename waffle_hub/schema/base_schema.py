from dataclasses import asdict, dataclass
from pathlib import Path

from waffle_utils.file import io


@dataclass
class BaseSchema:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return

    def to_dict(self):
        return asdict(self)

    def save_json(self, save_path):
        io.save_json(self.to_dict(), save_path, create_directory=True)

    def save_yaml(self, save_path):
        io.save_yaml(self.to_dict(), save_path, create_directory=True)

    @classmethod
    def load(cls, load_path):
        load_path = Path(load_path)
        if load_path.suffix == ".json":
            config = io.load_json(load_path)
        elif load_path.suffix == ".yaml":
            config = io.load_yaml(load_path)
        return cls(**config)
