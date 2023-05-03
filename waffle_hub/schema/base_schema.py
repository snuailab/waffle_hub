import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
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
        d = self.to_dict()
        for k, v in d.items():
            try:
                json.dumps(v)
            except:
                d[k] = str(v)

        io.save_json(d, save_path, create_directory=True)

    def save_yaml(self, save_path):
        d = self.to_dict()
        for k, v in d.items():
            try:
                json.dumps(v)  # yaml does not catch any error, so use json instead
            except:
                d[k] = str(v)
        io.save_yaml(d, save_path, create_directory=True)

    @classmethod
    def load(cls, load_path):
        load_path = Path(load_path)
        if load_path.suffix == ".json":
            config = io.load_json(load_path)
        elif load_path.suffix == ".yaml":
            config = io.load_yaml(load_path)
        return cls(**config)
