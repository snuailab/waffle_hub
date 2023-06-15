import inspect

import fire

from waffle_hub.dataset import Dataset
from waffle_hub.utils.base_cli import BaseCLI


class DatasetInstance(BaseCLI):
    def __init__(self, name: str, root_dir: str = None):
        self.dataset = None
        if name in Dataset.get_dataset_list():
            self.dataset = Dataset.load(name, root_dir=root_dir)
        else:
            raise ValueError(f"Dataset {name} does not exist.")

        super().__init__()

    def get_object(self):
        return self.dataset


def switch_type(command: str, **kwargs):

    class_method_names = [
        function_name for function_name, _ in inspect.getmembers(Dataset, predicate=inspect.ismethod)
    ]

    if command in class_method_names:
        if kwargs.get("help", False):
            return getattr(Dataset, command).__doc__
        else:
            try:
                return getattr(Dataset, command)(**kwargs)
            except TypeError:
                raise TypeError(
                    "You've given wrong arguments. Please check the help message below.\n\n"
                    + f"input: {kwargs}\n\n"
                    + getattr(Dataset, command).__doc__
                )
            except Exception as e:
                raise e
    elif kwargs.get("name", None) is not None:
        name = kwargs.pop("name")
        root_dir = kwargs.pop("root_dir", None)
        dataset_instance = DatasetInstance(name, root_dir=root_dir)

        instance_method_names = [
            function_name
            for function_name, _ in inspect.getmembers(dataset_instance, predicate=inspect.ismethod)
        ]
        if command not in instance_method_names:
            raise ValueError(f"Command {command} does not exist.")

        if kwargs.get("help", False):
            return getattr(dataset_instance, command).__doc__
        else:
            try:
                return getattr(dataset_instance, command)(**kwargs)
            except TypeError:
                raise TypeError(
                    "You've given wrong arguments. Please check the help message below.\n\n"
                    + f"input: {kwargs}\n\n"
                    + getattr(dataset_instance, command).__doc__
                )
            except Exception as e:
                raise e
    else:
        raise ValueError(f"Command {command} does not exist.")


def main():
    fire.Fire(switch_type, serialize=str)


if __name__ == "__main__":
    main()
