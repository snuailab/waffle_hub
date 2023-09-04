import fire
from waffle_utils.log import initialize_logger

from waffle_hub.dataset import Dataset
from waffle_hub.utils.base_cli import BaseCLI, cli

initialize_logger("dataset.log", root_level="INFO", console_level="INFO", file_level="DEBUG")


class DatasetInstance(BaseCLI):
    # not required for cli mode
    restrict_method_names = [
        "load",
        "new",
        "add_annotations",
        "add_categories",
        "add_images",
        "add_predictions",
        "create_index",
        "initialize",
        "save_dataset_info",
    ]

    def __init__(self, name: str, root_dir: str = None):
        self.dataset = None
        if name in Dataset.get_dataset_list(root_dir):
            self.dataset = Dataset.load(name, root_dir=root_dir)
        else:
            raise ValueError(f"Dataset {name} does not exist.")

        super().__init__()

    def get_object(self):
        return self.dataset


def main():
    fire.Fire(cli(Dataset, DatasetInstance), serialize=str)


if __name__ == "__main__":
    main()
