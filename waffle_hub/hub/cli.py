import fire
from waffle_utils.log import initialize_logger

from waffle_hub.hub import Hub
from waffle_hub.utils.base_cli import BaseCLI, cli

initialize_logger("hub.log", root_level="INFO", console_level="INFO", file_level="DEBUG")


class HubInstance(BaseCLI):
    # not required for cli mode
    restrict_method_names = [
        "load",
        "get_default_advance_train_params",
        "get_image_loader",
        "get_model",
        "after_evaluate",
        "after_export",
        "after_inference",
        "after_train",
        "before_evaluate",
        "before_export",
        "before_inference",
        "before_train",
        "evaluating",
        "exporting",
        "inferencing",
        "training",
        "on_evaluate_end",
        "on_evaluate_start",
        "on_export_end",
        "on_export_start",
        "on_inference_end",
        "on_inference_start",
        "on_train_end",
        "on_train_start",
        "save_model_config",
        "save_train_config",
    ]

    def __init__(self, name: str, root_dir: str = None):
        self.hub = None
        if name in Hub.get_hub_list(root_dir):
            self.hub = Hub.load(name, root_dir=root_dir)
        else:
            raise ValueError(f"Hub {name} does not exist.")

        super().__init__()

    def get_object(self):
        return self.hub


def main():
    fire.Fire(cli(Hub, HubInstance), serialize=str)


if __name__ == "__main__":
    main()
