import fire
from waffle_utils.log import initialize_logger

from waffle_hub.hub import Hub
from waffle_hub.utils.base_cli import BaseCLI, cli

initialize_logger("hub.log", root_level="INFO", console_level="INFO", file_level="DEBUG")


class HubInstance(BaseCLI):
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
