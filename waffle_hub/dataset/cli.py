import fire

from waffle_hub.dataset import Dataset


if __name__ == '__main__':
    fire.Fire(Dataset, name="dataset")