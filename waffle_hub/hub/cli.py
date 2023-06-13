import fire

from waffle_hub.hub import get_hub


if __name__ == '__main__':
    fire.Fire(get_hub("ultralytics"), name="ultralytics")
