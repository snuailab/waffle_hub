import os
import shutil

from waffle_hub.dataset import Dataset
from waffle_hub.hub import Hub


def delete_folder_contents(folder_path):
    try:
        # 폴더 안의 모든 파일 및 서브폴더를 삭제합니다.
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
        print(f"Contents of '{folder_path}' have been deleted.")
    except Exception as e:
        print(f"Error deleting contents of '{folder_path}': {str(e)}")


if __name__ == "__main__":
    n_trials = 3
    hpo_method = "BOHB"  # You can change this to "TPESampler" or "GridSampler" as needed
    search_space = {
        "lr0": [0.005, 0.05],
        "lrf": [0.001, 0.005],
        "mosaic": [0.6, 1],
        "cos_lr": [True, False],
        "hsv_h": [0.01, 0.02],
        "hsv_s": [0.01, 0.02],
        "hsv_v": [0.01, 0.02],
        "translate": [0.09, 0.11],
        "scale": [0.45, 0.55],
        "mosaic": [0.6, 1],
    }
    # print(hub.from_model_config("test2", "/home/daeun/workspace/waffle_hub/hubs/test1/hpo/trial_0/configs/model.yaml"))
    hub_name = "test2"
    # delete_folder_contents(f"/home/daeun/workspace/waffle_hub/hubs/")
    hub = Hub.new(
        name=hub_name,
        task="classification",
        model_type="yolov8",
        model_size="n",
    )

    dataset = Dataset.load(name="mnist_classification")
    direction = "maximize"

    result = hub.hpo(
        dataset=dataset,
        n_trials=n_trials,
        direction=direction,
        hpo_method=hpo_method,
        search_space=search_space,
        epochs=1,
        image_size=8,
        device="0",
    )
