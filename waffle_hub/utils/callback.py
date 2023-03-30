"""
Pseudo code for Training

Use Case 1.
# start train
train(..., hold=True)
# end train

Use Case 2.
# start train in another thread
callback = train(..., hold=False)

# wait for train to finish
while callback.end():
    print(callback.get_progress())

# end train

Sample code for Training
def train(hold: bool=True) -> Union[None, TrainCallback]:

    if hold:
        # train and return
        return
    else:
        # train in another thread and return callback
        return callback

Sample code for Inference
def inference(hold: bool=True) -> Union[None, InferenceCallback]:

    if hold:
        # inference and return
        return
    else:
        # inference in another thread and return callback
        return callback
"""
import threading
import time
import warnings


class ThreadProgressCallback:
    def __init__(self, total_steps: int):
        self._total_steps = total_steps

        self._thread = None
        self._finished = False
        self._progress = 0
        self._start_time = time.time()

    def get_progress(self) -> float:
        """Get the progress of the task. (0 ~ 1)"""
        return self._progress

    def is_finished(self) -> bool:
        """Check if the task has finished."""
        return self._finished

    def get_remaining_time(self) -> float:
        """Get the remaining time of the task. (seconds)"""
        elapsed = time.time() - self._start_time
        if self._progress == 0:
            return float("inf")
        return (elapsed / self._progress) - elapsed

    def update(self, step: int):
        """Update the progress of the task. (0 ~ total_steps)"""
        if self._finished:
            warnings.warn("Callback has already ended")
        elif step >= self._total_steps:
            self._finished = True
        else:
            self._progress = step / self._total_steps

    def force_finish(self):
        """Force the task to end."""
        self._finished = True

    def register_thread(self, thread: threading.Thread):
        """Register the thread that is running the task."""
        self._thread = thread

    def start(self):
        """Start the thread that is running the task."""
        if self._thread is not None:
            self._thread.start()

    def join(self):
        """Wait for the thread that is running the task to end."""
        if self._thread is not None:
            self._thread.join()


class TrainCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int, get_metric_func):
        super().__init__(total_steps)

        self._best_model_path: str = None
        self._last_model_path: str = None
        self._result_dir: str = None

        self._get_metric_func = get_metric_func

    @property
    def best_model_path(self) -> str:
        """Get the path of the best model."""
        return self._best_model_path

    @best_model_path.setter
    def best_model_path(self, path: str):
        self._best_model_path = path

    @property
    def last_model_path(self) -> str:
        """Get the path of the last model."""
        return self._last_model_path

    @last_model_path.setter
    def last_model_path(self, path: str):
        self._last_model_path = path

    @property
    def result_dir(self) -> str:
        """Get the path of the result directory."""
        return self._result_dir

    @result_dir.setter
    def result_dir(self, path: str):
        self._result_dir = path

    def get_result(self) -> list[dict]:
        """Get the metrics of the task. (list of dict)"""
        return self._get_metric_func()

    def get_progress(self) -> float:
        """Get the progress of the task. (0 ~ 1)"""
        metrics = self._get_metric_func()
        if len(metrics) == 0:
            return 0
        self.update(len(metrics))
        return len(metrics) / self._total_steps


class InferenceCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int):
        super().__init__(total_steps)

        self._result: list[dict] = []

        self._result_dir: str = None

    @property
    def result_dir(self) -> str:
        """Get the path of the result directory."""
        return self._result_dir

    @result_dir.setter
    def result_dir(self, path: str):
        self._result_dir = path

    def update(self, step: int, result: dict):
        super().update(step)
        self._result.append(result)

    def get_result(self) -> list[dict]:
        """Get the results of the task. (list of dict)"""
        return self._result


class ExportCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int):
        super().__init__(total_steps)

        self._result_file: str = None

    @property
    def result_file(self) -> str:
        """Get the path of the result file."""
        return self._result_file

    @result_file.setter
    def result_file(self, path: str):
        self._result_file = path

    def get_result(self) -> str:
        """Get the path of the result file."""
        return self._result_file
