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

    @property
    def progress(self) -> float:
        """The progress of the task. (0.0 ~ 1.0)"""
        return self._progress

    @property
    def finished(self) -> bool:
        """Whether the task has ended."""
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


class ResultCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int, result_dir: str):
        super().__init__(total_steps)

        self._results: list[dict] = []
        self._result_dir: str = result_dir

    def update(self, step: int, results: dict):
        super().update(step)
        self._results.append(results)

    @property
    def results(self) -> list[dict]:
        return self._results

    @property
    def result_dir(self) -> str:
        return self._result_dir


class TrainCallback(ResultCallback):
    pass


class InferenceCallback(ResultCallback):
    pass


class ExportCallback(ResultCallback):
    pass
