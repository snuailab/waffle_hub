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
        self._failed = False
        self._progress = 0
        self._start_time = time.time()

    def get_progress(self) -> float:
        """Get the progress of the task. (0 ~ 1)"""
        return self._progress

    def is_finished(self) -> bool:
        """Check if the task has finished."""
        return self._finished

    def is_failed(self) -> bool:
        """Check if the task has failed."""
        return self._failed

    def set_failed(self):
        """Set the task as failed."""
        self._failed = True

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
            self._progress = step / self._total_steps
        else:
            self._progress = step / self._total_steps

    def force_finish(self):
        """Force the task to end."""
        self._finished = True
        self._progress = 1.0

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

        self._get_metric_func = get_metric_func

    def get_progress(self) -> float:
        """Get the progress of the task. (0 ~ 1)"""
        if not self._finished:
            metrics = self._get_metric_func()
            if len(metrics) == 0:
                return 0
            self.update(len(metrics))
            self._progress = len(metrics) / self._total_steps
            return self._progress
        else:
            return 1.0


class EvaluateCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int):
        super().__init__(total_steps)


class InferenceCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int):
        super().__init__(total_steps)


class ExportCallback(ThreadProgressCallback):
    def __init__(self, total_steps: int):
        super().__init__(total_steps)
