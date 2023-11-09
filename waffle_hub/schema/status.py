from dataclasses import dataclass

from waffle_hub.schema.base_schema import BaseSchema


@dataclass
class Status(BaseSchema):
    status: str = None
    error_type: str = None
    error_msg: str = None
    step: int = None
    total_step: int = None


class StatusController:
    def __init__(self, save_path):
        self.save_path = save_path
        self.status_cls = Status()
        self.status_cls.status = "INIT"
        self.save()

    def save(self):
        self.status_cls.save_json(self.save_path)

    def set_total_step(self, total_step):
        self.status_cls.total_step = total_step
        self.status_cls.step = 0
        self.save()

    def set_current_step(self, step):
        self.status_cls.step = step
        self.save()

    def set_failed(self, e):
        self.status_cls.status = "FAILED"
        self.status_cls.error_type = e.__class__.__name__
        self.status_cls.error_msg = e
        self.save()

    def set_success(self):
        self.status_cls.status = "SUCCESS"
        self.status_cls.step = self.status_cls.total_step
        self.status_cls.error_type = None
        self.status_cls.error_msg = None
        self.save()

    def set_running(self):
        self.status_cls.status = "RUNNING"
        self.save()

    def set_stopped(self, e):
        self.status_cls.status = "STOPPED"
        self.status_cls.error_type = e.__class__.__name__
        self.status_cls.error_msg = e
        self.save()

    def set_status(self, status):  # for other status
        self.status_cls.status = status
        self.save()
