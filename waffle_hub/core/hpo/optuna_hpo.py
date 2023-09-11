from waffle_hub import TaskType
from waffle_hub.core.hpo.base_hpo import BaseHPO
from waffle_hub.hub import Hub


class OptunaHPO(BaseHPO):
    def __init__(self) -> None:
        self.scheduler = None
        self.pruner = None
        self.hpo_method = None

    @property
    def scheduler(
        self,
    ):
        pass

    @property
    def pruner(
        self,
    ):
        pass

    # hpo_method will be defined by optuna method
    # TODO : str -> optuna hpo method class
    @property
    def hpo_method(
        self,
    ):
        pass

    @hpo_method.setter
    def hpo_method(
        self,
    ):
        pass

    @pruner.setter
    def pruner(
        self,
    ):
        pass

    @scheduler.setter
    def scheduler(
        self,
    ):
        pass

    def create_studies(
        self,
    ):
        # create studies using hub
        pass

    def load_studies(
        self,
    ):
        # load studies using db
        # db must be located in study hub
        pass

    def visualization(
        self,
    ):
        # sets visualize opt
        pass

    def objectives(self, objectives):
        # using hub
        # using pruner, scheduler
        if self.hub.task == TaskType.CLASSIFICATION:
            pass
        elif self.hub.task == TaskType.OBJECT_DETECTION:
            pass
        else:
            raise NotImplementedError(f"{self.hub.task} is not implemented")

    def hpo(
        self,
        hub: Hub,
        percent: float = 0.3,
        search_opt: str = "m",
        *args,
        **kwargs,
    ):
        pass
