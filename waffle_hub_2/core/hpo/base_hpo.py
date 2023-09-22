from abc import abstractmethod

from waffle_hub.hub import Hub


class BaseHPO:
    def __init__(
        self,
        percent: float = 0.3,
        search_opt: str = "m",
        method: str = "BOHB",
        *args,
        **kwargs,
    ):
        """Base HPO

        Args:
            hub (Hub): Hub
            percent (float, optional) : search space scop. Defaults to 0.3
            search_opt (str, optional) : search params option. Default to medium
                m(medium) : only learning rate
                l(long) : learning rate and augmentation params
        """
        self.percent = percent
        self.search_opt = search_opt
        self.default_params = None
        self.scope_params = None

    # default params, scope_params, method -> getter setter
    # initialize them by params
    @property
    def _default_params(self) -> dict:
        # property
        # default param 선언
        pass

    @property
    def _scope_params(self) -> dict:
        # property
        # percent 범위대로 계산 후 반환
        # search_opt, percent 계산 후 반환
        pass

    @_default_params.setter
    def _default_params(
        self,
    ):
        # TODO: define default params
        pass

    @_scope_params.setter
    def _scope_params(
        self,
    ):
        # TODO: if selecting_args are not None :
        pass
