import inspect


class BaseCLI:
    def __init__(self):

        self.methods = []
        self.register_methods()

    def get_object(self):
        raise NotImplementedError("get_object method is not implemented.")

    def register_methods(self):
        class_object = self.get_object()

        self.methods = [
            function_name
            for function_name, _ in inspect.getmembers(class_object, predicate=inspect.ismethod)
        ]
        for method in self.methods:
            setattr(self, method, getattr(class_object, method))
