import inspect


class BaseCLI:
    def __init__(self):

        self.class_methods = []
        self.instance_methods = []
        self.register_class_methods()
        self.register_instance_methods()

    def get_class(self):
        pass

    def get_instance(self):
        pass

    def register_class_methods(self):
        class_object = self.get_class()

        self.class_methods = [
            function_name
            for function_name, _ in inspect.getmembers(class_object, predicate=inspect.ismethod)
        ]
        for method in self.class_methods:
            setattr(self, method, getattr(class_object, method))

    def register_instance_methods(self):
        instance_object = self.get_instance()

        if instance_object is not None:
            self.instance_methods = [
                function_name
                for function_name, _ in inspect.getmembers(
                    instance_object, predicate=inspect.ismethod
                )
                if function_name not in self.class_methods
            ]
            for method in self.instance_methods:
                setattr(self, method, getattr(instance_object, method))
