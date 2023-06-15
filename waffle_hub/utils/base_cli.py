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
            for function_name, _ in inspect.getmembers(class_object)
            if not function_name.startswith("_")
        ]
        for method in self.methods:
            setattr(self, method, getattr(class_object, method))


def cli(_class, _instance):
    def switch_type(command: str = None, **kwargs) -> str:
        name = kwargs.get("name", None)

        class_method_names = [
            function_name
            for function_name, _ in inspect.getmembers(_class, predicate=inspect.ismethod)
        ]
        instance_method_names = [
            function_name
            for function_name, _ in inspect.getmembers(_class, predicate=inspect.isfunction)
            if function_name not in class_method_names and not function_name.startswith("_")
        ]
        attribute_names = [
            attribute_name
            for attribute_name, _ in inspect.getmembers(_class)
            if not attribute_name.startswith("_")
            and attribute_name not in (class_method_names + instance_method_names)
        ]

        help_string = (
            "Available Class Methodas:\n"
            + "\n".join(class_method_names)
            + "\n\nAvailable Instance Methods (need to specify --name [Name]):\n"
            + "\n".join(instance_method_names)
            + "\n\nAvailable Attributes: (need to specify --name [Name])\n"
            + "\n".join(attribute_names)
        )

        if command is None or command == "help":
            return help_string

        if kwargs.get("help", False):
            if command in class_method_names:
                return getattr(_class, command).__doc__
            elif command in instance_method_names + attribute_names:
                return (
                    "You need to specify --name [Name] --root_dir [Root Directory of the instance (optional)]\n"
                    + "to use instance_method or to get attribute\n"
                    + getattr(_class, command).__doc__
                )

        elif command in class_method_names:
            try:
                return getattr(_class, command)(**kwargs)
            except Exception as e:
                raise e

        elif command in instance_method_names:
            name = kwargs.pop("name", None)
            if name is None:
                raise ValueError("You need to specify --name [Name]")

            root_dir = kwargs.pop("root_dir", None)
            instance = _instance(name, root_dir=root_dir)

            try:
                return getattr(instance, command)(**kwargs)
            except Exception as e:
                raise e

        elif command in attribute_names:
            name = kwargs.pop("name", None)
            if name is None:
                raise ValueError("You need to specify --name [Name]")

            root_dir = kwargs.pop("root_dir", None)
            instance = _instance(name, root_dir=root_dir)

            try:
                return getattr(instance, command)
            except Exception as e:
                raise e
        else:
            raise ValueError(f"Command {command} does not exist.")

    return switch_type
