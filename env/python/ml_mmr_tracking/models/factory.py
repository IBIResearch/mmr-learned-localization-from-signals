import typing as tp


class ModelFactory:
    models = {}

    @classmethod
    def add_model(cls, model_id):
        def decorator(model_cls):
            cls.models[model_id] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create_model(cls, model_id: int, params: tp.Optional[dict] = None):
        model_cls = cls.models.get(model_id)
        if model_cls:
            return model_cls(**(params or {}))
        else:
            raise ValueError(f"Model with ID '{model_id}' does not exist.")
