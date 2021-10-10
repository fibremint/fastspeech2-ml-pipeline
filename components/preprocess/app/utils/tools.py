import inspect


def parse_kwargs(model_cls, **kwargs):
    """
    Parse matched arguments in given class
    :param model_cls: module class
    :param kwargs: raw arguments
    :return: parsed(filtered) arguments
    """
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(model_cls).args}
    return kwargs
