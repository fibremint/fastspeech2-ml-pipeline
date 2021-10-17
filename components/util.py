
# ref: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_ops(docker_hub_usename: str = None, docker_image_prefix: str = None):
    if not docker_hub_usename or not docker_image_prefix:
        raise AttributeError()

    from kfp import components

    from . import ops
    from .configs import op_config

    loaded_ops = dict()


    for fn_component_name in ops.__all__:
        base_image = None
        packages_to_install = None

        if fn_component_name in op_config and op_config[fn_component_name]:
            if not 'base_image' in op_config[fn_component_name]:
                print(f"WARN: key 'base_image' is not exist in {fn_component_name}") 
            else:
                base_image = op_config[fn_component_name]['base_image'].format(docker_hub_usename, docker_image_prefix)

            if 'packages_to_install' in op_config[fn_component_name]:
                packages_to_install = op_config[fn_component_name]['packages_to_install']

        fn = getattr(ops, fn_component_name)
        loaded_ops[fn_component_name] = components.create_component_from_func(fn, 
                                                                              base_image=base_image,
                                                                              packages_to_install=packages_to_install)

    loaded_ops = dotdict(loaded_ops)

    return loaded_ops
