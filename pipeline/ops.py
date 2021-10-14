from kfp import dsl, components

IMAGE_PREFIX = 'fs2'
DATA_PATH = '/opt/storage'
# TODO: refactor this
PVC_NAME = 'local-pvc'
DOCKER_HUB_USERNAME = 'lunarbridge' 


def mfa_aligner_op(current_data_path: str) -> dsl.ContainerOp:
    '''parse txt and generate TextGrids'''
    name = 'mfa-aligner'

    return dsl.ContainerOp(
        name=f'{name}',
        image=f'docker.io/{DOCKER_HUB_USERNAME}/{IMAGE_PREFIX}-{name}',
        arguments=['--data-base-path', DATA_PATH,
                   '--current-data-path', current_data_path],
        pvolumes={f'{DATA_PATH}': dsl.PipelineVolume(pvc=PVC_NAME)}
    )


def mfa_aligner_op(current_data_path: str) -> dsl.ContainerOp:
    '''parse txt and generate TextGrids'''
    name = 'mfa-aligner'

    return dsl.ContainerOp(
        name=f'{name}',
        image=f'docker.io/{DOCKER_HUB_USERNAME}/{IMAGE_PREFIX}-{name}',
        arguments=['--data-base-path', DATA_PATH,
                   '--current-data-path', current_data_path],
        pvolumes={f'{DATA_PATH}': dsl.PipelineVolume(pvc=PVC_NAME)}
    )


def prepare_align_op(current_data_path: str) -> dsl.ContainerOp:
    '''prepare align'''
    name = 'prepare-align'

    return dsl.ContainerOp(
        name=f'{name}',
        image=f'docker.io/{DOCKER_HUB_USERNAME}/{IMAGE_PREFIX}-{name}',
        arguments=['--data-base-path', current_data_path],
        pvolumes={f'{DATA_PATH}': dsl.PipelineVolume(pvc=PVC_NAME)}
    )


def prepare_preprocess_op() -> dsl.ContainerOp:
    '''check new data exists and move new data to work directory and checks data duplication if is that case'''
    name = 'prepare-preprocess'

    return dsl.ContainerOp(
        name=f'{name}',
        image=f'docker.io/{DOCKER_HUB_USERNAME}/{IMAGE_PREFIX}-{name}',
        arguments=['--data-base-path', DATA_PATH],
        file_outputs={'current_data_path': '/tmp/curr-data-path',
                      'is_new_data_exist': '/tmp/is-new-data-exist',
                      'duplicated_files': '/tmp/data-dupl-relpaths.json',
                      'preprocess_target_files': '/tmp/data-relpaths.json'},
        pvolumes={f'{DATA_PATH}': dsl.PipelineVolume(pvc=PVC_NAME)}
    )


def preprocess_op(current_data_path: str) -> dsl.ContainerOp:
    '''preprocess on a Wav and TextGrid file'''
    name = 'preprocess'

    return dsl.ContainerOp(
        name=f'{name}',
        image=f'docker.io/{DOCKER_HUB_USERNAME}/{IMAGE_PREFIX}-{name}',
        arguments=['--data-base-path', current_data_path],
        pvolumes={f'{DATA_PATH}': dsl.PipelineVolume(pvc=PVC_NAME)}
    )


def train_op(current_data_path: str) -> dsl.ContainerOp:
    '''preprocess on a Wav and TextGrid file'''
    name = 'train'

    return dsl.ContainerOp(
        name=f'{name}',
        image=f'docker.io/{DOCKER_HUB_USERNAME}/{IMAGE_PREFIX}-{name}',
        arguments=['--current-data-path', current_data_path],
        pvolumes={f'{DATA_PATH}': dsl.PipelineVolume(pvc=PVC_NAME)}
    )
