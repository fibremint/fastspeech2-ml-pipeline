from kfp import dsl, Client, compiler, components
from kubernetes import client

from pipeline import ops

import os


def print_fn(msg) -> None:
    print(msg)


print_op = components.create_component_from_func(print_fn)


@dsl.pipeline(name='fastspeech2-fine-tuning-pipeline')
def fs2_pipeline():
    # cpu_limit = str((os.cpu_count() - 3) * 1000) + 'm'
    cpu_limit = '2'

    prepare_preprocess_task = ops.prepare_preprocess_op()
    curr_data_path = prepare_preprocess_task.outputs['current_data_path']
    is_new_data_exist = prepare_preprocess_task.outputs['is_new_data_exist']

    with dsl.Condition(is_new_data_exist == 'True', name='is-new-data-exist'):
        prepare_align_task = ops.prepare_align_op(curr_data_path)
        prepare_align_task.after(prepare_preprocess_task)

        mfa_align_task = ops.mfa_aligner_op(curr_data_path)
        mfa_align_task.set_cpu_request(cpu_limit)
        mfa_align_task.after(prepare_align_task)

        preprocess_task = ops.preprocess_op(curr_data_path)
        preprocess_task.after(mfa_align_task)

        train_task = ops.train_op(curr_data_path)
        train_task.set_memory_limit('4Gi')
        train_task.set_gpu_limit(1)
        train_task.after(preprocess_task)

    with dsl.Condition(is_new_data_exist == 'False', name='is-not-new-data-exist'):
        print_op('new data is not exist')


if __name__ == '__main__':
    # client = Client(host='http://192.168.0.101:8080')
    # print(client.list_experiments())
    compiler.Compiler().compile(pipeline_func=fs2_pipeline, package_path='pipeline.yaml')