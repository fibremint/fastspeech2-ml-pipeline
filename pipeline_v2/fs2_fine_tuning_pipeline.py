import argparse
import datetime
import os
from pathlib import Path

from kfp import compiler, components, dsl

from components_v2.util import load_ops


@components.func_to_container_op
def print_op(msg) -> None:
    print(msg)


def build_fine_tuning_pipeline(docker_hub_username, storage_pvc_name, docker_image_prefix, storage_mount_path, compiled_component_output_path):
    # TODO: train, eval with argument
    @dsl.pipeline(name='fs2 fine tuning')
    def pipeline(train_max_step: int = 1000, eval_max_step: int = 1000, batch_size: int = 8, model_save_interval: int = 200):
        ops = load_ops(docker_hub_usename=docker_hub_username,
                       docker_image_prefix=docker_image_prefix)

        # init workflow
        init_workflow_task = ops.init_workflow(data_base_path=storage_mount_path)
        init_workflow_task.add_pvolumes({
            storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
        })
        with dsl.Condition(init_workflow_task.outputs['is_new_data_exist'] == True, name='new-data-exists'):
            # prepare align
            prepare_align_task = ops.prepare_align(current_data_path=init_workflow_task.outputs['current_data_path'],
                                                   dataset_name='vctk')
            prepare_align_task.add_pvolumes({
                storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
            })
            prepare_align_task.after(init_workflow_task)
            # mfa align
            mfa_align_task = ops.mfa_align(data_base_path=storage_mount_path,
                                           current_data_path=init_workflow_task.outputs['current_data_path'])
            mfa_align_task.add_pvolumes({
                storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
            })
            mfa_align_task.after(prepare_align_task)
            # preprocess
            preprocess_task = ops.preprocess(data_base_path=storage_mount_path,
                                             current_data_path=init_workflow_task.outputs['current_data_path'])
            preprocess_task.add_pvolumes({
                storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
            })
            preprocess_task.after(mfa_align_task)
            # train
            train_task = ops.train(data_base_path=storage_mount_path,
                                   current_data_path=init_workflow_task.outputs['current_data_path'],
                                   train_max_step=train_max_step, batch_size=batch_size, model_save_interval=model_save_interval)
            train_task.add_pvolumes({
                storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
            })
            train_task.set_gpu_limit(1)
            train_task.after(preprocess_task)
            # evaluate
            evaluate_task = ops.evaluate(current_data_path=init_workflow_task.outputs['current_data_path'],
                                         data_ref_paths=train_task.outputs['data_ref_paths'],
                                         eval_max_step=eval_max_step, batch_size=batch_size)
            evaluate_task.add_pvolumes({
                storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
            })
            evaluate_task.set_gpu_limit(1)
            evaluate_task.after(train_task)
            # check deployable
            check_deployable_task = ops.check_deployable(data_base_path=storage_mount_path,
                                                         target_data_path=evaluate_task.outputs['train_finished_data_path'])
            check_deployable_task.add_pvolumes({
                storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
            })
            check_deployable_task.after(evaluate_task)

            with dsl.Condition(check_deployable_task.output == True, name='is-optimal-than-before'):
                # update optimal checkpoint
                update_optimal_checkpoint = ops.update_optimal_checkpoint(data_base_path=storage_mount_path,
                                                                          target_data_path=evaluate_task.outputs['train_finished_data_path'])
                update_optimal_checkpoint.add_pvolumes({
                    storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
                })
                update_optimal_checkpoint.after(check_deployable_task)
                # export model
                export_model_task = ops.export_model(data_base_path=storage_mount_path)
                export_model_task.add_pvolumes({
                    storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
                })
                export_model_task.after(update_optimal_checkpoint)
                # deploy
                deploy_task = ops.deploy(data_base_path=storage_mount_path,
                                         model_version=export_model_task.outputs['model_version'])
                deploy_task.add_pvolumes({
                    storage_mount_path: dsl.PipelineVolume(pvc=storage_pvc_name)
                })
                deploy_task.after(export_model_task)


            with dsl.Condition(check_deployable_task.output == False, 'is-not-optimal-than-before'):
                print_op('trained model is not optimal than previously deployed one.')


        with dsl.Condition(init_workflow_task.outputs['is_new_data_exist'] == False, name='is-not-new-data-exists'):
            print_op('new data is not exist. aborted.')

    compiler.Compiler().compile(pipeline_func=pipeline, package_path=compiled_component_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docker-hub-username', type=str, default='', required=True)
    parser.add_argument('--storage-pvc-name', type=str, default='local-pvc')
    parser.add_argument('--docker-image-prefix', type=str, default='fs2')
    parser.add_argument('--storage-mount-path', type=str, default='/opt/storage')
    parser.add_argument('--compiled-component-output-path', type=str, default='./output')
    parser.add_argument('--compiled-component-output-filename', 
                        type=str,
                        default=f'pipeline-v2-{datetime.datetime.now().strftime("%y%m%d-%H%M")}.yaml')

    args = parser.parse_args()

    compiled_component_output_path = Path(args.compiled_component_output_path)
    if not compiled_component_output_path.exists():
        os.makedirs(compiled_component_output_path)

    build_fine_tuning_pipeline(docker_hub_username=args.docker_hub_username,
                               docker_image_prefix=args.docker_image_prefix,
                               storage_pvc_name=args.storage_pvc_name,
                               storage_mount_path=args.storage_mount_path,
                               compiled_component_output_path=str(compiled_component_output_path / args.compiled_component_output_filename))
