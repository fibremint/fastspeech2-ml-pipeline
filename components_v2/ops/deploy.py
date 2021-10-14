# ref: https://byeongjo-kim.tistory.com/14

import argparse
import datetime
import os
import subprocess
from pathlib import Path
import glob
import shutil

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

VOLUME_MOUNT_PATH = '/opt/storage'


# def _parse_args():
#     parser = argparse.ArgumentParser()
#     # TODO: fix this
#     parser.add_argument('--model-version', type=str, default='', required=False)
#     parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
#     parser.add_argument('--torchserve-config-path', type=str, default='./configs')
#     parser.add_argument('--torchserve-config-filename', type=str, default='torchserve-config.properties')

#     parser.add_argument('--container-name', type=str, default='torchserve')
#     parser.add_argument('--model-name', type=str, default='fastspeech2')
#     # parser.add_argument('--container-image', type=str, default='pytorch/torchserve:0.4.2-cpu')
#     parser.add_argument('--container-image', type=str, default='fibremint/fs2-torchserve-model-deploy:latest')
#     parser.add_argument('--model-base-path', type=str, default='./models')
#     parser.add_argument('--model-store-path', type=str, default='./model-store')
#     parser.add_argument('--model-archived-path', type=str, default='./archived')
#     parser.add_argument('--max-model-num', type=int, default=1)

#     parser.add_argument('--pvc-name', type=str, default='local-pvc')
#     parser.add_argument('--prediction-port', type=int, default=9000)
#     parser.add_argument('--management-port', type=int, default=9001)
#     parser.add_argument('--metric-port', type=int, default=9002)

#     return parser.parse_args()


# def archive_previous_models(args):
#     # model_name_version = args.model_name + '-' + version
#     fs2_data_path = Path(VOLUME_MOUNT_PATH) / args.fs2_data_path
#     archive_path = fs2_data_path / args.model_base_path / args.model_archived_path

#     if not archive_path.exists():
#         os.makedirs(f'{archive_path}')

#     mar_paths = glob.glob(f'{fs2_data_path / args.model_base_path / args.model_store_path / args.model_name}' + '*.mar')
#     mar_paths.sort()

#     # TODO: don't archive when length of an archive mar paths is < 0
#     archive_mar_paths = mar_paths[:-1 * args.max_model_num]

#     for curr_mar_path in archive_mar_paths:
#         shutil.move(f'{curr_mar_path}', f'{archive_path}')


# def serving(args, version):
#     fs2_data_path = Path(VOLUME_MOUNT_PATH) / args.fs2_data_path

#     # config.load_kube_config() # for test
#     config.load_incluster_config() # for release

#     k8s_app_v1 = client.AppsV1Api()
#     torchserve_template = client.V1PodTemplateSpec(
#         metadata=client.V1ObjectMeta(
#             labels={
#                 'namespace': 'torchserve',
#                 'app': 'torchserve',
#                 'app.kubernetes.io/version': version
#             }
#         ),
#         spec=client.V1PodSpec(
#             volumes=[
#                 client.V1Volume(
#                     name='local-persistent-storage',
#                     persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
#                         claim_name=args.pvc_name
#                     )
#                 )
#             ],
#             containers=[
#                 client.V1Container(
#                     name=args.container_name,
#                     image=args.container_image,
#                     args=['torchserve', 
#                           '--start',
#                           '--model-store',
#                           str(fs2_data_path / args.model_base_path / args.model_store_path), 
#                           '--ts-config', 
#                           str(fs2_data_path / args.torchserve_config_path / args.torchserve_config_filename)],
#                     image_pull_policy='Always',
#                     ports=[
#                         client.V1ContainerPort(
#                             name='ts',
#                             container_port=args.prediction_port
#                         ),
#                         client.V1ContainerPort(
#                             name='ts-management',
#                             container_port=args.management_port
#                         ),
#                         client.V1ContainerPort(
#                             name='ts-metrics',
#                             container_port=args.metric_port
#                         )
#                     ],
#                     volume_mounts=[
#                         client.V1VolumeMount(
#                             name='local-persistent-storage',
#                             mount_path=VOLUME_MOUNT_PATH
#                         )
#                     ],
#                     # resources=client.V1ResourceRequirements(
#                     #     limits={
#                     #         'cpu': '1000m',
#                     #         'memory': '2Gi',
#                     #         'nvidia.com/gpu': 0
#                     #     }
#                     # )
#                 )
#             ]
#         )
#     )

#     torchserve_deployment = client.V1Deployment(
#         api_version='apps/v1',
#         kind='Deployment',
#         metadata=client.V1ObjectMeta(
#             name='torchserve',
#             labels={
#                 'app': 'torchserve',
#                 'app.kubernetes.io/version': version
#             }
#         ),
#         spec=client.V1DeploymentSpec(
#             # replicas=2,
#             replicas=1,
#             selector=client.V1LabelSelector(
#                 match_labels={'app': 'torchserve'}
#             ),
#             strategy=client.V1DeploymentStrategy(
#                 type='RollingUpdate',
#                 rolling_update=client.V1RollingUpdateDeployment(
#                     max_surge=1,
#                     # max_unavailable=1,
#                     max_unavailable=0
#                 )
#             ),
#             template=torchserve_template
#         )
#     )

#     k8s_core_v1 = client.CoreV1Api()
#     torchserve_service = client.V1Service(
#         api_version="v1",
#         kind="Service",
#         metadata=client.V1ObjectMeta(
#             name="torchserve",
#             labels={
#                 "app": "torchserve"
#             }
#         ),
#         spec=client.V1ServiceSpec(
#             type="LoadBalancer",
#             selector={"app": "torchserve"},
#             ports=[
#                 client.V1ServicePort(
#                     name="preds",
#                     port=args.prediction_port,
#                     target_port="ts"
#                 ),
#                 client.V1ServicePort(
#                     name="mdl",
#                     port=args.management_port,
#                     target_port="ts-management"
#                 ),
#                 client.V1ServicePort(
#                     name="metrics",
#                     port=args.metric_port,
#                     target_port="ts-metrics"
#                 )
#             ]
#         )
#     )

#     print('INFO: Refersh Torch Serve model deployer')
#     print('INFO: Delete previous deployment and service')

#     try:
#         k8s_app_v1.delete_namespaced_deployment(name='torchserve', namespace='torchserve')
#         print('INFO: Torch Serve deployment is deleted.')
#     except ApiException as e:
#         if e.reason == 'Not Found':
#             print('INFO: Torch Serve deployment is not exist. a new one would be created')
#         else:
#             print('Api Error')
#             print(e)
#     except Exception as e:
#         print(e)

#     try:
#         k8s_core_v1.delete_namespaced_service(name='torchserve', namespace='torchserve')
#         print('INFO: Torch Serve service is deleted')
#     except ApiException as e:
#         if e.reason == 'Not Found':
#             print('INFO: Torch Serve deployment is not exist a new one would be created')
#         else:
#             print('Api Error')
#             print(e)
#     except Exception as e:
#         print(e)

#     print('INFO: Create deployment and service')

#     try:
#         k8s_app_v1.create_namespaced_deployment(body=torchserve_deployment, namespace='torchserve')
#         print('INFO: Torch Serve deployment is created')
#     except ApiException as e:
#         if e.reason == 'Conflict':
#             k8s_app_v1.replace_namespaced_deployment(name='torchserve', namespace='torchserve', body=torchserve_deployment)
#             print('INFO: Torch Serve replaced')
#         else:
#             print('ERR: Api error')
#             print(e)
#     except Exception as e:
#         print(e)

#     try:
#         k8s_core_v1.create_namespaced_service(body=torchserve_service, namespace='torchserve')
#         print('INFO: Torch Serve service is created')
#     except ApiException as e:
#         if e.reason == 'Conflict':
#             print('INFO: Torch Serve service is already created')
#         else:
#             print('ERR: Api error')
#             print(e)
#     except Exception as e:
#         print(e)
    

def deploy(model_version: str, data_base_path: str) -> None:
    import argparse
    import datetime
    import os
    import subprocess
    from pathlib import Path
    import glob
    import shutil

    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    VOLUME_MOUNT_PATH = '/opt/storage'


    def _parse_args():
        parser = argparse.ArgumentParser()
        # TODO: fix this
        parser.add_argument('--model-version', type=str, default='', required=False)
        parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
        parser.add_argument('--torchserve-config-path', type=str, default='./configs')
        parser.add_argument('--torchserve-config-filename', type=str, default='torchserve-config.properties')

        parser.add_argument('--container-name', type=str, default='torchserve')
        parser.add_argument('--model-name', type=str, default='fastspeech2')
        # parser.add_argument('--container-image', type=str, default='pytorch/torchserve:0.4.2-cpu')
        parser.add_argument('--container-image', type=str, default='fibremint/fs2-torchserve-model-deploy:latest')
        parser.add_argument('--model-base-path', type=str, default='./models')
        parser.add_argument('--model-store-path', type=str, default='./model-store')
        parser.add_argument('--model-archived-path', type=str, default='./archived')
        parser.add_argument('--max-model-num', type=int, default=1)

        parser.add_argument('--pvc-name', type=str, default='local-pvc')
        parser.add_argument('--prediction-port', type=int, default=9000)
        parser.add_argument('--management-port', type=int, default=9001)
        parser.add_argument('--metric-port', type=int, default=9002)

        return parser.parse_args()


    def archive_previous_models(args):
        # model_name_version = args.model_name + '-' + version
        fs2_data_path = Path(VOLUME_MOUNT_PATH) / args.fs2_data_path
        archive_path = fs2_data_path / args.model_base_path / args.model_archived_path

        if not archive_path.exists():
            os.makedirs(f'{archive_path}')

        mar_paths = glob.glob(f'{fs2_data_path / args.model_base_path / args.model_store_path / args.model_name}' + '*.mar')
        mar_paths.sort()

        # TODO: don't archive when length of an archive mar paths is < 0
        archive_mar_paths = mar_paths[:-1 * args.max_model_num]

        for curr_mar_path in archive_mar_paths:
            shutil.move(f'{curr_mar_path}', f'{archive_path}')


    def serving(args, version):
        fs2_data_path = Path(VOLUME_MOUNT_PATH) / args.fs2_data_path

        # config.load_kube_config() # for test
        config.load_incluster_config() # for release

        k8s_app_v1 = client.AppsV1Api()
        torchserve_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    'namespace': 'torchserve',
                    'app': 'torchserve',
                    'app.kubernetes.io/version': version
                }
            ),
            spec=client.V1PodSpec(
                volumes=[
                    client.V1Volume(
                        name='local-persistent-storage',
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=args.pvc_name
                        )
                    )
                ],
                containers=[
                    client.V1Container(
                        name=args.container_name,
                        image=args.container_image,
                        args=['torchserve', 
                            '--start',
                            '--model-store',
                            str(fs2_data_path / args.model_base_path / args.model_store_path), 
                            '--ts-config', 
                            str(fs2_data_path / args.torchserve_config_path / args.torchserve_config_filename)],
                        image_pull_policy='Always',
                        ports=[
                            client.V1ContainerPort(
                                name='ts',
                                container_port=args.prediction_port
                            ),
                            client.V1ContainerPort(
                                name='ts-management',
                                container_port=args.management_port
                            ),
                            client.V1ContainerPort(
                                name='ts-metrics',
                                container_port=args.metric_port
                            )
                        ],
                        volume_mounts=[
                            client.V1VolumeMount(
                                name='local-persistent-storage',
                                mount_path=VOLUME_MOUNT_PATH
                            )
                        ],
                        # resources=client.V1ResourceRequirements(
                        #     limits={
                        #         'cpu': '1000m',
                        #         'memory': '2Gi',
                        #         'nvidia.com/gpu': 0
                        #     }
                        # )
                    )
                ]
            )
        )

        torchserve_deployment = client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(
                name='torchserve',
                labels={
                    'app': 'torchserve',
                    'app.kubernetes.io/version': version
                }
            ),
            spec=client.V1DeploymentSpec(
                # replicas=2,
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={'app': 'torchserve'}
                ),
                strategy=client.V1DeploymentStrategy(
                    type='RollingUpdate',
                    rolling_update=client.V1RollingUpdateDeployment(
                        max_surge=1,
                        # max_unavailable=1,
                        max_unavailable=0
                    )
                ),
                template=torchserve_template
            )
        )

        k8s_core_v1 = client.CoreV1Api()
        torchserve_service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name="torchserve",
                labels={
                    "app": "torchserve"
                }
            ),
            spec=client.V1ServiceSpec(
                type="LoadBalancer",
                selector={"app": "torchserve"},
                ports=[
                    client.V1ServicePort(
                        name="preds",
                        port=args.prediction_port,
                        target_port="ts"
                    ),
                    client.V1ServicePort(
                        name="mdl",
                        port=args.management_port,
                        target_port="ts-management"
                    ),
                    client.V1ServicePort(
                        name="metrics",
                        port=args.metric_port,
                        target_port="ts-metrics"
                    )
                ]
            )
        )

        print('INFO: Refersh Torch Serve model deployer')
        print('INFO: Delete previous deployment and service')

        try:
            k8s_app_v1.delete_namespaced_deployment(name='torchserve', namespace='torchserve')
            print('INFO: Torch Serve deployment is deleted.')
        except ApiException as e:
            if e.reason == 'Not Found':
                print('INFO: Torch Serve deployment is not exist. a new one would be created')
            else:
                print('Api Error')
                print(e)
        except Exception as e:
            print(e)

        try:
            k8s_core_v1.delete_namespaced_service(name='torchserve', namespace='torchserve')
            print('INFO: Torch Serve service is deleted')
        except ApiException as e:
            if e.reason == 'Not Found':
                print('INFO: Torch Serve deployment is not exist a new one would be created')
            else:
                print('Api Error')
                print(e)
        except Exception as e:
            print(e)

        print('INFO: Create deployment and service')

        try:
            k8s_app_v1.create_namespaced_deployment(body=torchserve_deployment, namespace='torchserve')
            print('INFO: Torch Serve deployment is created')
        except ApiException as e:
            if e.reason == 'Conflict':
                k8s_app_v1.replace_namespaced_deployment(name='torchserve', namespace='torchserve', body=torchserve_deployment)
                print('INFO: Torch Serve replaced')
            else:
                print('ERR: Api error')
                print(e)
        except Exception as e:
            print(e)

        try:
            k8s_core_v1.create_namespaced_service(body=torchserve_service, namespace='torchserve')
            print('INFO: Torch Serve service is created')
        except ApiException as e:
            if e.reason == 'Conflict':
                print('INFO: Torch Serve service is already created')
            else:
                print('ERR: Api error')
                print(e)
        except Exception as e:
            print(e)
    

    args = _parse_args()

    fs2_data_path = Path(VOLUME_MOUNT_PATH) / args.fs2_data_path
    torchserve_config_path = fs2_data_path / args.torchserve_config_path
    if not torchserve_config_path.exists():
        os.makedirs(torchserve_config_path)

    torchserve_config = \
        f"inference_address=http://0.0.0.0:{args.prediction_port}\n" \
        f"management_address=http://0.0.0.0:{args.management_port}\n" \
        f"metrics_address=http://0.0.0.0:{args.metric_port}\n" \
        f"job_queue_size=100\n" \
        f"install_py_dep_per_module=true\n" \
        f"load_models=all\n"

    with open(f'{torchserve_config_path / args.torchserve_config_filename}', 'w') as f:
        f.write(torchserve_config)
    
    archive_previous_models(args)
    serving(args, version=args.version)    


# def create_component():
#     from kfp import components

#     components.create_component_from_func(deploy, output_component_file='./components_v2/resources/deploy.yaml')


# if __name__ == '__main__':
#     create_component()
