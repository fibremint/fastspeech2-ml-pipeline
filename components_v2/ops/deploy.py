# ref: https://byeongjo-kim.tistory.com/14


def deploy(model_version: str, data_base_path: str, pvc_name: str) -> None:
    import argparse
    import datetime
    import os
    import subprocess
    from pathlib import Path
    import glob
    import shutil

    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    from fs2_env import get_paths

    paths = get_paths(base_path=data_base_path)


    def archive_previous_models(**configs):
        if not Path(paths['archived_models']).exists():
            os.makedirs(paths['archived_models'])

        mar_paths = glob.glob(os.path.join(paths['deployed_models'], configs["model_name"]) + '*.mar')
        mar_paths.sort()

        if len(mar_paths) < 2:
            print('INFO: a number of model is less than 2. archive will be skipped.')
        else:
            print('INFO: archive previous images.')
            archive_mar_paths = mar_paths[:-1 * configs['max_model_num']]

            for curr_mar_path in archive_mar_paths:
                shutil.move(curr_mar_path, paths['archived_models'])


    def serving(model_version, **configs):
        # config.load_kube_config() # for test
        config.load_incluster_config() # for execute in K8s pod

        k8s_app_v1 = client.AppsV1Api()
        torchserve_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    'namespace': 'torchserve',
                    'app': 'torchserve',
                    'app.kubernetes.io/version': model_version
                }
            ),
            spec=client.V1PodSpec(
                volumes=[
                    client.V1Volume(
                        name='local-persistent-storage',
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=pvc_name
                        )
                    )
                ],
                containers=[
                    client.V1Container(
                        name=configs['container_name'],
                        image=configs['container_image'],
                        args=['CUDA_VISIBLE_DEVICES="" torchserve', 
                              '--start',
                              '--model-store',
                              paths['deployed_models'],                    
                              '--ts-config', 
                              paths['torchserve_config']],
                        image_pull_policy='Always',
                        ports=[
                            client.V1ContainerPort(
                                name='ts',
                                container_port=configs['prediction_port']
                            ),
                            client.V1ContainerPort(
                                name='ts-management',
                                container_port=configs['management_port']
                            ),
                            client.V1ContainerPort(
                                name='ts-metrics',
                                container_port=configs['metric_port']
                            )
                        ],
                        volume_mounts=[
                            client.V1VolumeMount(
                                name='local-persistent-storage',
                                mount_path=data_base_path
                            )
                        ],
                        resources=client.V1ResourceRequirements(
                            limits={
                                'cpu': '1000m',
                                'memory': '2Gi',
                                'nvidia.com/gpu': 0
                            }
                        )
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
                    'app.kubernetes.io/version': model_version
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
                        port=configs['prediction_port'],
                        target_port="ts"
                    ),
                    client.V1ServicePort(
                        name="mdl",
                        port=configs['management_port'],
                        target_port="ts-management"
                    ),
                    client.V1ServicePort(
                        name="metrics",
                        port=configs['metric_port'],
                        target_port="ts-metrics"
                    )
                ]
            )
        )

        print('INFO: Refersh Torch Serve model deployer')
        print('INFO: Delete previous deployment and service')
        is_success = True
        try:
            k8s_app_v1.delete_namespaced_deployment(name='torchserve', namespace='torchserve')
            print('INFO: Torch Serve deployment is deleted.')
        except ApiException as e:
            if e.reason == 'Not Found':
                print('INFO: Torch Serve deployment is not exist. a new one would be created')
            else:
                print('Api Error')
                print(e)
                is_success = False
        except Exception as e:
            print(e)
            is_success = False

        try:
            k8s_core_v1.delete_namespaced_service(name='torchserve', namespace='torchserve')
            print('INFO: Torch Serve service is deleted')
        except ApiException as e:
            if e.reason == 'Not Found':
                print('INFO: Torch Serve deployment is not exist a new one would be created')
            else:
                print('Api Error')
                print(e)
                is_success = False
        except Exception as e:
            print(e)
            is_success = False

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
                is_success = False
        except Exception as e:
            print(e)
            is_success = False

        try:
            k8s_core_v1.create_namespaced_service(body=torchserve_service, namespace='torchserve')
            print('INFO: Torch Serve service is created')
        except ApiException as e:
            if e.reason == 'Conflict':
                print('INFO: Torch Serve service is already created')
            else:
                print('ERR: Api error')
                print(e)
                is_success = False
        except Exception as e:
            print(e)
            is_success = False

        if not is_success:
            raise RuntimeError("Kubernetes interaction failed.")


    configs = {
        'fs2_data_path': './fs2-data',
        'torchserve_config_path': './configs',
        'torchserve_config_filename': 'torchserve-config.properties',
        'container_name': 'torchserve',
        'model_name': 'fastspeech2',
        'container_image': 'fibremint/fs2-torchserve-model-deploy:latest',
        'model_base_path': './models',
        'model_store_path': './model-store',
        'model_archived_path': './archived',
        # TODO: set value by argument
        'max_model_num': 1,
        # 'pvc_name': 'local-pvc',
        'prediction_port': 9000,
        'management_port': 9001,
        'metric_port': 9002
    }

    os.makedirs(paths['configs'], exist_ok=True)

    torchserve_config = \
        f"inference_address=http://0.0.0.0:{configs['prediction_port']}\n" \
        f"management_address=http://0.0.0.0:{configs['management_port']}\n" \
        f"metrics_address=http://0.0.0.0:{configs['metric_port']}\n" \
        f"job_queue_size=100\n" \
        f"install_py_dep_per_module=true\n" \
        f"load_models=all\n"

    with open(paths['torchserve_config'], 'w') as f:
        f.write(torchserve_config)
    
    archive_previous_models(**configs)
    serving(**configs, model_version=model_version)    


if __name__ == '__main__':
    deploy(model_version='test', data_base_path='/local-storage', pvc_name='local-pvc')
