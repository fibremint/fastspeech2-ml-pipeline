image_prefix = 'fs2'

op_config = {
    'check_deployable': {'base_image': 'python:3.8-alpine',
                         'packages_to_install': ['fs2-env==0.1.1']},
    'deploy': {'base_image': 'python:3.8-alpine',
               'packages_to_install': ['fs2-env==0.1.1', 'kubernetes==18.20.0']},
    'evaluate': {'base_image': '{}/{}-runtime:latest',
                 'packages_to_install': ['fs2-env==0.1.1']},
    'export_model': {'base_image': '{}/{}-export-model:latest',
                     'packages_to_install': ['fs2-env==0.1.1']},
    'init_workflow': {'base_image': '{}/{}-init-workflow:latest',
                      'packages_to_install': ['fs2-env==0.1.1']},
    'mfa_align': {'base_image': '{}/{}-mfa-align:latest',
                  'packages_to_install': ['fs2-env==0.1.1']},
    'prepare_align': {'base_image': '{}/{}-runtime:latest',
                      'packages_to_install': ['fs2-env==0.1.1']},
    'preprocess': {'base_image': '{}/{}-runtime:latest', 
                   'packages_to_install': ['fs2-env==0.1.1']},
    'train': {'base_image': '{}/{}-runtime:latest',
              'packages_to_install': ['fs2-env==0.1.1']},
    'update_optimal_checkpoint': {'base_image': 'python:3.8-alpine',
                                  'packages_to_install': ['fs2-env==0.1.1']},
}
