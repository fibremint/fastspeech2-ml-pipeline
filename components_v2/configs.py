image_prefix = 'fs2'

op_config = {
    'check_deployable': None,
    'deploy': {'base_image': 'python:3.8-alpine',
               'packages_to_install': ['kubernetes==18.20.0']},
    'evaluate': {'base_image': '{}/{}-runtime:latest',
                 'packages_to_install': ['fastspeech2', 'pytorch_sound'] },
    'export_model': {'base_image': '{}/{}-export-model:latest'},
    'init_workflow': {'base_image': '{}/{}-init-workflow:latest'},
    'mfa_align': {'base_image': '{}/{}-mfa-align:latest'},
    'prepare_align': {'base_image': '{}/{}-runtime:latest'},
    'preprocess': {'base_image': '{}/{}-runtime:latest'},
    'train': {'base_image': '{}/{}-runtime:latest' },
    'update_optimal_checkpoint': None
}
