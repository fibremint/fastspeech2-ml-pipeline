image_prefix = 'fs2'

# op_config = {
#     'check_performance': None,
#     'deploy': {'base_image': 'python:3.8-alpine',
#                'packages_to_install': ['kubernetes==18.20.0']},
#     'evaluate': {'base_image': f'fibremint/{image_prefix}-evaluate:latest',
#                  'packages_to_install': ['fastspeech2', 'pytorch_sound'] },
#     'export_model': {'base_image': f'fibremint/{image_prefix}-torchserve:latest'},
#     'mfa_aligner': {'base_image': f'fibremint/{image_prefix}-mfa-aligner:latest'},
#     'prepare_align': {'base_image': f'fibremint/{image_prefix}-prepare-align:latest'},
#     'prepare_preprocess': None,
#     'preprocess': {'base_image': f'fibremint/{image_prefix}-preprocess:latest'},
#     'train': {'base_image': f'fibremint/{image_prefix}-train:latest' },
#     'update_optimal_checkpoint': None
# }

op_config = {
    'check_deployable': None,
    'deploy': {'base_image': 'python:3.8-alpine',
               'packages_to_install': ['kubernetes==18.20.0']},
    'evaluate': {'base_image': '{}/{}-evaluate:latest',
                 'packages_to_install': ['fastspeech2', 'pytorch_sound'] },
    'export_model': {'base_image': '{}/{}-torchserve:latest'},
    'mfa_align': {'base_image': '{}/{}-mfa-align:latest'},
    'prepare_align': {'base_image': '{}/{}-prepare-align:latest'},
    'prepare_preprocess': None,
    'preprocess': {'base_image': '{}/{}-runtime-base:latest'},
    'train': {'base_image': '{}/{}-train:latest' },
    'update_optimal_checkpoint': None
}
