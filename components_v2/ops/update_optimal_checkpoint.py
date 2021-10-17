def update_optimal_checkpoint(data_base_path: str, target_data_path: str) -> None:
    import json
    import os
    from pathlib import Path

    from fs2_env import get_paths
    

    paths = get_paths(base_path=data_base_path, current_data_path=target_data_path)

    configs = {
        'fs2_data_path': './fs2-data',
        'metadata_path': './metadata',
        'optimal_checkpoint_stat_path': './optimal-checkpoint-status.json',
        "global_optimal_checkpoint_stat_path": "./global-optimal-checkpoint-status.json"
    }

    def _write_deploy_stat(metadata_path, stat, base_path=None):
        payload = {
            'base_path': base_path,
            'deployed_checkpoint': stat
        }

        with open(f'{metadata_path}', 'w') as f:
            json.dump(payload, f, indent=2)

    with open(paths['optimal_checkpoint_status'], 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    if not Path(paths['metadata']).exists():
        os.makedirs(paths['metadata'])

    _write_deploy_stat(paths['global_optimal_checkpoint_status'], base_path=target_data_path, stat=optimal_checkpoint_stat)


if __name__ == '__main__':
    update_optimal_checkpoint(data_base_path='/local-storage', target_data_path='/local-storage/fs2-data/data/20211017-012441')