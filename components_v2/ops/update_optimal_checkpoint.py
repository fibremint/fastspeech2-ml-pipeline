# import argparse
# import json
# import os
# from pathlib import Path


# def _parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-base-path', type=str, default='', required=True)
#     parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
#     parser.add_argument('--metadata-path', type=str, default='./metadata')
#     parser.add_argument('--target-data-path', type=str, default='', required=True)
#     parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')
#     parser.add_argument('--deployed-checkpoint-stat-path', type=str, default='./deployed-checkpoint-status.json')

#     return parser.parse_args()


# def _write_deploy_stat(metadata_path, base_path, stat):
#     payload = {
#         'base_path': base_path,
#         'deployed_checkpoint': stat
#     }

#     with open(f'{metadata_path}', 'w') as f:
#         json.dump(payload, f, indent=2)


def update_optimal_checkpoint(data_base_path: str, target_data_path: str) -> None:
    import argparse
    import json
    import os
    from pathlib import Path

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

        # with open(f'{metadata_path}', 'w') as f:
        #     json.dump(stat, f, indent=2)

        with open(f'{metadata_path}', 'w') as f:
            json.dump(payload, f, indent=2)


    # args = _parse_args()
    optimal_checkpoint_stat_path = Path(target_data_path) / configs["optimal_checkpoint_stat_path"]
    with open(f'{optimal_checkpoint_stat_path}', 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    metadata_path = Path(data_base_path) / configs["fs2_data_path"] / configs["metadata_path"]
    if not metadata_path.exists():
        os.makedirs(metadata_path)

    deployed_checkpoint_stat_path = metadata_path / configs["global_optimal_checkpoint_stat_path"]

    _write_deploy_stat(deployed_checkpoint_stat_path, base_path=target_data_path, stat=optimal_checkpoint_stat)


if __name__ == '__main__':
    update_optimal_checkpoint(data_base_path='/local-storage', target_data_path='/local-storage/fs2-data/data.2/20211010-105342')