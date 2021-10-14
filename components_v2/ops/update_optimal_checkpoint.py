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


    def _parse_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--data-base-path', type=str, default='', required=True)
        parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
        parser.add_argument('--metadata-path', type=str, default='./metadata')
        # parser.add_argument('--target-data-path', type=str, default='', required=True)
        parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')
        parser.add_argument('--deployed-checkpoint-stat-path', type=str, default='./deployed-checkpoint-status.json')

        return parser.parse_args()


    def _write_deploy_stat(metadata_path, base_path, stat):
        payload = {
            'base_path': base_path,
            'deployed_checkpoint': stat
        }

        with open(f'{metadata_path}', 'w') as f:
            json.dump(payload, f, indent=2)


    args = _parse_args()

    optimal_checkpoint_stat_path = os.path.join(target_data_path,
                                                args.optimal_checkpoint_stat_path)

    with open(optimal_checkpoint_stat_path, 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    data_base_path = Path(data_base_path) / args.fs2_data_path
    deployed_checkpoint_stat_path = data_base_path / args.metadata_path / args.deployed_checkpoint_stat_path

    _write_deploy_stat(deployed_checkpoint_stat_path, base_path=target_data_path, stat=optimal_checkpoint_stat)
