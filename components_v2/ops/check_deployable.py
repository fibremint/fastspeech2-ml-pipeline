# import argparse
# import json
# import os
# from pathlib import Path
# from kfp import components as comp
from kfp import components as comp


# def _parse_args():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--data-base-path', type=str, default='', required=True)
#     parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
#     parser.add_argument('--metadata-path', type=str, default='./metadata')
#     # parser.add_argument('--target-data-path', type=str, default='', required=True)
#     parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')
#     parser.add_argument('--deployed-checkpoint-stat-path', type=str, default='./deployed-checkpoint-status.json')

#     return parser.parse_args()


def check_deployable(data_base_path: str, target_data_path: str) -> bool:
    import argparse
    import json
    import os
    from pathlib import Path
    from kfp import components as comp

    
    def _parse_args():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--data-base-path', type=str, default='', required=True)
        parser.add_argument('--fs2-data-path', type=str, default='./fs2-data')
        parser.add_argument('--metadata-path', type=str, default='./metadata')
        # parser.add_argument('--target-data-path', type=str, default='', required=True)
        parser.add_argument('--optimal-checkpoint-stat-path', type=str, default='./optimal-checkpoint-status.json')
        parser.add_argument('--deployed-checkpoint-stat-path', type=str, default='./deployed-checkpoint-status.json')

        return parser.parse_args()


    opt = _parse_args()

    optimal_checkpoint_stat_path = os.path.join(target_data_path,
                                                opt.optimal_checkpoint_stat_path)

    with open(optimal_checkpoint_stat_path, 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    data_base_path = Path(data_base_path) / opt.fs2_data_path
    metadata_path = data_base_path / opt.metadata_path
    if not metadata_path.exists():
        os.makedirs(metadata_path)

    is_optimal_than_global = False
    deployed_checkpoint_stat_path = metadata_path / opt.deployed_checkpoint_stat_path
    if not deployed_checkpoint_stat_path.exists():
        print('INFO: previously deployed checkpoint status is not exist. set current checkpoint as deployed.')
        is_optimal_than_global = True

    else:
        with open(deployed_checkpoint_stat_path, 'r') as f:
            deployed_checkpoint_stat = json.load(f)

        if optimal_checkpoint_stat['loss'] < deployed_checkpoint_stat['deployed_checkpoint']['loss']:
            print('INFO: current checkpoint is optimal than previously deployed one. set current checkpoint as deployed.')
            is_optimal_than_global = True
        else:
            print('INFO: current checkpoint is not optimal than previously deployed one. unchanged deployed checkpoint.')

    return is_optimal_than_global
                    
    # with open('/tmp/is-optimal-than-deployed.txt', 'w') as f:
    #     f.write(str(is_optimal_than_global))
