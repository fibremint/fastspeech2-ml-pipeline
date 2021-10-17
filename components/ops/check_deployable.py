def check_deployable(data_base_path: str, target_data_path: str) -> bool:
    import json
    import os
    from pathlib import Path

    from fs2_env import get_paths

    paths = get_paths(base_path=data_base_path, current_data_path=target_data_path)

    with open(paths['optimal_checkpoint_status'], 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    os.makedirs(paths['metadata'], exist_ok=True)

    is_optimal_than_global = False
    if not Path(paths['global_optimal_checkpoint_status']).exists():
        print('INFO: previously deployed checkpoint status is not exist. set current checkpoint as deployed.')
        is_optimal_than_global = True

    else:
        with open(paths['global_optimal_checkpoint_status'], 'r') as f:
            deployed_checkpoint_stat = json.load(f)

        if optimal_checkpoint_stat['loss'] < deployed_checkpoint_stat['deployed_checkpoint']['loss']:
            print('INFO: current checkpoint is optimal than previously deployed one. this will be set as global optimal.')
            is_optimal_than_global = True
        else:
            print('INFO: current checkpoint is not optimal than previously deployed one. unchanged deployed checkpoint.')

    return is_optimal_than_global
                    

if __name__ == '__main__':
    check_deployable(data_base_path='/local-storage', target_data_path='/local-storage/fs2-data/data/20211016-192849')