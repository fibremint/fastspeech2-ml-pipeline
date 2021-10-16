def check_deployable(data_base_path: str, target_data_path: str) -> bool:
    import json
    import os
    from pathlib import Path

    fs2_data_path = './fs2-data'
    metadata_path = './metadata'
    optimal_checkpoint_stat_path = './optimal-checkpoint-status.json'
    global_optimal_checkpoint_stat_path = "./global-optimal-checkpoint-status.json"

    optimal_checkpoint_stat_path = os.path.join(target_data_path,
                                                optimal_checkpoint_stat_path)

    with open(optimal_checkpoint_stat_path, 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    data_base_path = Path(data_base_path) / fs2_data_path
    metadata_path = data_base_path / metadata_path
    if not metadata_path.exists():
        os.makedirs(metadata_path)

    is_optimal_than_global = False
    deployed_checkpoint_stat_path = metadata_path / global_optimal_checkpoint_stat_path
    if not deployed_checkpoint_stat_path.exists():
        print('INFO: previously deployed checkpoint status is not exist. set current checkpoint as deployed.')
        is_optimal_than_global = True

    else:
        with open(deployed_checkpoint_stat_path, 'r') as f:
            deployed_checkpoint_stat = json.load(f)

        if optimal_checkpoint_stat['loss'] < deployed_checkpoint_stat['deployed_checkpoint']['loss']:
        # if optimal_checkpoint_stat['loss'] < deployed_checkpoint_stat['loss']:
            print('INFO: current checkpoint is optimal than previously deployed one. this will be set as global optimal.')
            is_optimal_than_global = True
        else:
            print('INFO: current checkpoint is not optimal than previously deployed one. unchanged deployed checkpoint.')

    return is_optimal_than_global
                    

if __name__ == '__main__':
    check_deployable(data_base_path='/local-storage', target_data_path='/local-storage/fs2-data/data.2/20211010-105342')