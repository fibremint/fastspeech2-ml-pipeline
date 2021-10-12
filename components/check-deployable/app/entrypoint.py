import json
import os
from pathlib import Path

from opts import opt


def main(opt):
    optimal_checkpoint_stat_path = os.path.join(opt.target_data_path,
                                                opt.optimal_checkpoint_stat_path)

    with open(optimal_checkpoint_stat_path, 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    data_base_path = Path(opt.data_base_path) / opt.fs2_data_path
    metadata_path = data_base_path / opt.metadata_path
    if not metadata_path.exists():
        os.makedirs(metadata_path)

    is_optimal_than_deployed = False
    deployed_checkpoint_stat_path = metadata_path / opt.deployed_checkpoint_stat_path
    if not deployed_checkpoint_stat_path.exists():
        print('INFO: previously deployed checkpoint status is not exist. set current checkpoint as deployed.')
        is_optimal_than_deployed = True

    else:
        with open(deployed_checkpoint_stat_path, 'r') as f:
            deployed_checkpoint_stat = json.load(f)

        if optimal_checkpoint_stat['loss'] < deployed_checkpoint_stat['deployed_checkpoint']['loss']:
            print('INFO: current checkpoint is optimal than previously deployed one. set current checkpoint as deployed.')
            is_optimal_than_deployed = True
        else:
            print('INFO: current checkpoint is not optimal than previously deployed one. unchanged deployed checkpoint.')
            
        
    with open('/tmp/is-optimal-than-deployed.txt', 'w') as f:
        f.write(str(is_optimal_than_deployed))


if __name__ == '__main__':
    main(opt)
