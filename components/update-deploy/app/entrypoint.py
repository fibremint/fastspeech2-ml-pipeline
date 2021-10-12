from opts import opt
import os
import json
from pathlib import Path


def main(opt):
    def _write_deploy_stat(metadata_path, base_path, stat):
        payload = {
            'base_path': base_path,
            'deployed_checkpoint': stat
        }

        with open(f'{metadata_path}', 'w') as f:
            json.dump(payload, f, indent=2)
            

    optimal_checkpoint_stat_path = os.path.join(opt.target_data_path,
                                                opt.optimal_checkpoint_stat_path)

    with open(optimal_checkpoint_stat_path, 'r') as f:
        optimal_checkpoint_stat = json.load(f)

    data_base_path = Path(opt.data_base_path) / opt.fs2_data_path
    deployed_checkpoint_stat_path = data_base_path / opt.metadata_path / opt.deployed_checkpoint_stat_path

    _write_deploy_stat(deployed_checkpoint_stat_path, base_path=opt.target_data_path, stat=optimal_checkpoint_stat)

    # is_optimal_than_deployed = False
    # deployed_checkpoint_stat_path = metadata_path / opt.deployed_checkpoint_stat_path
    # if not deployed_checkpoint_stat_path.exists():
    #     print('INFO: previously deployed checkpoint status is not exist. set current checkpoint as deployed.')
    #     _write_deploy_stat(deployed_checkpoint_stat_path,
    #                        base_path=opt.target_data_path, 
    #                        stat=optimal_checkpoint_stat)

    #     is_optimal_than_deployed = True

    # else:
    #     with open(deployed_checkpoint_stat_path, 'r') as f:
    #         deployed_checkpoint_stat = json.load(f)

    #     if optimal_checkpoint_stat['loss'] < deployed_checkpoint_stat['deployed_checkpoint']['loss']:
    #         print('INFO: current checkpoint is optimal than previously deployed one. set current checkpoint as deployed.')
    #         _write_deploy_stat(deployed_checkpoint_stat_path,
    #                 base_path=opt.target_data_path, 
    #                 stat=optimal_checkpoint_stat)

    #         is_optimal_than_deployed = True
    #     else:
    #         print('INFO: current checkpoint is not optimal than previously deployed one. unchanged deployed checkpoint.')
            
        
    # with open('/tmp/is-optimal-than-deployed.txt', 'w') as f:
    #     f.write(str(is_optimal_than_deployed))



if __name__ == '__main__':
    main(opt)