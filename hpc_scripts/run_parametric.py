import sys
from hpc_scripts.param_generator import get_all_params
import time

if __name__ == "__main__":
    exp_num, gpu_id = sys.argv[1:]
    print(exp_num, gpu_id)
    all_params = get_all_params()
    time.sleep(10)
    params = all_params[exp_num]
    params['gpu_id'] = gpu_id % 4
