import argparse
import os
import re
import h5py
import netCDF4
import numpy as np
from tqdm import tqdm
from mpi4py import MPI


def get_variables(data):
    list_of_variables = []
    for v in data.variables.keys():
        var_shape = data.variables[v][:].shape
        if var_shape[1:] == (60, 100, 100) and var_shape[0] in range(28, 32):
            list_of_variables.append(v)
    return list_of_variables


def extract_num(line):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    v = re.findall(pattern, line)
    return v[0]


def get_params(forward, path):
    os.chdir(f"{path}")
    namelist = open("namelist.ocean")
    namelist_lines = namelist.readlines()
    gm = float(extract_num(namelist_lines[81]))
    redi = float(extract_num(namelist_lines[57]))
    cvmix_b_diff = float(extract_num(namelist_lines[109]))
    imp_bot_drag = float(extract_num(namelist_lines[275]))
    namelist.close()
    return [gm, redi, cvmix_b_diff, imp_bot_drag]


def populate_empty_array(list_of_variables, data, params, time_res=1):
    all_the_data = []
    FULL_LEN = 30
    TIME_STEPS = FULL_LEN // time_res
    time_idx = np.arange(0, FULL_LEN, time_res)
    for v in list_of_variables:
        init = np.array(data.variables[v][time_idx])
        reshaped_init = np.reshape(init, (TIME_STEPS, 60, 100, 100, 1))
        all_the_data.append(reshaped_init)
    for p in params:
        all_the_data.append(np.full((TIME_STEPS, 60, 100, 100, 1), p))
    return np.concatenate(all_the_data, axis=4)


def find_max_depth_index(x, y, data):
    thickness = list(data.variables["refBottomDepth"][:])
    bottom = list(data.variables["bottomDepth"][:])[x][y]
    for t in range(len(thickness)):
        if bottom < thickness[t]:
            return t - 1
    return len(thickness) - 1


def process_single_month(forward, args):
    results_year = []
    for month in range(1, 13):
        dir_path = os.path.join(args.path, f"output_{forward}")
        os.chdir(dir_path)
        os.system(f"cp {args.PROJECT_DIR}/gm/output_0/scrip.nc scrip.nc")
        os.system(f"cp {args.PROJECT_DIR}/gm/output_0/grd.nc grd.nc")

        os.system(
            f"ncremap -P mpas -s scrip.nc -g grd.nc output.0003-{month:02}-01_00.00.00.nc output-0003-{month:02}-01-rgr.nc"
        )
        data = netCDF4.Dataset(f"{dir_path}/output-0003-{month:02}-01-rgr.nc")
        list_of_variables = get_variables(data)

        params = get_params(forward, f"{args.path}/output_{forward}")[
            args.var_id
        ]
        if not isinstance(params, list):
            params = [params]

        result = populate_empty_array(
            list_of_variables, data, params, args.time_res
        )
        for x in range(100):
            for y in range(100):
                t = find_max_depth_index(x, y, data)
                if x == 50 and y == 50:
                    print("max depth", t)
                if t != 59:
                    for t_ in range(t + 1, 60):
                        result[:, t_, x, y] = np.zeros(16 + len(params))
        results_year.append(result)
        os.system(f"rm output-0003-{month:02}-01-rgr.nc")

    full_result = np.concatenate(results_year, axis=0)
    return full_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str)
    parser.add_argument("-var_id", type=int)
    parser.add_argument("-time_res", type=int, default=15)
    parser.add_argument(
        "-PAR_NAME", type=str, default="GM-biweekly-all-forwards"
    )
    parser.add_argument(
        "-PROJECT_DIR",
        type=str,
        default="/global/cfs/projectdirs/m4259/ecucuzzella/soma_ppe_data/",
    )
    parser.add_argument(
        "-SAVE_PATH",
        type=str,
        default="/global/cfs/projectdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/",
    )
    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the workload among processes
    forwards = list(range(100))
    local_forwards = forwards[rank::size]

    results = []
    for forward in local_forwards:
        result = process_single_month(forward, args)
        results.append((forward, result))

    # Gather results at the root process
    all_results = comm.gather(results, root=0)

    if rank == 0:
        with h5py.File(
            f"{args.SAVE_PATH}/data-{args.PAR_NAME}.hdf5", "w"
        ) as f:
            for process_results in all_results:
                for forward, result in process_results:
                    f.create_dataset(
                        f"forward_{forward}", data=result, dtype="f"
                    )


if __name__ == "__main__":
    main()
