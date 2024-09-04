import argparse
import multiprocessing as mp
import os
import re

import h5py
import netCDF4
import numpy as np
from tqdm import tqdm

# logging.basicConfig(
#     filename="dataProcessing.log", encoding="utf-8", level=logging.DEBUG
# )


# NAME: get_variables
# PURPOSE: to get the variables that are spatially and temporally varying
# We want only those that are spatially and temporally varying because we
# want the variables that change throughout the whole grid and also
# change over the course of a year.
# PARAMETERS: data which is the netCDF4 file
# RETURNS: the list of variables of size (30, 60, 100, 100)
def get_variables(data):
    list_of_variables = []
    for v in data.variables.keys():
        var_shape = data.variables[v][:].shape
        if var_shape[1:] == (60, 100, 100) and var_shape[0] in range(28, 32):
            list_of_variables.append(v)
    return list_of_variables


# NAME: get_GM
# PURPOSE: get the GM value for a particular forward because we need to
# include it in our dataset as it is the input for training.
# PARAMETERS: forward which is the number of the forward we care about
# RETURNS: the gm value for that particular forward
# In latin hyper cude we sample for 4 parameters: GM, Redi, cvmix, implicit_bottom_drag
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
    # os.chdir("..")
    return [gm, redi, cvmix_b_diff, imp_bot_drag]


# NAME: populate_empty_array
# PURPOSE: to populate the array with the values for all of the different
# variables that we care about, i.e. the ones that we found from
# calling get_variables
# PARAMETERS: list_of_variables which is the result of calling
# get_variables, data which is the netCDF4 file for a forward, and gm
# which is the result of calling get_GM
# RETURNS: a numpy array that is of size (30, 60, 100, 100, 17) that
# contains the data for each of the different variables that we care about
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


# NAME: find_max_depth_index
# PURPOSE: to figure out the index where the maximum depth of a particular
# spatial point is exceeded. This is necessary because past a certain
# depth, some cells return NaNs.
# PARAMETERS: x and y which is the location of the point in question, data
# which is the netCDF4 data for a forward
# RETURNS: the index of the last vertical layer where data should exist
# based on the layer thicknesses and maximum depths
def find_max_depth_index(x, y, data):
    thickness = list(data.variables["refBottomDepth"][:])
    bottom = list(data.variables["bottomDepth"][:])[x][y]
    for t in range(len(thickness)):
        if bottom < thickness[t]:
            return t - 1
    return len(thickness) - 1


# go through all the simulations
def process_sinlge_month(forward, args):
    results_year = []
    for month in range(1, 13):
        # dir_path = os.path.join(
        #     args.DIR, f"output_{forward}/analysis_members/"
        # )
        dir_path = os.path.join(args.path, f"output_{forward}")
        os.chdir(dir_path)
        os.system(f"cp {args.PROJECT_DIR}/gm/output_0/scrip.nc scrip.nc")
        os.system(f"cp {args.PROJECT_DIR}/gm/output_0/grd.nc grd.nc")

        # need to regrid each of the .nc files for a forward based on the
        # scrip.nc and grd.nc files
        # try:
        # os.system(
        #     f"ncremap -P mpas -s scrip.nc -g grd.nc mpaso.hist.am.timeSeriesStatsDaily.0003-{month:02}-01.nc dayAvgOutput-0003-{month:02}-01-rgr.nc"
        # )
        os.system(
            f"ncremap -P mpas -s scrip.nc -g grd.nc output.0003-{month:02}-01_00.00.00.nc output-0003-{month:02}-01-rgr.nc"
        )
        # currently just getting data for the first month for Yixuan's training
        data = netCDF4.Dataset(f"{dir_path}/output-0003-{month:02}-01-rgr.nc")
        list_of_variables = get_variables(data)

        params = get_params(forward, f"{args.path}/output_{forward}")[
            args.var_id
        ]  # specifiy which parameter to include in the input
        if not isinstance(params, list):
            params = [params]
        #    os.chdir(f"{PROJECT_DIR}/ysun/")
        # grp = f.create_group("forward_" + str(forward))

        result = populate_empty_array(
            list_of_variables, data, params, args.time_res
        )
        for x in range(100):
            for y in range(100):
                t = find_max_depth_index(x, y, data)
                if x == 50 and y == 50:
                    print("max depth", t)
                # if the index is 59, that means the vertical layers go
                # all the way to the bottom of the vertical grid, so there
                # is no need to set any values to 0
                if t != 59:
                    for t_ in range(t + 1, 60):
                        result[:, t_, x, y] = np.zeros(16 + len(params))
            # create a dataset for each of the forwards
        results_year.append(result)
        os.system(f"rm output-0003-{month:02}-01-rgr.nc")
    return np.concatenate(results_year, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str)
    parser.add_argument("-var_id", type=int)
    parser.add_argument(
        "-time_res", type=int, default=15
    )  # time resolution by days
    parser.add_argument(
        "-PAR_NAME", type=str, default="test"
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

    nprocs = mp.cpu_count()
    pool = mp.Pool(processes=nprocs)
    print(f"Number of processors: {nprocs}")

    # open a hdf5 file to write the results to
    f = h5py.File(f"{args.SAVE_PATH}/data-{args.PAR_NAME}.hdf5", "w")
    DIR = args.path
    print(f"Data from {args.path} are being processed!!!!")
    PROJECT_DIR = "/global/cfs/projectdirs/m4259/ecucuzzella/soma_ppe_data/"

    for forward in tqdm(range(100)):
        results = []
        result = process_sinlge_month(forward, args)
        dset = f.create_dataset(
            f"forward_{forward}", result.shape, dtype="f", data=result
        )
