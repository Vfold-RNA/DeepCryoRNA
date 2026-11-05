import sys
import os
import argparse
import time
from reconstruct_rna_structures import preprocess_cryoEM_map, reconstruct_structures_from_cryoEM_map, display_duration_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct RNA 3D structures from cryo-EM maps based on a deep learning model.")
    parser.add_argument('-r', '--rna', default='test', help='The name of RNA and working directory.')
    parser.add_argument('-m', '--map', default=None, help='The RNA cryo-EM map file.')
    parser.add_argument('-c', '--contour', default=None, type=float, help='The contour value.')
    parser.add_argument('-s', '--seq', default=None, help='The RNA sequence file.')
    parser.add_argument('-g', '--gpu', default=None, help='The gpu device index to be used.')
    parser.add_argument('-n', '--ncpu', default=10, type=int, help='The number of CPUs to be used.')
    parser.add_argument('-i', '--input', default=None, help='The input file')
    args = parser.parse_args()
    print(args, flush=True)
    
    rna_name = args.rna
    map_file = args.map
    contour = args.contour
    seq = args.seq
    gpu = args.gpu
    ncpu = args.ncpu
    inputfile = args.input

    if inputfile is not None:
        if not os.path.exists(inputfile):
            raise ValueError(f"The input file '{inputfile}' does not exist!")
        with open(inputfile) as f:
            lines = f.read().splitlines()
        for line in lines:
            line = line.split()
            if len(line) < 2:
                continue
            if line[0] == "rna":
                rna_name = line[1]
            elif line[0] == "map":
                map_file = line[1]
            elif line[0] == "contour":
                contour = line[1]
                try:
                    contour = float(contour)
                except ValueError:
                    print(f"The contour level value in the input file '{inputfile}' should be a number. The given one is '{line[1]}'.")
                    sys.exit()
            elif line[0] == "seq":
                seq = line[1]
            elif line[0].lower() == "gpu":
                if gpu is not None:
                    continue
                gpu = line[1]
                try:
                    gpu_idx = int(gpu)
                except ValueError:
                    print(f"gpu should be an integer number. The given one is '{line[1]}'.")
                    sys.exit()
            elif line[0].lower() == "ncpu":
                ncpu = line[1]
                try:
                    ncpu = int(ncpu)
                except ValueError:
                    print(f"The ncpu in the input file '{inputfile}' should be an integer. The given one is '{line[1]}'.")
                    sys.exit()
            else:
                print(f"Skip the unknown parameter: '{' '.join(line)}'")

    if gpu is None:
        gpu = ""

    if map_file is None or not os.path.exists(map_file):
        raise ValueError(f"The cryo-EM map {map_file} is invalid. Please provide a valid RNA cryo-EM map.")
    if not (map_file.endswith(".map") or map_file.endswith(".mrc")):
        raise ValueError(f"The cryo-EM map {map_file} is invalid. A valid cryo-EM map should end with '.map' or '.mrc' (case-sensitive).")
    if (not isinstance(contour,int)) and (not isinstance(contour,float)):
        raise ValueError(f"The contour level value {contour} is invalid. Please provide an integer or float number.")
    if seq is None:
        raise ValueError(f"Please provide the RNA sequence which contains only A, G, C, U (case-insensitive), and different chains are set apart by hyphens.")
    for c in seq:
        if c not in "AUGCaugc-":
            raise ValueError(f"Invalid nucleotide name '{c}' in the give sequence '{seq}'.")
    seq = seq.upper()
    list_seq = seq.split("-")

    DEEPCRYORNA_HOME = os.getenv("DEEPCRYORNA_HOME")
    if DEEPCRYORNA_HOME is None:
        raise ValueError(f"Please set the environment variable 'DEEPCRYORNA_HOME', following the installation instructions in the manual.")
    if not os.path.isdir(DEEPCRYORNA_HOME):
        raise ValueError(f"The DEEPCRYORNA_HOME '{DEEPCRYORNA_HOME}' set in the environment variable does not exist!")

    print(f"\n****** Input Info ******")
    print(f"RNA name: {rna_name}")
    print(f"Map file: {map_file}")
    print(f"Contour value: {contour}")
    print(f"RNA sequence: {seq}")
    print(f"GPU: {gpu}")
    print(f"nCPU: {ncpu}")
    print(f"DEEPCRYORNA_HOME: {DEEPCRYORNA_HOME}\n")

    print(f"------------------------------------------------------------------------------------------------------------------------")
    print(f"Start to reconstruct the 3D structure of RNA '{rna_name}' based on the cryo-EM map '{map_file}'.")
    print(f"------------------------------------------------------------------------------------------------------------------------\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    time1 = time.time()

    processed_map_file, map_data, map_origin = preprocess_cryoEM_map(DEEPCRYORNA_HOME, map_file, contour)

    reconstruct_structures_from_cryoEM_map(DEEPCRYORNA_HOME, rna_name, map_data, map_origin, contour, list_seq, gpu, ncpu)

    time2 = time.time()
    duration = time2 - time1
    display_duration_time(duration)
