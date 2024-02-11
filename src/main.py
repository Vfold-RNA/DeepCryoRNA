import sys
import os
import numpy as np
import argparse
import mrcfile
import time
from predict_structures_from_atoms import cluster_atoms, convert_atoms_to_structures, generate_structure_files
from predict_atoms_by_Unet_from_cryoEM_maps import convert_map_to_atoms_by_two_patch_sizes
from detect_ring_penetration import shrink_ring_if_ring_penetration
from run_em import run_EM_by_QRNAS_parallel

def preprocess_cryoEM_map(DEEPCRYORNA_HOME, map_file, contour):
    if map_file.endswith(".map"):
        map_name = map_file.split("/")[-1].split(".map")[0]
    elif map_file.endswith(".mrc"):
        map_name = map_file.split("/")[-1].split(".mrc")[0]
    else:
        raise ValueError(f"Cryo-EM map name should end with '.map' or '.mrc'.")

    print(f"(1) Preprocessing the input cryo-EM map '{map_file}' using the software ChimeraX,")
    print(f"    including masking the map based on the given contour level value {contour} and changing the voxel size to 0.5 angstrom ...\n")
    print(f"      The preprocessing log info by ChimeraX will be stored in the file 'chimerax.log'.")

    cmd = f"chimerax --nogui {DEEPCRYORNA_HOME}/preprocess_cryoEM_map.py {map_file} {contour} > chimerax.log 2>&1"
    print(f"      Chimerax is currently running the following command:")
    print(f"        {cmd}", flush=True)
    os.system(cmd)

    processed_map_file = f"tmp/{map_name}_masked_apix0.5.mrc"
    if not os.path.exists(processed_map_file):
        raise ValueError(f"Error in preprocessing the cryo-EM map. Please see the log file 'chimerax.log' for detailed info.")
    else:
        print(f"      The masked and scaled cryo-EM map '{processed_map_file}' has been generated.\n", flush=True)

    with mrcfile.open(processed_map_file) as f:
        map_data = f.data
        x = f.header['origin']['x']
        y = f.header['origin']['y']
        z = f.header['origin']['z']
        map_origin = np.array([x,y,z])
        apix = f.voxel_size['x']
        assert apix == 0.5

    return processed_map_file, map_data, map_origin


def reconstruct_structures_from_cryoEM_map(DEEPCRYORNA_HOME, rna_name, map_data, map_origin, contour, list_seq, gpu, ncpu):
    unet_model = f"{DEEPCRYORNA_HOME}/DeepCryoRNA_Unet.hdf5"
    if not os.path.exists(unet_model):
        raise ValueError(f"The Unet model 'DeepCryoRNA_Unet.hdf5' does not exist in DEEPCRYORNA_HOME '{DEEPCRYORNA_HOME}'.")
    patch_size1 = 128
    patch_size2 = 64

    print(f"(2) Converting the processed cryo-EM map into atoms based on the Unet model ...\n", flush=True)
    pred_atoms1, pred_prob1, pred_atoms2, pred_prob2 = convert_map_to_atoms_by_two_patch_sizes(unet_model, map_data, contour, patch_size1, patch_size2)
    np.save(f"tmp/pred_atoms_{rna_name}_ps{patch_size1}.npy",pred_atoms1)
    np.save(f"tmp/pred_atoms_{rna_name}_ps{patch_size2}.npy",pred_atoms2)
    
    print(f"(3) Clustering the predicted atoms ...\n", flush=True)
    clustered_atoms1 = cluster_atoms(pred_atoms1)
    clustered_atoms2 = cluster_atoms(pred_atoms2)
    np.save(f"tmp/clustered_atoms_{rna_name}_ps{patch_size1}.npy",clustered_atoms1)
    np.save(f"tmp/clustered_atoms_{rna_name}_ps{patch_size2}.npy",clustered_atoms2)
    clustered_atoms1 = clustered_atoms1.T # transpose the clusted atoms since the cryo-EM map data is stored in z-y-x axis ordering.
    clustered_atoms2 = clustered_atoms2.T # transpose the clusted atoms since the cryo-EM map data is stored in z-y-x axis ordering.
   
    print(f"(4) Generating the 3D structures based on the clustered atoms,")
    print(f"    including combining the clustered atoms into nucleotides, threading the nucleotides into chains, and assigning the sequences ...\n", flush=True)

    out_file_name1 = f"ps{patch_size1}/pred_{rna_name}_ps{patch_size1}"
    if not os.path.exists(f"ps{patch_size1}"):
        os.mkdir(f"ps{patch_size1}")
    list_best_alignments1, list_mapped_nt_chains1, dict_right_chain_order1, list_outpdbfile_names1 = convert_atoms_to_structures(DEEPCRYORNA_HOME, list_seq, map_origin, clustered_atoms1, ncpu, out_file_name1)

    out_file_name2 = f"ps{patch_size2}/pred_{rna_name}_ps{patch_size2}"
    if not os.path.exists(f"ps{patch_size2}"):
        os.mkdir(f"ps{patch_size2}")
    list_best_alignments2, list_mapped_nt_chains2, dict_right_chain_order2, list_outpdbfile_names2 = convert_atoms_to_structures(DEEPCRYORNA_HOME, list_seq, map_origin, clustered_atoms2, ncpu, out_file_name2)
    assert dict_right_chain_order1 == dict_right_chain_order2

    list_best_alignments_scores = []
    for alignment in list_best_alignments1:
        list_best_alignments_scores.append(alignment[2])
    for alignment in list_best_alignments2:
        list_best_alignments_scores.append(alignment[2])
    best_alignments_index = sorted(list(range(len(list_best_alignments_scores))), key=lambda k: list_best_alignments_scores[k], reverse=True) 

    list_best_alignments = []
    list_mapped_nt_chains = []
    num_best_alignment = 10
    for idx in best_alignments_index[0:num_best_alignment]:
        if idx < len(list_best_alignments1):
            list_best_alignments.append(list_best_alignments1[idx])
            list_mapped_nt_chains.append(list_mapped_nt_chains1[idx])
        else:
            list_best_alignments.append(list_best_alignments2[idx-len(list_best_alignments1)])
            list_mapped_nt_chains.append(list_mapped_nt_chains2[idx-len(list_best_alignments1)])
    print(f"      For patch size Both ({patch_size2} and {patch_size1}):", flush=True)
    print(f"        Got {len(list_best_alignments)} best alignments.", flush=True)
    alignment_result_file = f"psBoth/best_alignment_results_psBoth.txt"
    if not os.path.exists(f"psBoth"):
        os.mkdir(f"psBoth")
    if len(list_best_alignments) > 0:
        print(f"        The best alignment results are stored in '{alignment_result_file}'.", flush=True)
        with open(alignment_result_file, "w") as f:
            for i, alignment in enumerate(list_best_alignments):
                f.write(f"best alignment {i+1}\n")
                f.write(f"alignment score: {alignment[2]}\n")
                f.write(f"native seq: {alignment[0]}\n")
                f.write(f"  pred seq: {alignment[1]}\n\n")

    out_file_name3 = f"psBoth/pred_{rna_name}"
    list_outpdbfile_names3 = generate_structure_files(DEEPCRYORNA_HOME, map_origin, dict_right_chain_order1, list_mapped_nt_chains, out_file_name3, ncpu)

    num_outpdbfile = len(list_outpdbfile_names3)
    if num_outpdbfile == 1:
        print(f"        {num_outpdbfile} structure has been generated in the following file:")
        print(f"          {list_outpdbfile_names3[0]}\n")
    elif num_outpdbfile > 1:
        print(f"        {num_outpdbfile} structures have been generated in the following files:")
        for pdbfile_name in list_outpdbfile_names3:
            print(f"          {pdbfile_name}")
        print()
    else:
        print(f"        Failed in predicting the RNA structures from the given cryo-EM map.\n")

    print(f"(5) Performing energy minimization by QRNAS ...\n", flush=True)
    list_em_files1 = run_EM_by_QRNAS_parallel(DEEPCRYORNA_HOME, list_outpdbfile_names1, ncpu)
    print(f"      {len(list_em_files1)} energy-minimized structures for patch size {patch_size1} have been generated in the following files:")
    for em_file in list_em_files1:
        print(f"        {em_file}", flush=True)
    print()

    list_em_files2 = run_EM_by_QRNAS_parallel(DEEPCRYORNA_HOME, list_outpdbfile_names2, ncpu)
    print(f"      {len(list_em_files2)} energy-minimized structures for patch size {patch_size2} have been generated in the following files:")
    for em_file in list_em_files2:
        print(f"        {em_file}", flush=True)
    print()

    list_em_files3 = run_EM_by_QRNAS_parallel(DEEPCRYORNA_HOME, list_outpdbfile_names3, ncpu)
    print(f"      {len(list_em_files3)} energy-minimized structures for patch size Both ({patch_size2} and {patch_size1}) have been generated in the following files:")
    for em_file in list_em_files3:
        print(f"        {em_file}", flush=True)

    return list_outpdbfile_names3


def display_duration_time(duration):
    hours = int(duration/3600.)
    duration = duration - hours*3600
    mins = int(duration/60.)
    duration = duration - mins*60
    secs = int(duration)
    list_info = []
    if hours > 0:
        list_info.append(f"{hours} hr")
    if mins > 0:
        list_info.append(f"{mins} min")
    if secs > 0:
        list_info.append(f"{secs} s")
    time_info = " ".join(list_info)
    print(f"\nThe job is finished in {time_info}.")


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

    list_outpdbfile_names = reconstruct_structures_from_cryoEM_map(DEEPCRYORNA_HOME, rna_name, map_data, map_origin, contour, list_seq, gpu, ncpu)

    time2 = time.time()
    duration = time2 - time1
    display_duration_time(duration)
