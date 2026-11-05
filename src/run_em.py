import sys, os
import numpy as np
import multiprocessing
from functools import partial
from detect_ring_penetration import read_pdb, shrink_ring_if_ring_penetration

def get_non_bonded_nt(infile):
    list_breaking_nts = []

    dict_chain = read_pdb(infile)

    for chain_name in dict_chain:
        for nt_idx in range(0,max(dict_chain[chain_name].keys())):
            if nt_idx in dict_chain[chain_name].keys():
                nt1 = dict_chain[chain_name][nt_idx]
                r_O3s = nt1["atoms"]["O3'"]
            else:
                continue
            if nt_idx + 1 in dict_chain[chain_name].keys():
                nt2 = dict_chain[chain_name][nt_idx+1]
                r_P = nt2["atoms"]["P"]
            else:
                continue
            dis = np.linalg.norm(r_P - r_O3s)
            if dis > 4.5:
                list_breaking_nts.append((chain_name,nt_idx+1))

    return list_breaking_nts

def split_large_structure_into_files(infile):
    fn = infile.split(".pdb")[0]

    with open(infile) as f:
        lines = f.read().splitlines()

    dict_chains = {}

    list_breaking_nts = get_non_bonded_nt(infile)

    list_segments = []
    segment = []

    old_nt_idx = None
    old_chain = None
    chain_num = 0
    r_last_O3s = None
    for line in lines:
        if len(line) < 10:
            continue
        if line[0:4] != "ATOM" and line[0:6] != "HETATM":
            continue

        nt_idx = int(line[22:26])
        chain = line[21]
        line = list(line)
        line[4:11] = [' ']*7 # remove atom id
        line = "".join(line)
        if old_nt_idx is None or (chain == old_chain and nt_idx == old_nt_idx) or ((nt_idx - old_nt_idx) == 1 and chain == old_chain and (chain, nt_idx) not in list_breaking_nts):
            segment.append(line)            
        else:
            list_segments.append(segment)
            segment = [line]
        old_nt_idx = nt_idx
        old_chain = chain
    list_segments.append(segment)

    list_segment_files = []
    for i in range(len(list_segments)):
        segment_file = f"{fn}-{i}.pdb"
        list_segment_files.append(segment_file)
        with open(segment_file,"w") as f:
            for line in list_segments[i]:
                f.write(line+"\n")

    return list_segment_files

def combine_segment_files(list_segment_files, outfile):
    f = open(outfile,"w")
  
    atom_id = 1
    for segment_file in list_segment_files:
        with open(segment_file) as g:
            lines = g.read().splitlines()

        for line in lines:
            if len(line) < 10:
                continue
            if line[0:4] != "ATOM" and line[0:6] != "HETATM":
                continue
            line = list(line)
            atom_id_str = f"{atom_id:>7}"
            line[4:11] = list(atom_id_str)
            line = "".join(line)
            f.write(line+"\n")
            atom_id += 1

    f.close()

def format_PDB_file(infile,outfile=None):
    with open(infile) as f:
        lines = f.read().splitlines()

    if outfile is None:
        outfile = infile

    f = open(outfile,"w")

    atom_idx = 1
    for line in lines:
        if len(line) < 4:
            f.write(line+"\n")
            continue
        if line[0:4] != "ATOM":
            f.write(line+"\n")
            continue
        if line[0:3] == "TER":
            f.write("TER+\n")
            continue
        line = list(line)
        atom_name = "".join(line[12:16]).strip()
        if "H" in atom_name:
            continue
        nt_idx = "".join(line[22:26])
        nt_idx = int(nt_idx)
        if atom_name == "OP3":
            continue
        nt_name = "".join(line[17:20]).strip()
        if nt_name in ["RA5","RA3"]:
            line[17:20] = list("  A")
        elif nt_name in ["RG5","RG3"]:
            line[17:20] = list("  G")
        elif nt_name in ["RC5","RC3"]:
            line[17:20] = list("  C")
        elif nt_name in ["RU5","RU3"]:
            line[17:20] = list("  U")

        line[4:11] = list(f"{atom_idx:>7}")
        atom_idx += 1

        for i in range(len(line),80):
            line.append("")
        line[56:60] = ["1",".","0","0"]
        line[62:66] = ["1",".","0","0"]
        if "H" in atom_name:
            line[77] = "H"
        elif "O" in atom_name:
            line[77] = "O"
        elif "C" in atom_name:
            line[77] = "C"
        elif "N" in atom_name:
            line[77] = "N"
        elif "P" in atom_name:
            line[77] = "P"
        else:
            raise ValueError(f"unknown element in atom {atom_name}.")
        
        line = "".join(line)
        f.write(line+"\n")

    f.close()

def run_EM_by_QRNAS(DEEPCRYORNA_HOME, infile_tag):
    infile, tag = infile_tag

    names = infile.split("/")
    prefix = "/".join(names[0:-1])
    if prefix:
        prefix = prefix + "/"
    pdbname = names[-1]

    list_segment_files = split_large_structure_into_files(infile)

    list_em_segment_files = []
    list_shrinked_segment_files = []
    list_em_log_files = []
    for segid, segment_file in enumerate(list_segment_files):
        shrinked_file = segment_file.split(".pdb")[0] + "-shrinked.pdb"
        has_penetration = shrink_ring_if_ring_penetration(segment_file, shrinked_file)
        em_file = shrinked_file.split(".pdb")[0] + "-em.pdb"
        log_file = f"{prefix}em{tag}-{segid}.log"

        cmd = f"{DEEPCRYORNA_HOME}/QRNAS/QRNA -i {shrinked_file} -o {em_file} -c {DEEPCRYORNA_HOME}/QRNAS/configfile.txt > {log_file} 2>&1"
        os.system(cmd)
   
        for i in range(3):
            has_penetration = shrink_ring_if_ring_penetration(em_file, shrinked_file)
            if not has_penetration or i == 2:
                #print(f"No penetration; running energy minimization {i}", flush=True)
                cmd = f"{DEEPCRYORNA_HOME}/QRNAS/QRNA -i {shrinked_file} -o {em_file} -c {DEEPCRYORNA_HOME}/QRNAS/configfile2.txt > {log_file} 2>&1"
                os.system(cmd)
                break
            else:
                #print(f"Penetration; running energy minimization {i}", flush=True)
                cmd = f"{DEEPCRYORNA_HOME}/QRNAS/QRNA -i {shrinked_file} -o {em_file} -c {DEEPCRYORNA_HOME}/QRNAS/configfile.txt > {log_file} 2>&1"
                os.system(cmd)

        list_em_segment_files.append(em_file)
        list_shrinked_segment_files.append(shrinked_file)
        list_em_log_files.append(log_file)
    
    outfile = f"{prefix}em-{pdbname}"
    combine_segment_files(list_em_segment_files, outfile)
    format_PDB_file(outfile)

    for fn in list_segment_files:
        if os.path.exists(fn):
            os.remove(fn)

    for fn in list_em_segment_files:
        if os.path.exists(fn):
            os.remove(fn)

    for fn in list_shrinked_segment_files:
        if os.path.exists(fn):
            os.remove(fn)

    for fn in list_em_log_files:
        if os.path.exists(fn):
            os.remove(fn)

    return f"{prefix}em-{pdbname}"

def run_EM_by_QRNAS_parallel(DEEPCRYORNA_HOME, list_infile, ncpu):
    ncpu_available = multiprocessing.cpu_count()
    if ncpu > ncpu_available:
        ncpu = ncpu_available

    pool = multiprocessing.Pool(processes=ncpu)
    partial_function = partial(run_EM_by_QRNAS, DEEPCRYORNA_HOME)
    list_tag = list(range(1,len(list_infile)+1))
    list_variable_arg = [(infile, tag) for infile, tag in zip(list_infile, list_tag)]
    list_em_files = pool.map(partial_function, list_variable_arg)

    pool.close()
    pool.join()

    return list_em_files


if __name__ == "__main__":
    DEEPCRYORNA_HOME = os.getenv("DEEPCRYORNA_HOME")
    infile = sys.argv[1]
    run_EM_by_QRNAS(DEEPCRYORNA_HOME, (infile, 1))

    #list_infile = sys.argv[1:]
    #run_EM_by_QRNAS_parallel(DEEPCRYORNA_HOME, list_infile, 10)
