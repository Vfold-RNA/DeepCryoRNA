import sys, os
import numpy as np
import multiprocessing
from functools import partial
from detect_ring_penetration import read_pdb, shrink_ring_if_ring_penetration

def get_non_bonded_nt(infile):
    list_breaking_nts = []

    dict_chain = read_pdb(infile)

    list_chain = []
    list_single_chain = []
    for i in range(26):
        list_single_chain.append(chr(ord("A")+i))
    for i in range(26):
        list_single_chain.append(chr(ord("a")+i))
    for i in range(10):
        list_single_chain.append(str(i))
    for c in list_single_chain:
        list_chain.append(f" {c}")
    for c1 in list_single_chain:
        for c2 in list_single_chain:
            list_chain.append(f"{c1}{c2}")

    for i in range(len(list_chain)):
        chain_name = list_chain[i]
        if chain_name not in dict_chain.keys():
            continue
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

def rewrite_PDB_file(infile,outfile):
    with open(infile) as f:
        lines = f.read().splitlines()

    dict_chains = {}
    list_chain_candidates = []
    list_single_chain = []
    for i in range(26):
        list_single_chain.append(chr(ord("A")+i))
    for i in range(26):
        list_single_chain.append(chr(ord("a")+i))
    for i in range(10):
        list_single_chain.append(str(i))
    for c in list_single_chain:
        list_chain_candidates.append(f" {c}")
    for c1 in list_single_chain:
        for c2 in list_single_chain:
            list_chain_candidates.append(f"{c1}{c2}")

    list_breaking_nts = get_non_bonded_nt(infile)

    f = open(outfile,"w")

    old_nt_idx = None
    old_chain = None
    chain_num = 0
    r_last_O3s = None
    for line in lines:
        if len(line) < 10:
            continue
        if line[0:4] != "ATOM" and line[0:6] != "HETATM":
            continue

        new_line = list(line)
        nt_idx = int("".join(new_line[22:26]))
        chain = new_line[20]+new_line[21]
        if old_nt_idx is None or nt_idx == old_nt_idx or ((nt_idx - old_nt_idx) == 1 and chain == old_chain and (chain, nt_idx) not in list_breaking_nts):
            pass
        else:
            chain_num += 1
        if chain_num < len(list_chain_candidates):
            new_chain = list_chain_candidates[chain_num]
        else:
            new_chain = list_chain_candidates[-1]
        new_line[20] = new_chain[0]
        new_line[21] = new_chain[1]
        new_line = "".join(new_line)
        f.write(new_line+"\n")
        old_nt_idx = nt_idx
        old_chain = chain
        if new_chain not in dict_chains.keys():
            dict_chains[new_chain] = chain

    f.close()

    return dict_chains

def rewrite_PDB_file_with_right_chains(dict_chains,infile,outfile):
    with open(infile) as f:
        lines = f.read().splitlines()

    f = open(outfile,"w")
    for line in lines:
        if len(line) < 10:
            continue
        if line[0:4] != "ATOM" and line[0:6] != "HETATM":
            continue
        new_line = list(line)
        chain = new_line[20]+new_line[21]
        right_chain = dict_chains[chain]
        new_line[20] = right_chain[0]
        new_line[21] = right_chain[1]
        new_line = "".join(new_line)
        f.write(new_line+"\n")
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
        if nt_idx == 1 and atom_name == "OP3":
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

        line[6:11] = list(f"{atom_idx:>5}")
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

    dict_chains = rewrite_PDB_file(infile, f"{prefix}rechained-{pdbname}")
    has_penetration = shrink_ring_if_ring_penetration(f"{prefix}rechained-{pdbname}", f"{prefix}shrinked-{pdbname}")
    cmd = f"{DEEPCRYORNA_HOME}/QRNAS/QRNA -i {prefix}shrinked-{pdbname} -o {prefix}em-{pdbname} -c {DEEPCRYORNA_HOME}/QRNAS/configfile.txt > {prefix}em{tag}.log 2>&1"
    os.system(cmd)
   
    for i in range(5):
        has_penetration = shrink_ring_if_ring_penetration(f"{prefix}em-{pdbname}", f"{prefix}shrinked-{pdbname}")
        if not has_penetration or i == 4:
            #print(f"No penetration; running energy minimization {i}", flush=True)
            cmd = f"{DEEPCRYORNA_HOME}/QRNAS/QRNA -i {prefix}shrinked-{pdbname} -o {prefix}em-{pdbname} -c {DEEPCRYORNA_HOME}/QRNAS/configfile2.txt > {prefix}em{tag}.log 2>&1"
            os.system(cmd)
            break
        else:
            #print(f"Penetration; running energy minimization {i}", flush=True)
            cmd = f"{DEEPCRYORNA_HOME}/QRNAS/QRNA -i {prefix}shrinked-{pdbname} -o {prefix}em-{pdbname} -c {DEEPCRYORNA_HOME}/QRNAS/configfile.txt > {prefix}em{tag}.log 2>&1"
            os.system(cmd)
    
    rewrite_PDB_file_with_right_chains(dict_chains, f"{prefix}em-{pdbname}", f"{prefix}em-{pdbname}")
    format_PDB_file(f"{prefix}em-{pdbname}")
    
    if os.path.exists(f"{prefix}rechained-{pdbname}"):
        os.remove(f"{prefix}rechained-{pdbname}")

    if os.path.exists(f"{prefix}shrinked-{pdbname}"):
        os.remove(f"{prefix}shrinked-{pdbname}")

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
