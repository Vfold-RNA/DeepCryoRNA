import numpy as np
import sys

def read_pdb(pdbname):
    with open(pdbname) as f:
        lines = f.read().splitlines()

    list_atoms = []
    dict_atoms = {}
    dict_nt = {}
    dict_chain = {}
    pre_nt_idx = None
    pre_chain_name = None
    for line in lines:
        if len(line) < 6:
            continue
        if line[0:6] == "ENDMDL":
            break
        if line[0:4] != "ATOM":
            continue

        atom_name = line[12:16].strip()
        resi_name = line[17:20].strip()
        if resi_name == "RA3":
            resi_name = "A"
        elif resi_name == "RG3":
            resi_name = "G"
        elif resi_name == "RC3":
            resi_name = "C"
        elif resi_name == "RU3":
            resi_name = "U"
        elif resi_name not in ["A","G","C","U"]:
            continue
        if "H" in atom_name:
            continue
        chain_name = line[20:22]
        nt_idx = int(line[22:26].strip())
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coord = np.array([x,y,z])

        if pre_nt_idx is None or (nt_idx == pre_nt_idx and chain_name == pre_chain_name):
            dict_atoms[atom_name] = coord
        elif pre_nt_idx is not None and nt_idx != pre_nt_idx and chain_name == pre_chain_name:
            dict_nt[pre_nt_idx] = {"resi_name": pre_resi_name, "atoms": dict_atoms}
            dict_atoms = {}
            dict_atoms[atom_name] = coord
        elif pre_chain_name is not None and chain_name != pre_chain_name:
            dict_nt[pre_nt_idx] = {"resi_name": pre_resi_name, "atoms": dict_atoms}
            dict_chain[pre_chain_name] = dict_nt
            dict_atoms = {}
            dict_atoms[atom_name] = coord
            dict_nt = {}
        pre_nt_idx = nt_idx
        pre_resi_name = resi_name
        pre_chain_name = chain_name

    dict_nt[nt_idx] = {"resi_name": resi_name, "atoms": dict_atoms}
    dict_chain[chain_name] = dict_nt

    return dict_chain

def check_ring_penetration_between_one_bond_and_one_ring(bond_vertices, ring_vertices):
    assert len(ring_vertices) >= 3, print("There should be at least 3 vertices in a ring.")
    assert len(bond_vertices) == 2, print("There should be only 2 vertices in a bond.")

    bond_start, bond_end = bond_vertices

    ring_normal = np.cross(ring_vertices[1] - ring_vertices[0], ring_vertices[2] - ring_vertices[0])
    
    bond_vec = bond_end - bond_start

    if np.dot(bond_vec, ring_normal) == 0.0:
        return False

    t = np.dot(ring_vertices[0] - bond_start, ring_normal) / np.dot(bond_vec, ring_normal)
    if not (0 <= t <= 1):
        return False
    intersection_point = bond_start + bond_vec * t

    for i in range(len(ring_vertices)):
        curr_vertice = ring_vertices[i]
        if i == len(ring_vertices) - 1:
            next_vertice = ring_vertices[0]
        else:
            next_vertice = ring_vertices[i+1]

        edge = next_vertice - curr_vertice
        p = intersection_point - curr_vertice
        if np.dot(ring_normal, np.cross(edge, p)) < 0:
            return False

    return True

def get_ring_by_name(chain,nt_idx,nt,ring_name):
    if ring_name == "sugar":
        if "C4'" not in nt["atoms"].keys() or "C3'" not in nt["atoms"].keys() or "C2'" not in nt["atoms"].keys() or "C1'" not in nt["atoms"].keys() or "O4'" not in nt["atoms"].keys():
            raise ValueError(f"Incomplete sugar ring in nt {nt_idx} in chain {chain}.")
        coord_C4s = nt["atoms"]["C4'"]
        coord_C3s = nt["atoms"]["C3'"]
        coord_C2s = nt["atoms"]["C2'"]
        coord_C1s = nt["atoms"]["C1'"]
        coord_O4s = nt["atoms"]["O4'"]
        ring_vertices = [coord_C4s,coord_C3s,coord_C2s,coord_C1s,coord_O4s]
        return ring_vertices
    elif ring_name == "pentagon":
        if nt["resi_name"] in ["C","U"]:
            return None
        if "N9" not in nt["atoms"].keys() or "C4" not in nt["atoms"].keys() or "C5" not in nt["atoms"].keys() or "N7" not in nt["atoms"].keys() or "C8" not in nt["atoms"].keys():
            raise ValueError(f"Incomplete pentagon ring in nt {nt_idx} in chain {chain}.")
        coord_N9 = nt["atoms"]["N9"]
        coord_C4 = nt["atoms"]["C4"]
        coord_C5 = nt["atoms"]["C5"]
        coord_N7 = nt["atoms"]["N7"]
        coord_C8 = nt["atoms"]["C8"]
        ring_vertices = [coord_N9,coord_C4,coord_C5,coord_N7,coord_C8]
        return ring_vertices
    elif ring_name == "hexagon":
        if "N1" not in nt["atoms"].keys() or "C2" not in nt["atoms"].keys() or "N3" not in nt["atoms"].keys() or "C4" not in nt["atoms"].keys() or "C5" not in nt["atoms"].keys() or "C6" not in nt["atoms"].keys():
            raise ValueError(f"Incomplete hexagon ring in nt {nt_idx} in chain {chain}.")
        coord_N1 = nt["atoms"]["N1"]
        coord_C2 = nt["atoms"]["C2"]
        coord_N3 = nt["atoms"]["N3"]
        coord_C4 = nt["atoms"]["C4"]
        coord_C5 = nt["atoms"]["C5"]
        coord_C6 = nt["atoms"]["C6"]
        ring_vertices = [coord_N1,coord_C2,coord_N3,coord_C4,coord_C5,coord_C6]
        return ring_vertices
    else:
        raise ValueError(f"Wrong ring name: {ring_name}, which should be 'sugar', 'pentagon', or 'hexagon'")

def get_bonds_by_atom(dict_chain,chain_name,nt_idx,nt,atom_name):
    assert nt["resi_name"] in ["A","G","C","U"], print(f"wrong nt name: {nt['resi_name']}")

    list_bond_vertices = []
    list_bond_atoms_names = []
    bond_start = nt["atoms"][atom_name]
    if atom_name == "P":
        bond_end = nt["atoms"]["OP1"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"OP1"])
        bond_end = nt["atoms"]["OP2"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"OP2"])
        bond_end = nt["atoms"]["O5'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"O5'"])
        if nt_idx-1 in dict_chain[chain_name].keys():
            bond_end = dict_chain[chain_name][nt_idx-1]["atoms"]["O3'"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"O3'"])
    elif atom_name == "OP1":
        bond_end = nt["atoms"]["P"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"P"])
    elif atom_name == "OP2":
        bond_end = nt["atoms"]["P"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"P"])
    elif atom_name == "O5'":
        if "P" in nt["atoms"].keys():
            bond_end = nt["atoms"]["P"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"P"])
        bond_end = nt["atoms"]["C5'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C5'"])
    elif atom_name == "C5'":
        bond_end = nt["atoms"]["O5'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"O5'"])
        bond_end = nt["atoms"]["C4'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4'"])
    elif atom_name == "C4'":
        bond_end = nt["atoms"]["C5'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C5'"])
        bond_end = nt["atoms"]["C3'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C3'"])
        bond_end = nt["atoms"]["O4'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"O4'"])
    elif atom_name == "C3'":
        bond_end = nt["atoms"]["C4'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4'"])
        bond_end = nt["atoms"]["C2'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C2'"])
        bond_end = nt["atoms"]["O3'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"O3'"])
    elif atom_name == "O3'":
        bond_end = nt["atoms"]["C3'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C3'"])
        if nt_idx+1 in dict_chain[chain_name].keys():
            bond_end = dict_chain[chain_name][nt_idx+1]["atoms"]["P"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"P"])
    elif atom_name == "C2'":
        bond_end = nt["atoms"]["C3'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C3'"])
        bond_end = nt["atoms"]["C1'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C1'"])
        bond_end = nt["atoms"]["O2'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"O2'"])
    elif atom_name == "C1'":
        bond_end = nt["atoms"]["C2'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C2'"])
        bond_end = nt["atoms"]["O4'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"O4'"])
        if nt["resi_name"] in ["A","G"]:
            bond_end = nt["atoms"]["N9"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N9"])
        else:
            bond_end = nt["atoms"]["N1"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N1"])
    elif atom_name == "O4'":
        bond_end = nt["atoms"]["C1'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C1'"])
        bond_end = nt["atoms"]["C4'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4'"])
    elif atom_name == "N9":
        bond_end = nt["atoms"]["C1'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C1'"])
        bond_end = nt["atoms"]["C4"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4'"])
        bond_end = nt["atoms"]["C8"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C8"])
    elif atom_name == "C8":
        bond_end = nt["atoms"]["N9"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"N9"])
        bond_end = nt["atoms"]["N7"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"N7"])
    elif atom_name == "N7":
        bond_end = nt["atoms"]["C8"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C8"])
        bond_end = nt["atoms"]["C5"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C5"])
    elif atom_name == "C6":
        bond_end = nt["atoms"]["C5"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C5"])
        bond_end = nt["atoms"]["N1"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"N1"])
        if "O6" in nt["atoms"].keys():
            bond_end = nt["atoms"]["O6"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"O6"])
        if "N6" in nt["atoms"].keys():
            bond_end = nt["atoms"]["N6"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N6"])
    elif atom_name == "C5":
        bond_end = nt["atoms"]["C4"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4"])
        bond_end = nt["atoms"]["C6"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C6"])
        if "N7" in nt["atoms"].keys():
            bond_end = nt["atoms"]["N7"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N7"])
    elif atom_name == "C4":
        bond_end = nt["atoms"]["C5"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C5"])
        bond_end = nt["atoms"]["N3"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"N3"])
        if "N9" in nt["atoms"].keys():
            bond_end = nt["atoms"]["N9"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N9"])
        if "N4" in nt["atoms"].keys():
            bond_end = nt["atoms"]["N4"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N4"])
    elif atom_name == "N3":
        bond_end = nt["atoms"]["C4"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4"])
        bond_end = nt["atoms"]["C2"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C2"])
    elif atom_name == "C2":
        bond_end = nt["atoms"]["N1"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"N1"])
        bond_end = nt["atoms"]["N3"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"N3"])
        if "O2" in nt["atoms"].keys():
            bond_end = nt["atoms"]["O2"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"O2"])
        if "N2" in nt["atoms"].keys():
            bond_end = nt["atoms"]["N2"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"N2"])
    elif atom_name == "N1":
        bond_end = nt["atoms"]["C2"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C2"])
        bond_end = nt["atoms"]["C6"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C6"])
        if "O2" in nt["atoms"].keys():
            bond_end = nt["atoms"]["O2"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"O2"])
        if nt["resi_name"] in ["C","U"]:
            bond_end = nt["atoms"]["C1'"]
            bond_vertices = [bond_start,bond_end]
            list_bond_vertices.append(bond_vertices)
            list_bond_atoms_names.append([atom_name,"C1'"])
    elif atom_name in ["N6","O6"]:
        bond_end = nt["atoms"]["C6"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C6"])
    elif atom_name in ["N2","O2"]:
        bond_end = nt["atoms"]["C2"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C2"])
    elif atom_name in ["N4","O4"]:
        bond_end = nt["atoms"]["C4"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C4"])
    elif atom_name == "O2'":
        bond_end = nt["atoms"]["C2'"]
        bond_vertices = [bond_start,bond_end]
        list_bond_vertices.append(bond_vertices)
        list_bond_atoms_names.append([atom_name,"C2'"])
    elif atom_name == "OP3":
        pass
    else:
        raise ValueError(f"wrong atom name: {atom_name}")

    assert len(list_bond_vertices) == len(list_bond_atoms_names)
    return list_bond_vertices, list_bond_atoms_names

def shrink_ring(dict_chain,chain_name,nt_idx,nt,ring_name):
    if ring_name == "sugar":
        coord_C4s = nt["atoms"]["C4'"]
        for atom_name in ["C3'","C2'","C1'","O4'"]:
            nt["atoms"][atom_name] = coord_C4s + (nt["atoms"][atom_name]-coord_C4s)*0.15
        dict_chain[chain_name][nt_idx] = nt
    elif ring_name == "pentagon":
        coord_N9 = nt["atoms"]["N9"]
        for atom_name in ["C8","N7","C5","C4"]:
            nt["atoms"][atom_name] = coord_N9 + (nt["atoms"][atom_name]-coord_N9)*0.15
        dict_chain[chain_name][nt_idx] = nt
    elif ring_name == "hexagon":
        coord_C5 = nt["atoms"]["C5"]
        for atom_name in ["C4","N3","C2","N1","C6"]:
            nt["atoms"][atom_name] = coord_C5 + (nt["atoms"][atom_name]-coord_C5)*0.15
        dict_chain[chain_name][nt_idx] = nt
    else:
        raise ValueError(f"wrong ring name: {ring_name}")
    return dict_chain

def check_ring_penetration(dict_chain):
    has_penetration = False

    for chain1 in dict_chain:
        for nt_idx1 in dict_chain[chain1]:
            nt1 = dict_chain[chain1][nt_idx1]
            for ring_name in ["sugar","pentagon","hexagon"]:
                ring_vertices = get_ring_by_name(chain1,nt_idx1,nt1,ring_name)
                if ring_vertices is None:
                    continue
                penetration = False
                for chain2 in dict_chain:
                    for nt_idx2 in dict_chain[chain2]:
                        nt2 = dict_chain[chain2][nt_idx2]
                        coord_C4s = nt2["atoms"]["C4'"]
                        dis = np.linalg.norm(coord_C4s - ring_vertices[0])
                        if dis > 20.0:
                            continue
                        for atom_name in nt2["atoms"]:
                            if chain1 == chain2 and nt_idx1 == nt_idx2:
                                if ring_name == "sugar":
                                    if atom_name in ["C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'","N9"]:
                                        continue
                                    if atom_name == "N1" and nt1["resi_name"] in ["C","U"]:
                                        continue
                                elif ring_name == "pentagon":
                                    if atom_name in ["C1'","N9","C8","N7","C5","C4","N3","C6"]:
                                        continue
                                elif ring_name == "hexagon":
                                    if atom_name in ["N9","N7","C4","C5","C6","N1","C2","N3","O6","N2","N6","O2","O4","N4"]:
                                        continue
                                    if atom_name == "C1'" and nt1["resi_name"] in ["C","U"]:
                                        continue
                            atom_coord = nt2["atoms"][atom_name]
                            dis = np.linalg.norm(atom_coord - ring_vertices[0])
                            if dis > 4.0:
                                continue
                            list_bond_vertices, list_bond_atoms_names = get_bonds_by_atom(dict_chain,chain2,nt_idx2,nt2,atom_name)
                            for bond_vertices, bond_atoms_names in zip(list_bond_vertices, list_bond_atoms_names):
                                if chain1 == chain2 and nt_idx1 == nt_idx2:
                                    if ring_name == "sugar":
                                        if bond_atoms_names[0] in ["C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'","N9"]:
                                            if bond_atoms_names[1] in ["C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'","N9"]:
                                                continue
                                        if nt1["resi_name"] in ["C","U"]:
                                            if bond_atoms_names[0] in ["N1","C1'"]:
                                                if bond_atoms_names[1] in ["N1","C1'"]:
                                                    continue
                                    elif ring_name == "pentagon":
                                        if bond_atoms_names[0] in ["C1'","N9","C8","N7","C5","C4","N3","C6"]:
                                            if bond_atoms_names[1] in ["C1'","N9","C8","N7","C5","C4","N3","C6"]:
                                                continue
                                    elif ring_name == "hexagon":
                                        if bond_atoms_names[0] in ["N9","N7","C4","C5","C6","N1","C2","N3","O6","N2","N6","O2","O4","N4"]:
                                            if bond_atoms_names[1] in ["N9","N7","C4","C5","C6","N1","C2","N3","O6","N2","N6","O2","O4","N4"]:
                                                continue
                                        if nt1["resi_name"] in ["C","U"]:
                                            if bond_atoms_names[0] in ["N1","C1'"]:
                                                if bond_atoms_names[1] in ["N1","C1'"]:
                                                    continue
                                if check_ring_penetration_between_one_bond_and_one_ring(bond_vertices, ring_vertices):
                                    penetration = True
                                    break
                            if penetration:
                                break
                        if penetration:
                            break
                    if penetration:
                        break
                if penetration:
                    #print(f"penetration, {chain1}, {nt_idx1}, {ring_name}, {chain2}, {nt_idx2}, {atom_name}")
                    has_penetration = True
                    dict_chain = shrink_ring(dict_chain,chain1,nt_idx1,nt1,ring_name)
    return has_penetration, dict_chain

def output_PDB_structure(dict_chain,outfile):
    f = open(outfile,"w")

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

    atom_idx = 1
    for i in range(len(list_chain)):
        chain_name = list_chain[i]
        if chain_name not in dict_chain.keys():
            continue
        for nt_idx in range(0,max(dict_chain[chain_name].keys())+1):
            if nt_idx in dict_chain[chain_name].keys():
                nt = dict_chain[chain_name][nt_idx]
            else:
                continue
            nt_name = nt["resi_name"]
            atoms = nt["atoms"]
            for atom_name in atoms:
                coord = atoms[atom_name]
                coordx = f"{coord[0]:.3f}"
                if len(coordx) > 8:
                    coordx = coordx[:8]
                coordy = f"{coord[1]:.3f}"
                if len(coordy) > 8:
                    coordy = coordy[:8]
                coordz = f"{coord[2]:.3f}"
                if len(coordz) > 8:
                    coordz = coordz[:8]
                f.write(f"ATOM  {atom_idx:>5}  {atom_name:<3} {nt_name:>3}{chain_name}{nt_idx:>4}    {coordx:>8}{coordy:>8}{coordz:>8}\n")
                atom_idx += 1
        f.write("TER\n")
    f.write("END\n")
    f.close()

def shrink_ring_if_ring_penetration(pdbname, outpdbname):
    dict_chain = read_pdb(pdbname)
    has_penetration, dict_chain = check_ring_penetration(dict_chain)
    dict_chain_added = {}
    if has_penetration:
        output_PDB_structure(dict_chain,outpdbname)
        #print(f"Ring penetration is detected in the file '{pdbname}'.\nThe new structure with shrinked rings is written into the file '{outpdbname}'")
    return has_penetration
