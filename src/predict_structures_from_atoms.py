import numpy as np
import copy
import itertools
from skimage.feature import peak_local_max
from scipy.ndimage import convolve
import ctypes
import multiprocessing
from functools import partial
from alignment import global_alignment, get_alignment_score

def get_distance(coord1,coord2):
    return np.linalg.norm(coord1-coord2)

def remove_atom_P_coords(dict_atoms):
    if "P" not in dict_atoms.keys():
        return dict_atoms
    to_remove_idx = []
    for idx, coord in enumerate(dict_atoms["P"]):
        dis = get_distance(dict_atoms["C4'"][0],coord)
        if dis > 5.0:
            to_remove_idx.append(idx)
            continue
        if "O3'" in dict_atoms.keys():
            dis = get_distance(dict_atoms["O3'"][0],coord)
            if dis < 3.0:
                to_remove_idx.append(idx)
                continue
        if "C3'" in dict_atoms.keys():
            dis = get_distance(dict_atoms["C3'"][0],coord)
            if dis < 3.0:
                to_remove_idx.append(idx)
                continue
        if "C2'" in dict_atoms.keys():
            dis = get_distance(dict_atoms["C2'"][0],coord)
            if dis < 4.0:
                to_remove_idx.append(idx)
                continue
    for idx in sorted(to_remove_idx,reverse=True):
        del dict_atoms["P"][idx]

    if len(dict_atoms["P"]) == 0:
        dict_atoms.pop("P")
    elif len(dict_atoms["P"]) > 1:
        dict_atoms["P"] = [dict_atoms["P"][0]]
    return dict_atoms      
   
def remove_atom_coords(dict_atoms,atom_name,ref_coord,upper_dis):
    if atom_name not in dict_atoms.keys():
        return dict_atoms
    list_distances = []
    for idx, coord in enumerate(dict_atoms[atom_name]):
        list_distances.append(get_distance(ref_coord,coord))
    min_dis = min(list_distances)
    min_dis_idx = list_distances.index(min_dis)
    if min_dis > upper_dis:
        dict_atoms.pop(atom_name)
    else:
        dict_atoms[atom_name] = [ dict_atoms[atom_name][min_dis_idx] ]
    return dict_atoms

def change_base_atom_name(dict_atoms,nttype):
    if nttype == 1:
        if "N9" not in dict_atoms.keys():
            if "N1" in dict_atoms.keys():
                dict_atoms["N9"] = copy.deepcopy(dict_atoms["N1"])
        if "C4" not in dict_atoms.keys():
            if "C2" in dict_atoms.keys():
                dict_atoms["C4"] = copy.deepcopy(dict_atoms["C2"])
        if "C8" not in dict_atoms.keys():
            if "C6" in dict_atoms.keys():
                dict_atoms["C8"] = copy.deepcopy(dict_atoms["C6"])
        for name in ["N1","C2","C6"]:
            if name in dict_atoms.keys():
                dict_atoms.pop(name)

    if nttype == 2:
        if "N1" not in dict_atoms.keys():
            if "N9" in dict_atoms.keys():
                dict_atoms["N1"] = copy.deepcopy(dict_atoms["N9"])
        if "C2" not in dict_atoms.keys():
            if "C4" in dict_atoms.keys():
                dict_atoms["C2"] = copy.deepcopy(dict_atoms["C4"])
        if "C6" not in dict_atoms.keys():
            if "C8" in dict_atoms.keys():
                dict_atoms["C6"] = copy.deepcopy(dict_atoms["C8"])
        for name in ["N9","C4","C8"]:
            if name in dict_atoms.keys():
                dict_atoms.pop(name)
    return dict_atoms

def remove_invalid_nt(list_nts):
    num_nt = len(list_nts)
    to_remove_idx = []
    for i in range(num_nt):
        if i in to_remove_idx:
            continue
        idx1, nttype1, dict_atoms1 = list_nts[i]
        if "P" not in dict_atoms1.keys():
            continue
        coord_P1 = dict_atoms1["P"]
        coord_C4s1 = dict_atoms1["C4'"]
        if "C5'" in dict_atoms1.keys():
            coord_C5s1 = dict_atoms1["C5'"]
        else:
            coord_C5s1 = None
        if "C3'" in dict_atoms1.keys():
            coord_C3s1 = dict_atoms1["C3'"]
        else:
            coord_C3s1 = None
        dis_C4s_C5s_1 = -1.0
        if coord_C5s1 is not None:
            dis_C4s_C5s_1 = np.linalg.norm(coord_C4s1-coord_C5s1)
        dis_C4s_C3s_1 = -1.0
        if coord_C3s1 is not None:
            dis_C4s_C3s_1 = np.linalg.norm(coord_C4s1-coord_C3s1)
        for j in range(i+1,num_nt):
            if j in to_remove_idx:
                continue
            idx2, nttype2, dict_atoms2 = list_nts[j]
            if "P" not in dict_atoms2.keys():
                continue
            coord_P2 = dict_atoms2["P"]
            dis = np.linalg.norm(coord_P2-coord_P1)
            if dis > 1.0:
                continue
            coord_C4s2 = dict_atoms2["C4'"]
            if "C5'" in dict_atoms2.keys():
                coord_C5s2 = dict_atoms2["C5'"]
            else:
                coord_C5s2 = None
            if "C3'" in dict_atoms2.keys():
                coord_C3s2 = dict_atoms2["C3'"]
            else:
                coord_C3s2 = None
            dis_C4s_C5s_2 = -1.0
            if coord_C5s2 is not None:
                dis_C4s_C5s_2 = np.linalg.norm(coord_C4s2-coord_C5s2)
            dis_C4s_C3s_2 = -1.0
            if coord_C3s2 is not None:
                dis_C4s_C3s_2 = np.linalg.norm(coord_C4s2-coord_C3s2)
            if dis_C4s_C5s_1 > 2.2 or dis_C4s_C3s_1 > 2.2:
                to_remove_idx.append(i)
            elif dis_C4s_C5s_2 > 2.2 or dis_C4s_C3s_2 > 2.2:
                to_remove_idx.append(j)
            else:
                to_remove_idx.append(i)
            break

    to_remove_idx.sort(reverse=True)
    for idx in to_remove_idx:
        #print(f"Deleting nt {idx}")
        del list_nts[idx]

    return list_nts

def get_template_nucleotide(DEEPCRYORNA_HOME):
    list_backbone_atoms = ["P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'","NX","P2"]
    list_sugar_base_AG_atoms = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'","N9","C4","C8"]
    list_sugar_base_CU_atoms = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'","N1","C2","C6"]

    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/templates_backbone.txt") as f:
        lines = f.read().splitlines()
    list_backbone_templates = []
    for line in lines:
        line = line.split()
        dict_backbone_template = {}
        for i,atom_name in enumerate(list_backbone_atoms):
            coord = np.asarray(line[i*3:(i+1)*3],dtype=float)
            dict_backbone_template[atom_name] = coord
        list_backbone_templates.append(dict_backbone_template)

    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/sugar_base_AG.txt") as f:
        lines = f.read().splitlines()
    list_sugar_base_AG_templates = []
    for line in lines:
        line = line.split()
        dict_sugar_base_AG_template = {}
        for i,atom_name in enumerate(list_sugar_base_AG_atoms):
            coord = np.asarray(line[i*3:(i+1)*3],dtype=float)
            dict_sugar_base_AG_template[atom_name] = coord
        list_sugar_base_AG_templates.append(dict_sugar_base_AG_template)

    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/sugar_base_CU.txt") as f:
        lines = f.read().splitlines()
    list_sugar_base_CU_templates = []
    for line in lines:
        line = line.split()
        dict_sugar_base_CU_template = {}
        for i,atom_name in enumerate(list_sugar_base_CU_atoms):
            coord = np.asarray(line[i*3:(i+1)*3],dtype=float)
            dict_sugar_base_CU_template[atom_name] = coord
        list_sugar_base_CU_templates.append(dict_sugar_base_CU_template)

    dict_base_templates = {"Abase":{},"Gbase":{},"Cbase":{},"Ubase":{}}
    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/template_Abase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Abase"][atom_name] = coord

    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/template_Gbase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Gbase"][atom_name] = coord

    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/template_Cbase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Cbase"][atom_name] = coord

    with open(f"{DEEPCRYORNA_HOME}/nucleotide_template/template_Ubase.pdb") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.split()
        if line[0] != "ATOM":
            continue
        atom_name = line[2]
        coord = np.asarray(line[6:9],dtype=float)
        dict_base_templates["Ubase"][atom_name] = coord

    return list_backbone_templates, list_sugar_base_AG_templates, list_sugar_base_CU_templates, dict_base_templates

def assign_atom_to_nt(DEEPCRYORNA_HOME,atomseg,apix=0.5):
    atom_class_to_name = {1:"P",2:"O5'",3:"C5'",4:"C4'",5:"C3'",6:"C2'",7:"C1'",8:"O4'",9:"O2'",10:"O3'",11:"OP1",12:"OP2",13:"N9",14:"C4",15:"C8",16:"N1",17:"C2",18:"C6"}
    coord_C4s = np.argwhere(atomseg==4)
    print(f"        {len(coord_C4s)} C4' atoms are predicted.")
    boxsize = 14
    list_nts = []
    for idx, coord in enumerate(coord_C4s):
        list_nt = []
        dict_atoms = {}
        x0, y0, z0 = coord
        coord = coord * apix
        dict_atoms["C4'"] = [coord]
        for i in range(-boxsize,boxsize+1):
            x = x0 + i
            if x < 0 or x >= atomseg.shape[0]:
                continue
            for j in range(-boxsize,boxsize+1):
                y = y0 + j
                if y < 0 or y >= atomseg.shape[1]:
                    continue
                for k in range(-boxsize,boxsize+1):
                    z = z0 + k
                    if z < 0 or z >= atomseg.shape[2]:
                        continue
                    if atomseg[x,y,z] == 0:
                        continue
                    atom_class = atomseg[x,y,z]
                    atom_name = atom_class_to_name[atom_class]
                    if atom_name not in dict_atoms.keys():
                        dict_atoms[atom_name] = [np.array([x,y,z])*apix]
                    else:
                        dict_atoms[atom_name].append(np.array([x,y,z])*apix)
        
        for atom_name in ["C5'","C3'","O4'"]:
            ref_coord = coord
            upper_dis = 3.0
            dict_atoms = remove_atom_coords(dict_atoms,atom_name,ref_coord,upper_dis)

        if "C5'" in dict_atoms.keys():
            ref_coord = dict_atoms["C5'"][0]
            upper_dis = 3.0
            dict_atoms = remove_atom_coords(dict_atoms,"O5'",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 3.5
            dict_atoms = remove_atom_coords(dict_atoms,"O5'",ref_coord,upper_dis)

        if "C3'" in dict_atoms.keys():
            ref_coord = dict_atoms["C3'"][0]
            upper_dis = 3.0
            dict_atoms = remove_atom_coords(dict_atoms,"C2'",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"O3'",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 3.5
            dict_atoms = remove_atom_coords(dict_atoms,"C2'",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"O3'",ref_coord,upper_dis)

        if "O4'" in dict_atoms.keys():
            ref_coord = dict_atoms["O4'"][0]
            upper_dis = 3.0
            dict_atoms = remove_atom_coords(dict_atoms,"C1'",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 3.5
            dict_atoms = remove_atom_coords(dict_atoms,"C1'",ref_coord,upper_dis)

        if "C2'" in dict_atoms.keys():
            ref_coord = dict_atoms["C2'"][0]
            upper_dis = 3.0
            dict_atoms = remove_atom_coords(dict_atoms,"O2'",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 3.8
            dict_atoms = remove_atom_coords(dict_atoms,"O2'",ref_coord,upper_dis)

        if "C1'" in dict_atoms.keys():
            ref_coord = dict_atoms["C1'"][0]
            upper_dis = 3.0
            dict_atoms = remove_atom_coords(dict_atoms,"N9",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"N1",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 5.5
            dict_atoms = remove_atom_coords(dict_atoms,"N9",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"N1",ref_coord,upper_dis)

        if "C1'" in dict_atoms.keys():
            ref_coord = dict_atoms["C1'"][0]
            upper_dis = 4.0
            dict_atoms = remove_atom_coords(dict_atoms,"C8",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"C6",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 5.5
            dict_atoms = remove_atom_coords(dict_atoms,"C8",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"C6",ref_coord,upper_dis)

        if "C1'" in dict_atoms.keys():
            ref_coord = dict_atoms["C1'"][0]
            upper_dis = 4.0
            dict_atoms = remove_atom_coords(dict_atoms,"C4",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"C2",ref_coord,upper_dis)
        else:
            ref_coord = coord
            upper_dis = 6.5
            dict_atoms = remove_atom_coords(dict_atoms,"C4",ref_coord,upper_dis)
            dict_atoms = remove_atom_coords(dict_atoms,"C2",ref_coord,upper_dis)

        atom_name = "P"
        has_ref_coord = False
        for name in ["O5'","C5'"]:
            if name in dict_atoms.keys():
                ref_coord = dict_atoms[name][0]
                upper_dis = 3.0
                if name == "C5'":
                    upper_dis = 4.0
                has_ref_coord = True
                dict_atoms = remove_atom_coords(dict_atoms,atom_name,ref_coord,upper_dis)
                break
        if not has_ref_coord:
            if "P" in dict_atoms.keys():
                dict_atoms.pop("P")

        atom_name = "OP1"
        has_ref_coord = False
        for name in ["P","O5'"]:
            if name in dict_atoms.keys():
                ref_coord = dict_atoms[name][0]
                upper_dis = 3.0
                if name == "O5'":
                    upper_dis = 4.0
                has_ref_coord = True
                dict_atoms = remove_atom_coords(dict_atoms,atom_name,ref_coord,upper_dis)
                break
        if not has_ref_coord:
            if "OP1" in dict_atoms.keys():
                dict_atoms.pop("OP1")

        atom_name = "OP2"
        has_ref_coord = False
        for name in ["P","O5'"]:
            if name in dict_atoms.keys():
                ref_coord = dict_atoms[name][0]
                upper_dis = 3.0
                if name == "O5'":
                    upper_dis = 4.0
                has_ref_coord = True
                dict_atoms = remove_atom_coords(dict_atoms,atom_name,ref_coord,upper_dis)
                break
        if not has_ref_coord:
            if "OP2" in dict_atoms.keys():
                dict_atoms.pop("OP2")

        for atom_name in dict_atoms:
            dict_atoms[atom_name] = dict_atoms[atom_name][0]
        
        num_AG_atoms = 0
        num_CU_atoms = 0
        for name in ["N9","C4","C8"]:
            if name in dict_atoms.keys():
                num_AG_atoms += 1
        for name in ["N1","C2","C6"]:
            if name in dict_atoms.keys():
                num_CU_atoms += 1
        if num_AG_atoms == 0 and num_CU_atoms == 0:
            nttype = 0
        elif num_AG_atoms >= num_CU_atoms:
            nttype = 1
        else:
            nttype = 2
        if nttype > 0:
            dict_atoms = change_base_atom_name(dict_atoms,nttype)
        if len(dict_atoms) == 0:
            continue
        
        list_names = ["C4'","C5'","O5'","P","OP1"]
        for i in range(len(list_names)-1):
            j = i +1
            name1 = list_names[i]
            name2 = list_names[j]
            if name1 in dict_atoms.keys() and name2 in dict_atoms.keys():
                if get_distance(dict_atoms[name1],dict_atoms[name2]) > 3.0:
                    if name2 in dict_atoms.keys():
                        dict_atoms.pop(name2)

        list_names = ["C4'","C3'","C2'","C1'","O4'"]
        for i in range(len(list_names)-1):
            j = i +1
            name1 = list_names[i]
            name2 = list_names[j]
            if name1 in dict_atoms.keys() and name2 in dict_atoms.keys():
                if get_distance(dict_atoms[name1],dict_atoms[name2]) > 3.0:
                    if name2 in dict_atoms.keys():
                        dict_atoms.pop(name2)

        list_names = ["P","OP2"]
        for i in range(len(list_names)-1):
            j = i +1
            name1 = list_names[i]
            name2 = list_names[j]
            if name1 in dict_atoms.keys() and name2 in dict_atoms.keys():
                if get_distance(dict_atoms[name1],dict_atoms[name2]) > 3.0:
                    if name2 in dict_atoms.keys():
                        dict_atoms.pop(name2)

        list_names = ["C2'","O2'"]
        for i in range(len(list_names)-1):
            j = i +1
            name1 = list_names[i]
            name2 = list_names[j]
            if name1 in dict_atoms.keys() and name2 in dict_atoms.keys():
                if get_distance(dict_atoms[name1],dict_atoms[name2]) > 3.0:
                    if name2 in dict_atoms.keys():
                        dict_atoms.pop(name2)

        list_names = ["C3'","O3'"]
        for i in range(len(list_names)-1):
            j = i +1
            name1 = list_names[i]
            name2 = list_names[j]
            if name1 in dict_atoms.keys() and name2 in dict_atoms.keys():
                if get_distance(dict_atoms[name1],dict_atoms[name2]) > 3.0:
                    if name2 in dict_atoms.keys():
                        dict_atoms.pop(name2)

        if len(dict_atoms) < 5:
            continue
        list_nt.append(idx)
        list_nt.append(nttype)
        list_nt.append(dict_atoms)
        list_nts.append(list_nt)

    list_backbone_atoms = ["P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'"]
    list_sugar_base_AG_atoms = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'","N9","C4","C8"]
    list_sugar_base_CU_atoms = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'","N1","C2","C6"]
    list_backbone_templates, list_sugar_base_AG_templates, list_sugar_base_CU_templates, dict_base_templates = get_template_nucleotide(DEEPCRYORNA_HOME)

    for nt in list_nts:
        nt_id, nt_type, atoms = nt
        ref_atom_names = []
        rebuilt_atom_names = []
        for atom_name in list_backbone_atoms:
            if atom_name not in atoms.keys():
                rebuilt_atom_names.append(atom_name)
            else:
                ref_atom_names.append(atom_name)
        if len(rebuilt_atom_names) > 0:
            ref_atom_coords = [atoms[atom_name] for atom_name in ref_atom_names]
            ref_atom_coords = np.asarray(ref_atom_coords)
            rebuilt_atom_coords = rebuild_atom(list_backbone_templates,ref_atom_names,ref_atom_coords,rebuilt_atom_names)
            for tag, atom_name in enumerate(rebuilt_atom_names):
                atoms[atom_name] = rebuilt_atom_coords[tag]

        if True: # check if missing base atoms
            if nt_type == 1:
                list_sugar_base_atoms = list_sugar_base_AG_atoms
                list_sugar_base_templates = list_sugar_base_AG_templates
                rebuilt_atom_names = list_sugar_base_atoms[-3:]
            elif nt_type == 2:
                list_sugar_base_atoms = list_sugar_base_CU_atoms
                list_sugar_base_templates = list_sugar_base_CU_templates
                rebuilt_atom_names = list_sugar_base_atoms[-3:]
            else:
                continue
            ref_atom_names = []
            for name in list_sugar_base_atoms:
                if name in atoms.keys():
                    ref_atom_names.append(name)
            ref_atom_coords = [atoms[atom_name] for atom_name in ref_atom_names]
            ref_atom_coords = np.asarray(ref_atom_coords)
            rebuilt_atom_coords = rebuild_atom(list_sugar_base_templates,ref_atom_names,ref_atom_coords,rebuilt_atom_names)
            for tag, atom_name in enumerate(rebuilt_atom_names):
                atoms[atom_name] = rebuilt_atom_coords[tag]
   
    list_nts = remove_invalid_nt(list_nts)
    return list_nts

def determine_if_two_nts_are_neighbors(nt1,nt2):
    if "O3'" not in nt1[2].keys():
        return False, None
    else:
        atom_O3s_coord = nt1[2]["O3'"]

    if "C3'" not in nt1[2].keys():
        return False, None
    else:
        atom_C3s_coord = nt1[2]["C3'"]

    dict_sugar = {"C5'":np.array([5.621,7.881,-3.587]), "C4'":np.array([6.719,6.889,-3.258]), "C3'":np.array([6.776,5.642,-4.140]),
            "C2'":np.array([7.567,4.725,-3.208]), "C1'":np.array([6.874,4.999,-1.877]), "O4'":np.array([6.486,6.364,-1.919])}

    reference_coords1 = []
    coords1 = []
    for name in dict_sugar:
        if name in nt1[2].keys():
            reference_coords1.append(dict_sugar[name])
            coords1.append(nt1[2][name])
    rmsd1, rot1, tran1 = calc_rmsd([], np.asarray(reference_coords1), np.asarray(coords1))

    reference_coords2 = []
    coords2 = []
    for name in dict_sugar:
        if name in nt2[2].keys():
            reference_coords2.append(dict_sugar[name])
            coords2.append(nt2[2][name])
    rmsd2, rot2, tran2 = calc_rmsd([], np.asarray(reference_coords2), np.asarray(coords2))

    if "P" in nt2[2].keys():
        atom_P_coord = nt2[2]["P"]
        dis = np.linalg.norm(atom_O3s_coord-atom_P_coord)
        if dis <= 3.0:
            return True, abs(dis-1.6)
        dis = np.linalg.norm(atom_C3s_coord-atom_P_coord)
        if rmsd1 <= 0.6 and rmsd2 <= 0.6:
            upper_dis_C3s_P = 4.5
        else:
            upper_dis_C3s_P = 4.0
        if dis <= upper_dis_C3s_P:
            return True, abs(dis-2.6)
    
    if "O5'" in nt2[2].keys():
        atom_O5s_coord = nt2[2]["O5'"]
        if "P" in nt2[2].keys():
            dis_P_O5s = np.linalg.norm(atom_O5s_coord-atom_P_coord)
        else:
            dis_P_O5s = 1000.0
            
        if dis_P_O5s <= 2.6:
            dis = np.linalg.norm(atom_O3s_coord-atom_O5s_coord)
            if dis <= 3.5:
                return True, abs(dis-2.5)
            dis = np.linalg.norm(atom_C3s_coord-atom_O5s_coord)
            if dis <= 4.0:
                return True, abs(dis-3.1)

    return False, None

def determine_if_two_nts_are_good_neighbors(nt1, nt2):
    if "O3'" not in nt1[2].keys():
        return False
    if "P" not in nt2[2].keys():
        return False

    atom_O3s_coord = nt1[2]["O3'"]
    atom_P_coord = nt2[2]["P"]
    dis = np.linalg.norm(atom_O3s_coord-atom_P_coord)

    if dis <= 1.8:
        return True

    if dis >= 2.2:
        return False

    dict_backbone = {"P":np.array([3.063,8.025,-4.135]), "O5'":np.array([4.396,7.181,-3.881]), "C5'":np.array([5.621,7.881,-3.587]), 
            "C4'":np.array([6.719,6.889,-3.258]), "C3'":np.array([6.776,5.642,-4.140]), "O3'":np.array([7.397,5.908,-5.390]),
            "C2'":np.array([7.567,4.725,-3.208]), "O2'":np.array([8.962,4.911,-3.068]), "C1'":np.array([6.874,4.999,-1.877]),
            "O4'":np.array([6.486,6.364,-1.919])}

    reference_coords1 = []
    coords1 = []
    for name in dict_backbone:
        if name in nt1[2].keys():
            reference_coords1.append(dict_backbone[name])
            coords1.append(nt1[2][name])
    rmsd1, rot1, tran1 = calc_rmsd([], np.asarray(reference_coords1), np.asarray(coords1))
    if rmsd1 > 1.0:
        return False

    reference_coords2 = []
    coords2 = []
    for name in dict_backbone:
        if name in nt2[2].keys():
            reference_coords2.append(dict_backbone[name])
            coords2.append(nt2[2][name])
    rmsd2, rot2, tran2 = calc_rmsd([], np.asarray(reference_coords2), np.asarray(coords2))
    if rmsd2 > 1.0:
        return False

    return True

def filter_multiple_neighbors(list_nt,list_multiple_neighbors,multiple_type):
    list_good_neighbors = []
    list_bad_neighbors_to_remove = []
    for i, neighbor in enumerate(list_multiple_neighbors):
        nt1_idx, nt2_idx = neighbor
        if multiple_type == "branch":
            assert nt1_idx == list_multiple_neighbors[0][0]
        elif multiple_type == "join":
            assert nt2_idx == list_multiple_neighbors[0][1]
        nt1 = list_nt[nt1_idx]
        nt2 = list_nt[nt2_idx]
        good_neighbor = determine_if_two_nts_are_good_neighbors(nt1, nt2)
        if good_neighbor:
            list_good_neighbors.append([nt1_idx,nt2_idx])
        else:
            list_bad_neighbors_to_remove.append([nt1_idx,nt2_idx])
    if len(list_good_neighbors) > 1:
        return list_good_neighbors, list_bad_neighbors_to_remove
    elif len(list_good_neighbors) == 1:
        return [], list_bad_neighbors_to_remove
    else:
        return list_multiple_neighbors, []

def filter_all_multiple_neighbors(list_nt,list_all_multiple_neighbors,multiple_type):
    list_bad_neighbors_to_remove = []
    for i in range(len(list_all_multiple_neighbors)):
        list_all_multiple_neighbors[i], list_bad_neighbors_to_remove_tmp = filter_multiple_neighbors(list_nt,list_all_multiple_neighbors[i],multiple_type)
        list_bad_neighbors_to_remove.extend(list_bad_neighbors_to_remove_tmp)
    list_all_multiple_neighbors = list(filter(None,list_all_multiple_neighbors))
    return list_all_multiple_neighbors, list_bad_neighbors_to_remove

def get_chain(list_neighbors,seed):
    chain_forward = [seed[0],seed[1]]
    list_neighbors.remove(seed)
    while True:
        next_item = chain_forward[-1]
        found_next = False
        for item in list_neighbors:
            if item[0] == next_item:
                chain_forward.append(item[1])
                list_neighbors.remove(item)
                found_next = True
                break
        if not found_next:
            break
    chain_backward = [seed[0]]
    while True:
        last_item = chain_backward[-1]
        found_last = False
        for item in list_neighbors:
            if item[1] == last_item:
                chain_backward.append(item[0])
                list_neighbors.remove(item)
                found_last = True
                break
        if not found_last:
            break
    chain_backward.reverse()
    chain_backward.pop()
    chain = chain_backward + chain_forward
    if chain[-1] == chain[0]:
        chain.pop()
    return chain, list_neighbors

def check_same_object(list_nt_chains):
    passcheck = True
    for i in range(len(list_nt_chains)):
        for j in range(len(list_nt_chains[i])):
            for m in range(len(list_nt_chains)):
                for n in range(len(list_nt_chains[m])):
                    if i == m and j == n:
                        continue
                    if list_nt_chains[i][j] is list_nt_chains[m][n]:
                        #print(f"Same object: {i}-{j}-{list_nt_chains[i][j]} {m}-{n}-{list_nt_chains[m][n]}")
                        #print(f"Same object: {i}-{j} {m}-{n}")
                        passcheck = False
    return passcheck

def get_angle(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_in_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees

def get_chain_break_type(nt1, nt2, nt3, chain_length):
    idx1, type1, atoms1 = nt1
    idx2, type2, atoms2 = nt2
    idx3, type3, atoms3 = nt3
    r_c4s_1 = atoms1["C4'"]
    r_c4s_2 = atoms2["C4'"]
    r_c4s_3 = atoms3["C4'"]
    dis_c4s_c4s = np.linalg.norm(r_c4s_2-r_c4s_3)
    agl = get_angle(r_c4s_1,r_c4s_2,r_c4s_3)

    if dis_c4s_c4s <= 4.0:
        return 0

    dict_hp_nt1 = {"P":np.array([6.913,5.099,-6.683]), "C5'":np.array([8.988,3.595,-6.135]), "C4'":np.array([9.376,2.167,-5.806]), 
            "C3'":np.array([8.750,1.087,-6.688]), "C2'":np.array([8.920,-0.112,-5.756]), 
            "C1'":np.array([8.486,0.493,-4.425]), "O4'":np.array([8.896,1.851,-4.467])}
    dict_hp_nt2 = {"P":np.array([-6.022,-6.126,-0.961]), "C5'":np.array([-4.826,-8.391,-1.509]), "C4'":np.array([-3.467,-8.977,-1.838]), 
            "C3'":np.array([-2.309,-8.509,-0.956]), "C2'":np.array([-1.146,-8.847,-1.888]), 
            "C1'":np.array([-1.684,-8.332,-3.219]), "O4'":np.array([-3.086,-8.547,-3.177])}
    dict_hp_nt3 = {"P":np.array([-1.758,-8.408,1.587]), "C5'":np.array([0.473,-9.668,1.039]), "C4'":np.array([1.932,-9.427,0.710]), 
            "C3'":np.array([2.654,-8.408,1.592]), "C2'":np.array([3.815,-8.064,0.660]), 
            "C1'":np.array([3.084,-7.921,-0.671]), "O4'":np.array([2.020,-8.860,-0.629])}
    dict_hp_nt4 = {"P":np.array([3.063,-8.025,4.135]), "C5'":np.array([5.621,-7.881,3.587]), "C4'":np.array([6.719,-6.889,3.258]), 
            "C3'":np.array([6.776,-5.642,4.140]), "C2'":np.array([7.567,-4.725,3.208]), 
            "C1'":np.array([6.874,-4.999,1.877]), "O4'":np.array([6.486,-6.364,1.919])}

    r_dinuc_hp0 = []
    r_dinuc_hp_decoy0 = []
    for name in dict_hp_nt1:
        if name in atoms2.keys():
            r_dinuc_hp0.append(dict_hp_nt1[name])
            r_dinuc_hp_decoy0.append(atoms2[name])

    r_dinuc_hp = copy.deepcopy(r_dinuc_hp0)
    r_dinuc_hp_decoy = copy.deepcopy(r_dinuc_hp_decoy0)
    for name in dict_hp_nt2:
        if name in atoms3.keys():
            r_dinuc_hp.append(dict_hp_nt2[name])
            r_dinuc_hp_decoy.append(atoms3[name])
    rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_hp), np.asarray(r_dinuc_hp_decoy))
    if rmsd <= 1.5:
        return "NP"

    r_dinuc_hp = copy.deepcopy(r_dinuc_hp0)
    r_dinuc_hp_decoy = copy.deepcopy(r_dinuc_hp_decoy0)
    for name in dict_hp_nt3:
        if name in atoms3.keys():
            r_dinuc_hp.append(dict_hp_nt3[name])
            r_dinuc_hp_decoy.append(atoms3[name])
    rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_hp), np.asarray(r_dinuc_hp_decoy))
    if rmsd <= 1.5:
        return "NPP"

    r_dinuc_hp = copy.deepcopy(r_dinuc_hp0)
    r_dinuc_hp_decoy = copy.deepcopy(r_dinuc_hp_decoy0)
    for name in dict_hp_nt4:
        if name in atoms3.keys():
            r_dinuc_hp.append(dict_hp_nt4[name])
            r_dinuc_hp_decoy.append(atoms3[name])
    rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_hp), np.asarray(r_dinuc_hp_decoy))
    if rmsd <= 1.5:
        return "NP"

    if agl < 100.0 or dis_c4s_c4s > 21.0:
        num_P = round(dis_c4s_c4s/6.0)
        if num_P < 3:
            return "N"
        elif num_P < 6:
            return "N"+"".join(["P"]*(num_P-2))
        else:
            return "NPPP"

    dict_helix_nt1 = {"P":np.array([3.063,8.025,-4.135]), "O5'":np.array([4.396,7.181,-3.881]), 
            "C5'":np.array([5.621,7.881,-3.587]), "C4'":np.array([6.719,6.889,-3.258]), 
            "O4'":np.array([6.486,6.364,-1.919]), "C1'":np.array([6.874,4.999,-1.877]), 
            "C2'":np.array([7.567,4.725,-3.208]), "C3'":np.array([6.776,5.642,-4.140])}
    dict_helix_nt2 = {"P":np.array([6.913,5.099,-6.683]), "O5'":np.array([7.579,3.668,-6.429]), 
            "C5'":np.array([8.988,3.595,-6.135]), "C4'":np.array([9.376,2.167,-5.806]), 
            "O4'":np.array([8.896,1.851,-4.467]), "C1'":np.array([8.486,0.493,-4.425]), 
            "C2'":np.array([8.920,-0.112,-5.756]), "C3'":np.array([8.750,1.087,-6.688])}
    dict_helix_nt3 = {"P":np.array([8.572,0.556,-9.231]), "O5'":np.array([8.359,-1.008,-8.977]), 
            "C5'":np.array([9.505,-1.830,-8.683]), "C4'":np.array([9.061,-3.241,-8.354]), 
            "O4'":np.array([8.487,-3.248,-7.015]), "C1'":np.array([7.407,-4.169,-6.973]), 
            "C2'":np.array([7.446,-4.913,-8.304]), "C3'":np.array([7.950,-3.812,-9.236])}
    dict_helix_nt4 = {"P":np.array([7.514,-4.163,-11.779]), "O5'":np.array([6.490,-5.364,-11.525]), 
            "C5'":np.array([7.010,-6.675,-11.231]), "C4'":np.array([5.874,-7.623,-10.902]), 
            "O4'":np.array([5.387,-7.318,-9.563]), "C1'":np.array([3.981,-7.510,-9.521]), 
            "C2'":np.array([3.612,-8.157,-10.852]), "C3'":np.array([4.631,-7.503,-11.784])}
    dict_helix_nt5 = {"P":np.array([4.074,-7.563,-14.327]), "O5'":np.array([2.564,-8.020,-14.073]), 
            "C5'":np.array([2.293,-9.405,-13.779]), "C4'":np.array([0.825,-9.588,-13.450]), 
            "O4'":np.array([0.579,-9.069,-12.111]), "C1'":np.array([-0.707,-8.471,-12.069]), 
            "C2'":np.array([-1.368,-8.816,-13.400]), "C3'":np.array([-0.157,-8.816,-14.332])}

    r_dinuc_helix = []
    r_dinuc_decoy = []
    for name in dict_helix_nt1:
        if name in atoms2.keys():
            r_dinuc_helix.append(dict_helix_nt1[name])
            r_dinuc_decoy.append(atoms2[name])

    if dis_c4s_c4s <= 6.5:
        for name in dict_helix_nt2:
            if name in atoms3.keys():
                r_dinuc_helix.append(dict_helix_nt2[name])
                r_dinuc_decoy.append(atoms3[name])
        rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_helix), np.asarray(r_dinuc_decoy))
        if rmsd <= 2.0:
            return 0
    elif dis_c4s_c4s <= 9.0:
        for name in dict_helix_nt2:
            if name in atoms3.keys():
                r_dinuc_helix.append(dict_helix_nt2[name])
                r_dinuc_decoy.append(atoms3[name])
        rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_helix), np.asarray(r_dinuc_decoy))
        if rmsd <= 1.5:
            return 0
    elif dis_c4s_c4s <= 14.0:
        for name in dict_helix_nt3:
            if name in atoms3.keys():
                r_dinuc_helix.append(dict_helix_nt3[name])
                r_dinuc_decoy.append(atoms3[name])
        rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_helix), np.asarray(r_dinuc_decoy))
        if rmsd <= 1.5:
            return 1
    elif dis_c4s_c4s <= 18.0:
        for name in dict_helix_nt4:
            if name in atoms3.keys():
                r_dinuc_helix.append(dict_helix_nt4[name])
                r_dinuc_decoy.append(atoms3[name])
        rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_helix), np.asarray(r_dinuc_decoy))
        if rmsd <= 1.5:
            return 2
    elif dis_c4s_c4s <= 21.0:
        for name in dict_helix_nt5:
            if name in atoms3.keys():
                r_dinuc_helix.append(dict_helix_nt5[name])
                r_dinuc_decoy.append(atoms3[name])
        rmsd, rot, tran = calc_rmsd([], np.asarray(r_dinuc_helix), np.asarray(r_dinuc_decoy))
        if rmsd <= 1.5:
            return 3

    num_P = round(dis_c4s_c4s/6.0)
    if dis_c4s_c4s < 8.0 and chain_length <= 20:
        return "0"
    elif num_P <= 2 and chain_length <= 20:
        return "1"
    elif num_P < 3:
        return "N"
    elif num_P < 6:
        return "N"+"".join(["P"]*(num_P-2))
    else:
        return "NPPP"

def get_decoy_seq_and_closeness_matrix(list_nt,long_chain,calc_close_mat=True):
    new_long_chain = []

    list_break_idx = [-1]
    for i in range(len(long_chain)):
        nt_idx = long_chain[i]
        if nt_idx is None:
            list_break_idx.append(i)
    list_break_idx.append(len(long_chain))

    dict_flanking_chain_length = {}
    for i in range(1,len(list_break_idx)-1):
        idx1 = list_break_idx[i-1]
        idx2 = list_break_idx[i]
        idx3 = list_break_idx[i+1]
        chain1_length = idx2 - idx1 - 1
        chain2_length = idx3 - idx2 - 1
        dict_flanking_chain_length[idx2] = [chain1_length,chain2_length]

    decoy_seq = []
    fuse_bonus = 0
    for i in range(len(long_chain)):
        nt_idx = long_chain[i]
        if nt_idx is not None:
            nt_id, nt_type, atoms = list_nt[nt_idx]
            if nt_type == 0:
                decoy_seq.append("X")
            elif nt_type == 1:
                decoy_seq.append("A")
            elif nt_type == 2:
                decoy_seq.append("C")
            else:
                raise ValueError(f"Unknown nt type: {nt_type}")
            new_long_chain.append(nt_idx)
        else:
            nt_idx_a = long_chain[i-2]
            nt_idx_b = long_chain[i-1]
            nt_idx_c = long_chain[i+1]
            break_type = get_chain_break_type(list_nt[nt_idx_a],list_nt[nt_idx_b],list_nt[nt_idx_c],min(dict_flanking_chain_length[i]))
            if break_type == "N":
                decoy_seq.append("N")
                new_long_chain.append(None)
            elif break_type == "NP":
                decoy_seq.append("NP")
                new_long_chain.append(None)
                new_long_chain.append(None)
            elif break_type == "NPP":
                decoy_seq.append("NPP")
                new_long_chain.append(None)
                new_long_chain.append(None)
                new_long_chain.append(None)
            elif break_type == "NPPP":
                decoy_seq.append("NPPP")
                new_long_chain.append(None)
                new_long_chain.append(None)
                new_long_chain.append(None)
                new_long_chain.append(None)
            elif break_type == "0":
                fuse_bonus += 0
            elif break_type == "1":
                decoy_seq.append("P")
                new_long_chain.append(None)
                fuse_bonus += 0
            elif break_type == 1:
                decoy_seq.append("P")
                new_long_chain.append(None)
                fuse_bonus += min(min(dict_flanking_chain_length[i]), 5)
            elif break_type == 2:
                decoy_seq.append("PP")
                new_long_chain.append(None)
                new_long_chain.append(None)
                fuse_bonus += min(min(dict_flanking_chain_length[i]), 5)
            elif break_type == 3:
                decoy_seq.append("PPP")
                new_long_chain.append(None)
                new_long_chain.append(None)
                new_long_chain.append(None)
                fuse_bonus += min(min(dict_flanking_chain_length[i]), 5)
            else:
                assert break_type == 0
                fuse_bonus += min(min(dict_flanking_chain_length[i]), 5)

    decoy_seq = "".join(decoy_seq)
    assert len(decoy_seq) == len(new_long_chain)

    if calc_close_mat:
        closeness_matrix = np.zeros((len(decoy_seq),len(decoy_seq)),dtype=np.float32)
        for i in range(len(decoy_seq)):
            nt1_idx = new_long_chain[i]
            if nt1_idx is None:
                continue
            nt1_id, nt1_type, atoms1 = list_nt[nt1_idx]
            c4s_coord1 = atoms1["C4'"]
            for j in range(i+1,len(decoy_seq)):
                nt2_idx = new_long_chain[j]
                if nt2_idx is None:
                    continue
                nt2_id, nt2_type, atoms2 = list_nt[nt2_idx]
                c4s_coord2 = atoms2["C4'"]
                dis_c4s_c4s = np.linalg.norm(c4s_coord1-c4s_coord2)
                closeness_matrix[i,j] = dis_c4s_c4s
                closeness_matrix[j,i] = dis_c4s_c4s
    else:
        closeness_matrix = None

    return decoy_seq, closeness_matrix, new_long_chain, fuse_bonus

def align_sequence_for_one_long_chain(list_native_seq,list_nt,long_chain):
    decoy_seq, closeness_matrix, new_long_chain, fuse_bonus = get_decoy_seq_and_closeness_matrix(list_nt,long_chain)

    bool_polymer = True
    for i in range(0,len(list_native_seq)-1):
        if list_native_seq[i] != list_native_seq[i+1]:
            bool_polymer = False
            break
    if len(list_native_seq) == 1:
        bool_polymer = False
   
    if not bool_polymer:
        permutated_list_native_seq = list(itertools.permutations(list_native_seq))
        permutated_list_index = list(itertools.permutations(list(range(len(list_native_seq)))))
    else:
        permutated_list_native_seq = [tuple(list_native_seq)]
        permutated_list_index = [tuple(list(range(len(list_native_seq))))]

    list_alignments = []
    for list_seq, list_index in zip(permutated_list_native_seq, permutated_list_index):
        native_seq = "T".join(list_seq)
        scores, alignments = global_alignment(native_seq, decoy_seq, closeness_matrix)
   
        for score, alignment in zip(scores, alignments):
            seqA, seqB = alignment
            list_alignments.append([seqA,seqB,score+fuse_bonus,new_long_chain,list_index])
    return list_alignments

def align_sequence_for_one_long_chain2(native_seq,list_nt,long_chain,list_native_chain_index):
    decoy_seq, closeness_matrix, new_long_chain, fuse_bonus = get_decoy_seq_and_closeness_matrix(list_nt,long_chain)

    scores, alignments = global_alignment(native_seq, decoy_seq, closeness_matrix)
    list_alignments = []
    list_scores = []
    list_fuse_bonus = []
    for score, alignment in zip(scores, alignments):
        seqA, seqB = alignment
        list_alignments.append([seqA,seqB,score+fuse_bonus,new_long_chain,list_native_chain_index])
        list_scores.append(score+fuse_bonus)
        list_fuse_bonus.append(fuse_bonus)
        
    return list_alignments, list_scores, list_fuse_bonus

def get_alignment_score_for_one_long_chain(DEEPCRYORNA_HOME,list_native_seq,list_nt,long_chain):
    cpp_library = ctypes.CDLL(f'{DEEPCRYORNA_HOME}/alignment_score.so')
    score_cpp_function = cpp_library.get_alignment_score
    score_cpp_function.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    score_cpp_function.restype = ctypes.c_int

    decoy_seq, closeness_matrix, new_long_chain, fuse_bonus = get_decoy_seq_and_closeness_matrix(list_nt,long_chain,calc_close_mat=False)

    bool_polymer = True
    for i in range(0,len(list_native_seq)-1):
        if list_native_seq[i] != list_native_seq[i+1]:
            bool_polymer = False
            break
    if len(list_native_seq) == 1:
        bool_polymer = False
   
    if not bool_polymer:
        permutated_list_native_seq = list(itertools.permutations(list_native_seq))
        permutated_list_index = list(itertools.permutations(list(range(len(list_native_seq)))))
    else:
        permutated_list_native_seq = [tuple(list_native_seq)]
        permutated_list_index = [tuple(list(range(len(list_native_seq))))]

    list_alignment_score_info = []
    for list_seq, list_index in zip(permutated_list_native_seq, permutated_list_index):
        native_seq = "T".join(list_seq)
        score = score_cpp_function(native_seq.encode(), decoy_seq.encode())
        list_alignment_score_info.append((long_chain,native_seq,list_index,score+fuse_bonus,decoy_seq))

    return list_alignment_score_info

def align_sequence_for_all_long_chains_efficiently(list_native_seq,list_nt,list_all_long_chains,DEEPCRYORNA_HOME,ncpu):
    list_high_alignment_score_info = []
    num_high_score_alignment = 50

    ncpu_available = multiprocessing.cpu_count()
    if ncpu > ncpu_available:
        ncpu = ncpu_available

    pool = multiprocessing.Pool(processes=ncpu)
    partial_function = partial(get_alignment_score_for_one_long_chain, DEEPCRYORNA_HOME, list_native_seq, list_nt)

    if len(list_all_long_chains)%10000 == 0:
        num_chunk = int(len(list_all_long_chains)/10000)
    else:
        num_chunk = int(len(list_all_long_chains)/10000) + 1
    for i in range(num_chunk):
        ib = i*10000
        ie = (i+1)*10000
        if ie > len(list_all_long_chains):
            ie = len(list_all_long_chains)
        list_alignment_score_info = pool.map(partial_function,list_all_long_chains[ib:ie])
        list_alignment_score_info = [info for sublist in list_alignment_score_info for info in sublist]
        list_high_alignment_score_info.extend(list_alignment_score_info)
        if len(list_high_alignment_score_info) >= 10000:
            list_high_alignment_score_info.sort(key=lambda x:x[3], reverse=True)
            list_high_alignment_score_info = list_high_alignment_score_info[0:num_high_score_alignment]

    pool.close()
    pool.join()

    if len(list_high_alignment_score_info) > num_high_score_alignment:
        list_high_alignment_score_info.sort(key=lambda x:x[3], reverse=True)
        list_high_alignment_score_info = list_high_alignment_score_info[0:num_high_score_alignment]

    list_alignments = []
    list_alignments_cluster = []
    list_scores = []
    list_fuse_bonus = []
    for num, high_alignment_score_info in enumerate(list_high_alignment_score_info):
        long_chain, native_seq, list_native_chain_index, score, decoy_seq = high_alignment_score_info
        alignments, scores, fuse_bonus = align_sequence_for_one_long_chain2(native_seq,list_nt,long_chain,list_native_chain_index)
        list_alignments.extend(alignments)
        list_scores.extend(scores)
        list_fuse_bonus.extend(fuse_bonus)
        if len(alignments) > 0:
            list_alignments_cluster.append(alignments[0])

    if len(list_scores) > 0:
        max_score = max(list_scores)
        idx_max_score = list_scores.index(max_score)
        fuse_bonus_for_max_score = list_fuse_bonus[idx_max_score]
        for alignment in list_alignments:
            alignment[2] -= fuse_bonus_for_max_score
        
    return list_alignments, list_alignments_cluster

def get_short_chains(list_native_seq, neighbors_candidate):
    low_chain_len = 4

    list_chains = []
    while True:
        seed = neighbors_candidate[0]
        chain, neighbors_candidate = get_chain(neighbors_candidate,seed)
        if len(chain) >= low_chain_len:
            list_chains.append(chain)
        if not neighbors_candidate:
            break

    if(check_same_object(list_chains)):
        pass
    else:
        print("!!! Not pass same object check !!!")
        exit()
    return list_chains

def determine_if_two_short_chains_are_neighbors(chain1, chain2, list_nt, cutoff=20.0):
    chain1_nt1 = list_nt[chain1[0]]
    chain1_nt2 = list_nt[chain1[1]]
    chain1_nt3 = list_nt[chain1[-2]]
    chain1_nt4 = list_nt[chain1[-1]]

    chain2_nt1 = list_nt[chain2[0]]
    chain2_nt2 = list_nt[chain2[1]]
    chain2_nt3 = list_nt[chain2[-2]]
    chain2_nt4 = list_nt[chain2[-1]]

    if "C4'" not in chain1_nt1[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain1[0]}\n{chain1_nt1}")
    if "C4'" not in chain1_nt2[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain1[1]}\n{chain1_nt2}")
    if "C4'" not in chain1_nt3[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain1[-2]}\n{chain1_nt3}")
    if "C4'" not in chain1_nt4[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain1[-1]}\n{chain1_nt4}")

    if "C4'" not in chain2_nt1[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain2[0]}\n{chain2_nt1}")
    if "C4'" not in chain2_nt2[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain2[1]}\n{chain2_nt2}")
    if "C4'" not in chain2_nt3[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain2[-2]}\n{chain2_nt3}")
    if "C4'" not in chain2_nt4[2].keys():
        raise ValueError(f"There is no C4' atom in nt {chain2[-1]}\n{chain2_nt4}")

    rC4s_chain1_nt1 = chain1_nt1[2]["C4'"]
    rC4s_chain1_nt2 = chain1_nt2[2]["C4'"]
    rC4s_chain1_nt3 = chain1_nt3[2]["C4'"]
    rC4s_chain1_nt4 = chain1_nt4[2]["C4'"]

    rC4s_chain2_nt1 = chain2_nt1[2]["C4'"]
    rC4s_chain2_nt2 = chain2_nt2[2]["C4'"]
    rC4s_chain2_nt3 = chain2_nt3[2]["C4'"]
    rC4s_chain2_nt4 = chain2_nt4[2]["C4'"]

    dis1 = np.linalg.norm(rC4s_chain1_nt4 - rC4s_chain2_nt1)
    dis2 = np.linalg.norm(rC4s_chain2_nt4 - rC4s_chain1_nt1)

    neighbor_type = []
    if dis1 <= cutoff:
        neighbor_type.append(1)
    else:
        neighbor_type.append(0)
    if dis2 <= cutoff:
        neighbor_type.append(1)
    else:
        neighbor_type.append(0)

    chain_distance = [dis1, dis2]

    helix1 = np.asarray([rC4s_chain1_nt3, rC4s_chain1_nt4, rC4s_chain2_nt1, rC4s_chain2_nt2])
    helix2 = np.asarray([rC4s_chain2_nt3, rC4s_chain2_nt4, rC4s_chain1_nt1, rC4s_chain1_nt2])

    rC4s_in_Ahelix = np.asarray([
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [9.061, -3.241, -8.354], [5.874, -7.623, -10.902]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [5.874, -7.623, -10.902], [0.825, -9.588, -13.450]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [0.825, -9.588, -13.450], [-4.486, -8.514, -15.998]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [-4.486, -8.514, -15.998], [-8.374, -4.741, -18.546]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [-8.374, -4.741, -18.546], [-9.608, 0.534, -21.094]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [-9.608, 0.534, -21.094], [-7.797, 5.640, -23.642]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [-7.797, 5.640, -23.642], [-3.514, 8.959, -26.190]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [-3.514, 8.959, -26.190], [1.883, 9.437, -28.738]],
            [[6.719, 6.889, -3.258], [9.376, 2.167, -5.806], [1.883, 9.437, -28.738], [6.683, 6.924, -31.286]]
            ])    

    best_rmsd1 = 10000.
    for i in range(len(rC4s_in_Ahelix)):
        rmsd1, rot, tran = calc_rmsd(["C4'","C4'","C4'","C4'"], rC4s_in_Ahelix[i], helix1)
        if rmsd1 < best_rmsd1:
            best_rmsd1 = rmsd1

    best_rmsd2 = 10000.
    for i in range(len(rC4s_in_Ahelix)):
        rmsd2, rot, tran = calc_rmsd(["C4'","C4'","C4'","C4'"], rC4s_in_Ahelix[i], helix2)
        if rmsd2 < best_rmsd2:
            best_rmsd2 = rmsd2

    helix_rmsd = [best_rmsd1, best_rmsd2]

    if best_rmsd1 <= 1.5:
        neighbor_type[0] = 1
    if best_rmsd2 <= 1.5:
        neighbor_type[1] = 1

    return neighbor_type, chain_distance, helix_rmsd

def expand_neighbor_chains_for_one_chain(list_chains,dict_next_neighbor,dict_last_neighbor,total_elem,num_native_chain,dict_neighbor_chain_dis,chain_num_before_expanding):
    flatten_chain = []
    for chain in list_chains:
        flatten_chain.extend(chain)

    if len(flatten_chain) == total_elem and max(list(map(int,flatten_chain))) == total_elem - 1:
        bool_full_chain = True
    else:
        bool_full_chain = False
    if bool_full_chain:
        return [list_chains]

    last_chain = list_chains[-1]

    last_elem = last_chain[-1]
    next_neighbor = dict_next_neighbor[last_elem]

    first_elem = last_chain[0]
    last_neighbor = dict_last_neighbor[first_elem]

    list_expanded_chains = []

    chain_num_threshold = 10000

    if not next_neighbor and not last_neighbor:
        for i in range(total_elem):
            if str(i) not in flatten_chain:
                expanded_chain = list_chains + [[str(i)]]
                if len(expanded_chain) <= num_native_chain + 4:
                    list_expanded_chains.append(expanded_chain)
                return list_expanded_chains

    minimal_neighbor_dis1 = 100000.
    if next_neighbor:
        for next_elem in next_neighbor:
            if next_elem not in flatten_chain:
                if dict_neighbor_chain_dis[(last_chain[-1],next_elem)] < minimal_neighbor_dis1:
                    minimal_neighbor_dis1 = dict_neighbor_chain_dis[(last_chain[-1],next_elem)]

    minimal_neighbor_dis2 = 100000.
    if last_neighbor:
        for last_elem in last_neighbor:
            if last_elem not in flatten_chain:
                if dict_neighbor_chain_dis[(last_elem,last_chain[0])] < minimal_neighbor_dis2:
                    minimal_neighbor_dis2 = dict_neighbor_chain_dis[(last_elem,last_chain[0])]

    if next_neighbor and not last_neighbor:
        for next_elem in next_neighbor:
            if next_elem not in flatten_chain:
                expanded_chain = list_chains[0:-1] + [last_chain + [next_elem]]
                list_expanded_chains.append(expanded_chain)
        if chain_num_before_expanding <= chain_num_threshold and minimal_neighbor_dis1 > 8.:
            for i in range(total_elem):
                if str(i) not in flatten_chain:
                    expanded_chain = list_chains + [[str(i)]]
                    if len(expanded_chain) <= num_native_chain + 4:
                        list_expanded_chains.append(expanded_chain)
                        break

    if last_neighbor and not next_neighbor:
        for last_elem in last_neighbor:
            if last_elem not in flatten_chain:
                expanded_chain = list_chains[0:-1] + [[last_elem] + last_chain]
                list_expanded_chains.append(expanded_chain)
        if chain_num_before_expanding <= chain_num_threshold and minimal_neighbor_dis2 > 8.:
            for i in range(total_elem):
                if str(i) not in flatten_chain:
                    expanded_chain = list_chains + [[str(i)]]
                    if len(expanded_chain) <= num_native_chain + 4:
                        list_expanded_chains.append(expanded_chain)
                        break

    if next_neighbor and last_neighbor:
        for next_elem in next_neighbor:
            for last_elem in last_neighbor:
                if next_elem not in flatten_chain and last_elem not in flatten_chain:
                    if last_elem != next_elem:
                        expanded_chain = list_chains[0:-1] + [[last_elem] + last_chain + [next_elem]]
                        list_expanded_chains.append(expanded_chain)
                        if chain_num_before_expanding <= chain_num_threshold:
                            if minimal_neighbor_dis2 > 10.:
                                expanded_chain = list_chains[0:-1] + [last_chain + [next_elem]]
                                list_expanded_chains.append(expanded_chain)
                            if minimal_neighbor_dis1 > 10.:
                                expanded_chain = list_chains[0:-1] + [[last_elem] + last_chain]
                                list_expanded_chains.append(expanded_chain)
                    else:
                        expanded_chain = list_chains[0:-1] + [[last_elem] + last_chain]
                        list_expanded_chains.append(expanded_chain)
                        expanded_chain = list_chains[0:-1] + [last_chain + [next_elem]]
                        list_expanded_chains.append(expanded_chain)
                elif next_elem in flatten_chain and last_elem not in flatten_chain:
                    expanded_chain = list_chains[0:-1] + [[last_elem] + last_chain]
                    list_expanded_chains.append(expanded_chain)
                elif next_elem not in flatten_chain and last_elem in flatten_chain:
                    expanded_chain = list_chains[0:-1] + [last_chain + [next_elem]]
                    list_expanded_chains.append(expanded_chain)
        if chain_num_before_expanding <= chain_num_threshold and minimal_neighbor_dis1 > 8. and minimal_neighbor_dis2 > 8.:
            for i in range(total_elem):
                if str(i) not in flatten_chain:
                    expanded_chain = list_chains + [[str(i)]]
                    if len(expanded_chain) <= num_native_chain + 4:
                        list_expanded_chains.append(expanded_chain)
                        break

    if len(list_expanded_chains) == 0:
        for i in range(total_elem):
            if str(i) not in flatten_chain:
                expanded_chain = list_chains + [[str(i)]]
                if len(expanded_chain) <= num_native_chain + 4:
                    list_expanded_chains.append(expanded_chain)
                break

    idx_to_remove = []
    for i in range(len(list_expanded_chains)):
        for j in range(i+1,len(list_expanded_chains)):
            if j in idx_to_remove:
                continue
            if list_expanded_chains[i] == list_expanded_chains[j]:
                idx_to_remove.append(j)
    idx_to_remove.sort(reverse=True)
    for idx in idx_to_remove:
        list_expanded_chains.pop(idx)
    return list_expanded_chains

def expand_neighbor_chains_for_all_chains(list_all_chains,dict_next_neighbor,dict_last_neighbor,total_elem,num_native_chain,dict_neighbor_chain_dis):
    list_all_expanded_chains = []
    list_all_bool_full_chain = []
    chain_num_before_expanding = len(list_all_chains)
    for list_chains in list_all_chains:
        list_expanded_chains = expand_neighbor_chains_for_one_chain(list_chains,dict_next_neighbor,dict_last_neighbor,total_elem,num_native_chain,dict_neighbor_chain_dis,chain_num_before_expanding)
        if list_expanded_chains:
            list_all_expanded_chains.extend(list_expanded_chains)

    list_all_expanded_chains_string = [] 
    for list_chains in list_all_expanded_chains:
        list_chains_string = [] 
        for chain in list_chains:
            chain_string = "-".join(chain)
            list_chains_string.append(chain_string)
        expanded_chains_string = ";".join(list_chains_string)
        list_all_expanded_chains_string.append(expanded_chains_string)

    list_all_expanded_chains_string = list(dict.fromkeys(list_all_expanded_chains_string))
    
    list_all_expanded_chains = []
    for expanded_chains_string in list_all_expanded_chains_string:
        list_all_expanded_chains.append(expanded_chains_string.split(";"))
    for i in range(len(list_all_expanded_chains)):
        for j in range(len(list_all_expanded_chains[i])):
            list_all_expanded_chains[i][j] = list_all_expanded_chains[i][j].split("-")

    for list_chains in list_all_expanded_chains:
        flatten_chain = []
        for chain in list_chains:
            flatten_chain.extend(chain)
        if len(flatten_chain) == total_elem and len(set(flatten_chain)) == total_elem:
            bool_full_chain = True
        else:
            bool_full_chain = False
        list_all_bool_full_chain.append(bool_full_chain)
    return list_all_expanded_chains, list_all_bool_full_chain

def expand_neighbor_chains_from_initial_seed(dict_next_neighbor,dict_last_neighbor,total_elem,num_native_chain,dict_neighbor_chain_dis):
    list_all_chains = [[['0']]]
    while True:
        list_all_chains, list_all_bool_full_chain = expand_neighbor_chains_for_all_chains(list_all_chains,dict_next_neighbor,dict_last_neighbor,total_elem,num_native_chain,dict_neighbor_chain_dis)
        if total_elem > 40:
            max_num_chain = 100000
        else:
            max_num_chain = 250000
        if len(list_all_chains) > max_num_chain:
            list_all_chains = list_all_chains[0:max_num_chain]
            list_all_bool_full_chain = list_all_bool_full_chain[0:max_num_chain]

        if False not in list_all_bool_full_chain:
            break
    return list_all_chains

def sort_list_neighbor_chains(list_neighbor_chains,dis_between_short_chains):
    if len(list_neighbor_chains) <= 6:
        list_idx = list(range(len(list_neighbor_chains)))
        permutated_list_idx = list(itertools.permutations(list_idx))
        list_dis = []
        for p_list_idx in permutated_list_idx:
            dis = 0.0
            for i in range(len(p_list_idx)-1):
                chain1 = list_neighbor_chains[p_list_idx[i]][-1]
                chain2 = list_neighbor_chains[p_list_idx[i+1]][0]
                dis += dis_between_short_chains[int(chain1)][int(chain2)]
            list_dis.append(dis)

        idx_ranked_by_dis = sorted(range(len(permutated_list_idx)), key=lambda k:list_dis[k])
        list_short_dis_gap_chains = []
        for i in idx_ranked_by_dis:
            if list_dis[i] - list_dis[idx_ranked_by_dis[0]] <= 25.:
                sorted_list_neighbor_chains = []
                for idx in permutated_list_idx[i]:
                    sorted_list_neighbor_chains.append(list_neighbor_chains[idx])
                list_short_dis_gap_chains.append(sorted_list_neighbor_chains)
            else:
                break
        return list_short_dis_gap_chains
    else:
        sorted_list_idx = [0]
        while True:
            first_neighbor_chains = sorted_list_idx[0]
            last_neighbor_chains = sorted_list_idx[-1]
            best_dis1 = 10000000.
            best_idx1 = None
            best_dis2 = 10000000.
            best_idx2 = None
            for i in range(len(list_neighbor_chains)):
                if i in sorted_list_idx:
                    continue
                id1 = list_neighbor_chains[i][-1]
                id2 = list_neighbor_chains[first_neighbor_chains][0]
                id3 = list_neighbor_chains[last_neighbor_chains][-1]
                id4 = list_neighbor_chains[i][0]
                dis1 = dis_between_short_chains[int(id1)][int(id2)]
                dis2 = dis_between_short_chains[int(id3)][int(id4)]
                if dis1 < best_dis1:
                    best_dis1 = dis1
                    best_idx1 = i
                if dis2 < best_dis2:
                    best_dis2 = dis2
                    best_idx2 = i
            if best_dis1 < best_dis2:
                sorted_list_idx = [best_idx1] + sorted_list_idx
            else:
                sorted_list_idx = sorted_list_idx + [best_idx2]
            if len(sorted_list_idx) == len(list_neighbor_chains):
                break
        sorted_list_neighbor_chains = []
        for idx in sorted_list_idx:
            sorted_list_neighbor_chains.append(list_neighbor_chains[idx])
        return [sorted_list_neighbor_chains]

def thread_one_set_of_short_chains_into_long_chains(short_chains,list_nt,num_native_chain,split_chains_idx,adjust_cutoff=False):
    dict_next_neighbor = {}
    dict_next_neighbor_dis = {}
    dict_next_neighbor_helix_rmsd = {}

    dict_last_neighbor = {}

    dict_neighbor_chain_dis = {}

    for i in range(len(short_chains)):
        dict_next_neighbor[str(i)] = []
        dict_next_neighbor_dis[str(i)] = []
        dict_next_neighbor_helix_rmsd[str(i)] = []
        dict_last_neighbor[str(i)] = []

    dis_between_short_chains = np.zeros((len(short_chains),len(short_chains)))

    list_short_chain_idx_not_filtered = []

    for i in range(len(short_chains)):
        for j in range(i+1,len(short_chains)):
            neighbor_type, chain_dis, helix_rmsd = determine_if_two_short_chains_are_neighbors(short_chains[i],short_chains[j],list_nt,cutoff=20.0)
            dis_between_short_chains[i][j] = chain_dis[0]
            dis_between_short_chains[j][i] = chain_dis[1]

            if neighbor_type[0] == 1:
                dict_next_neighbor[str(i)].append(str(j))
                dict_next_neighbor_dis[str(i)].append(chain_dis[0])
                dict_next_neighbor_helix_rmsd[str(i)].append(helix_rmsd[0])
                dict_last_neighbor[str(j)].append(str(i))
                dict_neighbor_chain_dis[(str(i),str(j))] = chain_dis[0]

            if neighbor_type[1] == 1:
                dict_next_neighbor[str(j)].append(str(i))
                dict_next_neighbor_dis[str(j)].append(chain_dis[1])
                dict_next_neighbor_helix_rmsd[str(j)].append(helix_rmsd[1])
                dict_last_neighbor[str(i)].append(str(j))
                dict_neighbor_chain_dis[(str(j),str(i))] = chain_dis[1]

        if adjust_cutoff:
            num_next_neighbor = len(dict_next_neighbor[str(i)])
            num_last_neighbor = len(dict_last_neighbor[str(i)])
            if num_next_neighbor == 0 or num_last_neighbor == 0:
                for j in range(0,len(short_chains)):
                    if j == i:
                        continue
                    neighbor_type, chain_dis, helix_rmsd = determine_if_two_short_chains_are_neighbors(short_chains[i],short_chains[j],list_nt,cutoff=30.0)
                    dis_between_short_chains[i][j] = chain_dis[0]
                    dis_between_short_chains[j][i] = chain_dis[1]

                    if neighbor_type[0] == 1 and num_next_neighbor == 0:
                        dict_next_neighbor[str(i)].append(str(j))
                        dict_next_neighbor_dis[str(i)].append(chain_dis[0])
                        dict_next_neighbor_helix_rmsd[str(i)].append(helix_rmsd[0])
                        dict_last_neighbor[str(j)].append(str(i))
                        dict_neighbor_chain_dis[(str(i),str(j))] = chain_dis[0]
                        if (i,j) not in list_short_chain_idx_not_filtered:
                            list_short_chain_idx_not_filtered.append((i,j))
                        if (j,i) not in list_short_chain_idx_not_filtered:
                            list_short_chain_idx_not_filtered.append((j,i))

                    if neighbor_type[1] == 1 and num_last_neighbor == 0:
                        dict_next_neighbor[str(j)].append(str(i))
                        dict_next_neighbor_dis[str(j)].append(chain_dis[1])
                        dict_next_neighbor_helix_rmsd[str(j)].append(helix_rmsd[1])
                        dict_last_neighbor[str(i)].append(str(j))
                        dict_neighbor_chain_dis[(str(j),str(i))] = chain_dis[1]
                        if (i,j) not in list_short_chain_idx_not_filtered:
                            list_short_chain_idx_not_filtered.append((i,j))
                        if (j,i) not in list_short_chain_idx_not_filtered:
                            list_short_chain_idx_not_filtered.append((j,i))

    if len(short_chains) >= 8: # further filter neighbor chains for too many short chains
        for i in range(len(short_chains)):
            if len(dict_next_neighbor[str(i)]) <= 1:
                continue
            to_remove_idx = []
            min_dis = min(dict_next_neighbor_dis[str(i)])
            for j in range(len(dict_next_neighbor[str(i)])):
                dis_ij = dict_next_neighbor_dis[str(i)][j]
                rmsd_ij = dict_next_neighbor_helix_rmsd[str(i)][j]
                if dis_ij > 10.0 and dis_ij - min_dis > 6.0:
                    if rmsd_ij > 1.5:
                        to_remove_idx.append(j)
            for idx in reversed(to_remove_idx):
                chain_idx = dict_next_neighbor[str(i)][idx]
                if (i,int(chain_idx)) not in list_short_chain_idx_not_filtered or min_dis > 20.0:
                    chain_idx = dict_next_neighbor[str(i)].pop(idx)
                    dict_last_neighbor[chain_idx].remove(str(i))

    count = 1
    for chain_idx in dict_next_neighbor:
        num_next_neighbor = len(dict_next_neighbor[chain_idx])
        if num_next_neighbor > 1:
            count *= num_next_neighbor
        if count > 100000:
            break
    if count > 100000:
        dict_connected_neighbor_chains = {}
        for chain_i in dict_next_neighbor:
            for chain_j in dict_next_neighbor[chain_i]:
                dis_ij = dict_neighbor_chain_dis[(chain_i,chain_j)]
                if dis_ij <= 10.0:
                    if chain_i not in dict_connected_neighbor_chains.keys():
                        dict_connected_neighbor_chains[chain_i] = [chain_j]
                    else:
                        dict_connected_neighbor_chains[chain_i].append(chain_j)

        list_neighbor_chains_to_remove = []
        for chain_i in dict_connected_neighbor_chains:
            for chain_j in dict_connected_neighbor_chains[chain_i]:
                if int(chain_i) in split_chains_idx and int(chain_j) in split_chains_idx:
                    continue
                for chain_k in dict_last_neighbor[chain_j]:
                    if chain_k in dict_connected_neighbor_chains.keys() and chain_j in dict_connected_neighbor_chains[chain_k]:
                        continue
                    else:
                        if (chain_k,chain_j) not in list_neighbor_chains_to_remove:
                            list_neighbor_chains_to_remove.append((chain_k,chain_j))

        for nc in list_neighbor_chains_to_remove:
            chain_i, chain_j = nc
            dict_next_neighbor[chain_i].remove(chain_j)
            dict_last_neighbor[chain_j].remove(chain_i)

    total_elem = len(short_chains)

    list_all_neighbor_chains = expand_neighbor_chains_from_initial_seed(dict_next_neighbor,dict_last_neighbor,total_elem,num_native_chain,dict_neighbor_chain_dis)

    list_long_chains = []
    list_long_chain_num = []
    for num, list_neighbor_chains in enumerate(list_all_neighbor_chains):
        if len(list_neighbor_chains) <= 3:
            permutated_neighbor_chains = list(itertools.permutations(list_neighbor_chains))
        else:
            permutated_neighbor_chains = sort_list_neighbor_chains(list_neighbor_chains,dis_between_short_chains)

        for neighbor_chains in permutated_neighbor_chains:
            list_nts_idx = []
            for neighbor_chain in neighbor_chains:
                for chain_idx in neighbor_chain:
                    chain_idx = int(chain_idx)
                    for nt_idx in short_chains[chain_idx]:
                        list_nts_idx.append(nt_idx)
                    list_nts_idx.append(None)
            list_nts_idx.pop()
            if list_nts_idx not in list_long_chains:
                list_long_chains.append(list_nts_idx)
                list_long_chain_num.append(len(neighbor_chains))
            else:
                idx_for_same_long_chains = list_long_chains.index(list_nts_idx)
                long_chain_num = min(list_long_chain_num[idx_for_same_long_chains], len(neighbor_chains))
                list_long_chain_num[idx_for_same_long_chains] = long_chain_num
    return list_long_chains, list_long_chain_num

def sort_list_nt(list_native_seq, list_nt):
    list_neighbors = []
    for i in range(len(list_nt)):
        nt1 = list_nt[i]
        for j in range(i+1,len(list_nt)):
            nt2 = list_nt[j]
            is_neighbor1, dis1 = determine_if_two_nts_are_neighbors(nt1,nt2)
            is_neighbor2, dis2 = determine_if_two_nts_are_neighbors(nt2,nt1)
            if is_neighbor1 and (not is_neighbor2):
                list_neighbors.append([i,j])
            elif is_neighbor2 and (not is_neighbor1):
                list_neighbors.append([j,i])
            elif is_neighbor1 and is_neighbor2:
                if dis1 < dis2:
                    list_neighbors.append([i,j])
                else:
                    list_neighbors.append([j,i])

    list_found_neighbors = [i for neighbor in list_neighbors for i in neighbor]

    list_not_found_neighbors = []
    for i in range(len(list_nt)):
        if i not in list_found_neighbors:
            list_not_found_neighbors.append(i)
   
    list_multiple_neighbors_branch = []
    for i in range(len(list_neighbors)):
        skip = False
        for multiple_neighbors in list_multiple_neighbors_branch:
            if list_neighbors[i] in multiple_neighbors:
                skip = True
                break
        if skip:
            continue
        multiple_neighbors = []
        for j in range(i+1,len(list_neighbors)):
            if list_neighbors[i][0] == list_neighbors[j][0]:
                multiple_neighbors.append(list_neighbors[j])
        if len(multiple_neighbors) > 0:
            multiple_neighbors.append(list_neighbors[i])
            list_multiple_neighbors_branch.append(multiple_neighbors)

    list_multiple_neighbors_branch, list_bad_neighbors_to_remove = filter_all_multiple_neighbors(list_nt,list_multiple_neighbors_branch,"branch")

    for bad_neighbor in list_bad_neighbors_to_remove:
        if bad_neighbor in list_neighbors:
            list_neighbors.remove(bad_neighbor)

    list_multiple_neighbors_join = []
    for i in range(len(list_neighbors)):
        skip = False
        for multiple_neighbors in list_multiple_neighbors_join:
            if list_neighbors[i] in multiple_neighbors:
                skip = True
                break
        if skip:
            continue
        multiple_neighbors = []
        for j in range(i+1,len(list_neighbors)):
            if list_neighbors[i][1] == list_neighbors[j][1]:
                multiple_neighbors.append(list_neighbors[j])
        if len(multiple_neighbors) > 0:
            multiple_neighbors.append(list_neighbors[i])
            list_multiple_neighbors_join.append(multiple_neighbors)

    list_multiple_neighbors_join, list_bad_neighbors_to_remove = filter_all_multiple_neighbors(list_nt,list_multiple_neighbors_join,"join")

    for bad_neighbor in list_bad_neighbors_to_remove:
        if bad_neighbor in list_neighbors:
            list_neighbors.remove(bad_neighbor)

    list_multiple_neighbors_branch = []
    for i in range(len(list_neighbors)):
        skip = False
        for multiple_neighbors in list_multiple_neighbors_branch:
            if list_neighbors[i] in multiple_neighbors:
                skip = True
                break
        if skip:
            continue
        multiple_neighbors = []
        for j in range(i+1,len(list_neighbors)):
            if list_neighbors[i][0] == list_neighbors[j][0]:
                multiple_neighbors.append(list_neighbors[j])
        if len(multiple_neighbors) > 0:
            multiple_neighbors.append(list_neighbors[i])
            list_multiple_neighbors_branch.append(multiple_neighbors)

    list_multiple_neighbors = list_multiple_neighbors_branch + list_multiple_neighbors_join

    list_neighbors_excluding_multiple_neighbors = []
    list_multiple_neighbors_flatten = [neighbor for multiple_neighbors in list_multiple_neighbors for neighbor in multiple_neighbors]
    for neighbor in list_neighbors:
        if neighbor not in list_multiple_neighbors_flatten:
            list_neighbors_excluding_multiple_neighbors.append(neighbor)

    list_multiple_neighbors_combination = [p for p in itertools.product(*list_multiple_neighbors)]
    list_multiple_neighbors_combination = [list(p) for p in list_multiple_neighbors_combination]

    for i in range(len(list_multiple_neighbors_combination)):
        different_elem = []
        for elem in list_multiple_neighbors_combination[i]:
            if elem not in different_elem:
                different_elem.append(elem)
        list_multiple_neighbors_combination[i] = different_elem

    to_remove_idx = []
    for i, combination in enumerate(list_multiple_neighbors_combination):
        list_ib = []
        list_ie = []
        for v in combination:
            list_ib.append(v[0])
            list_ie.append(v[1])
        if len(list_ib) != len(set(list_ib)):
            to_remove_idx.append(i)
            continue
        if len(list_ie) != len(set(list_ie)):
            to_remove_idx.append(i)
            continue
    for i in reversed(to_remove_idx):
        list_multiple_neighbors_combination.pop(i)
    
    list_neighbors_candidates = []
    for multiple_neighbors_combination in list_multiple_neighbors_combination:
        if multiple_neighbors_combination:
            candidate = list_neighbors_excluding_multiple_neighbors + multiple_neighbors_combination
        else:
            candidate = list_neighbors_excluding_multiple_neighbors            
        list_neighbors_candidates.append(candidate)

    return list_neighbors_candidates

def check_short_chains_similarity(list_all_short_chains, short_chains):
    if not list_all_short_chains:
        list_all_short_chains.append(short_chains)
        return list_all_short_chains

    replaced = False
    for idx, chains in enumerate(list_all_short_chains):
        similar1 = True
        similar2 = True
        if len(chains) != len(short_chains):
            similar1 = False
            similar2 = False
        else:
            for i in range(len(chains)):
                chain1 = list(map(str,chains[i]))
                chain1 = "".join(chain1)
                chain2 = list(map(str,short_chains[i]))
                chain2 = "".join(chain2)
                if chain1 != chain2 and chain1 in chain2:
                    similar1 = False
                elif chain1 != chain2 and chain2 in chain1:
                    similar2 = False
                elif chain1 != chain2:
                    similar1 = False
                    similar2 = False
        if similar2:
            list_all_short_chains[idx] = short_chains
            replaced = True
            break
        if similar1:
            replaced = True
            break
    if not replaced:
        list_all_short_chains.append(short_chains)

    return list_all_short_chains

def get_all_short_chains(list_native_seq,list_neighbors_candidates):
    list_all_short_chains = []
    for neighbors_candidate in list_neighbors_candidates:
        short_chains = get_short_chains(list_native_seq, neighbors_candidate)
        list_all_short_chains = check_short_chains_similarity(list_all_short_chains, short_chains)
    return list_all_short_chains

def determine_position_between_point_and_triangle(triangle, point):
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    normal = np.cross(edge1, edge2)
    
    vector_to_point = point - triangle[0]
    dot_product = np.dot(vector_to_point, normal)
    projected_point = point - dot_product * normal
    
    u = np.dot(np.cross(triangle[1] - projected_point, triangle[2] - projected_point), normal) / np.linalg.norm(normal)**2
    v = np.dot(np.cross(triangle[2] - projected_point, triangle[0] - projected_point), normal) / np.linalg.norm(normal)**2
    if u >= 0 and v >= 0 and u + v <= 1:
        point_projected_in_triangle = True
    else:
        point_projected_in_triangle = False

    if dot_product > 0:
        above_triangle = True
    else:
        above_triangle = False
    
    return point_projected_in_triangle, above_triangle

def check_if_curve_passes_through_polygon(polygon_points, curve_points):
    if len(polygon_points) < 3:
        return False

    list_above_triangle = []
    for i in range(1, len(polygon_points)-1):
        triangle = [polygon_points[0], polygon_points[-1], polygon_points[i]]
        for point in curve_points:
            point_projected_in_triangle, above_triangle = determine_position_between_point_and_triangle(triangle, point)
            if above_triangle not in list_above_triangle and point_projected_in_triangle:
                list_above_triangle.append(above_triangle)
            if len(list_above_triangle) == 2:
                return True

    return False

def check_if_loop_passes_through_hp(list_nt, array_c4s_c4s_dis, short_chains, chain1_idx, chain2_idx, hp_loop):
    nearby_short_chains_and_nts = {}
    hp_loop_breakpoint = hp_loop[1][0]
    for i in range(len(array_c4s_c4s_dis[hp_loop_breakpoint])):
        if i == hp_loop_breakpoint:
            continue
        if array_c4s_c4s_dis[hp_loop_breakpoint][i] < 20.:
            for chain_idx, short_chain in enumerate(short_chains):
                if chain_idx == chain1_idx or chain_idx == chain2_idx:
                    continue
                if i in short_chain:
                    if chain_idx not in nearby_short_chains_and_nts:
                        nearby_short_chains_and_nts[chain_idx] = [short_chain.index(i)]
                    else:
                        nearby_short_chains_and_nts[chain_idx].append(short_chain.index(i))

    for chain_idx in nearby_short_chains_and_nts:
        nearby_short_chains_and_nts[chain_idx].sort()
        if len(nearby_short_chains_and_nts[chain_idx])%2 == 0:
            mid_idx = int(len(nearby_short_chains_and_nts[chain_idx])/2)
        else:
            mid_idx = int((len(nearby_short_chains_and_nts[chain_idx])-1)/2)
        to_del = []
        for i in range(len(nearby_short_chains_and_nts[chain_idx])):
            if i == mid_idx:
                continue
            if abs(nearby_short_chains_and_nts[chain_idx][i] - nearby_short_chains_and_nts[chain_idx][mid_idx]) < 5:
                to_del.append(i)
        for i in sorted(to_del,reverse=True):
            nearby_short_chains_and_nts[chain_idx].pop(i)

    hp_loop_polygon = []
    for loop in hp_loop:
        for nt_idx in loop:
            hp_loop_polygon.append(list_nt[nt_idx][2]["C4'"])

    list_nt_idx1 = []
    list_nt_idx2 = []
    for nt_idx in hp_loop[0]:
        list_nt_idx1.append(str(nt_idx))
    for nt_idx in hp_loop[1]:
        list_nt_idx2.append(str(nt_idx))
    list_nt_idx1 = "+".join(list_nt_idx1)
    list_nt_idx2 = "+".join(list_nt_idx2)

    for chain_idx in nearby_short_chains_and_nts:
        for mid_nt_idx_in_loop in nearby_short_chains_and_nts[chain_idx]:
            curve_points = []
            starting_nt_idx_in_loop = max(0,mid_nt_idx_in_loop-5)
            ending_nt_idx_in_loop = min(mid_nt_idx_in_loop+5,len(short_chains[chain_idx])-1) + 1
            for nt_idx_in_loop in range(starting_nt_idx_in_loop, ending_nt_idx_in_loop):
                nt_idx = short_chains[chain_idx][nt_idx_in_loop]
                curve_points.append(list_nt[nt_idx][2]["C4'"])
            if check_if_curve_passes_through_polygon(hp_loop_polygon, curve_points):
                return True

    return False
    
def determine_if_forming_hp_loop_for_two_loops(list_nt, array_c4s_c4s_dis, short_chains, chain1_idx, loop1, chain2_idx, loop2, head_tail):
    sugar_coord_in_helix =  np.asarray([[6.72, 6.89, -3.26], [6.78, 5.64, -4.14], [7.57, 4.72, -3.21], [6.87, 5.00, -1.88], [6.49, 6.36, -1.92],
        [9.38, 2.17, -5.81], [8.75, 1.09, -6.69], [8.92, -0.11, -5.76], [8.49, 0.49, -4.43], [8.90, 1.85, -4.47],
        [1.93, -9.43, 0.71], [2.65, -8.41, 1.59], [3.82, -8.06, 0.66], [3.08, -7.92, -0.67], [2.02, -8.86, -0.63],
        [6.72, -6.89, 3.26], [6.78, -5.64, 4.14], [7.57, -4.72, 3.21], [6.87, -5.00, 1.88], [6.49, -6.36, 1.92]])

    if head_tail == "head":
        ib = 0
        ie = min(12, len(loop1)-1)
        jb = max(0, len(loop2) - 13)
        je = len(loop2) - 1
        for i in range(ib,ie):
            atoms1 = list_nt[loop1[i]][2]
            atoms2 = list_nt[loop1[i+1]][2]
            for j in range(je-1,jb-1,-1):
                atoms3 = list_nt[loop2[j]][2]
                atoms4 = list_nt[loop2[j+1]][2]
                dis_c4s_c4s = np.linalg.norm(atoms1["C4'"] - atoms4["C4'"])
                if dis_c4s_c4s > 19.0 or dis_c4s_c4s < 11.0:
                    continue
                sugar_coord = []
                for atoms in [atoms1, atoms2, atoms3, atoms4]:
                    for name in ["C4'", "C3'", "C2'", "C1'", "O4'"]:
                        sugar_coord.append(atoms[name])
                sugar_coord = np.asarray(sugar_coord)
                rmsd, rot, tran = calc_rmsd([], sugar_coord, sugar_coord_in_helix)
                if rmsd <= 1.5:
                    hp_loop = [loop2[j+1:],loop1[0:(i+1)]]
                    if not check_if_loop_passes_through_hp(list_nt, array_c4s_c4s_dis, short_chains, chain1_idx, chain2_idx, hp_loop):
                        return True
    else:
        ib = max(0, len(loop1) - 13)
        ie = len(loop1) - 1
        jb = 0
        je = min(12, len(loop2)-1)
        for i in range(ie-1,ib-1,-1):
            atoms1 = list_nt[loop1[i]][2]
            atoms2 = list_nt[loop1[i+1]][2]
            for j in range(jb,je):
                atoms3 = list_nt[loop2[j]][2]
                atoms4 = list_nt[loop2[j+1]][2]
                dis_c4s_c4s = np.linalg.norm(atoms1["C4'"] - atoms4["C4'"])
                if dis_c4s_c4s > 19.0 or dis_c4s_c4s < 11.0:
                    continue
                sugar_coord = []
                for atoms in [atoms1, atoms2, atoms3, atoms4]:
                    for name in ["C4'", "C3'", "C2'", "C1'", "O4'"]:
                        sugar_coord.append(atoms[name])
                sugar_coord = np.asarray(sugar_coord)
                rmsd, rot, tran = calc_rmsd([], sugar_coord, sugar_coord_in_helix)
                if rmsd <= 1.5:
                    hp_loop = [loop1[i+1:],loop2[0:(j+1)]]
                    if not check_if_loop_passes_through_hp(list_nt, array_c4s_c4s_dis, short_chains, chain1_idx, chain2_idx, hp_loop):
                        return True

    return False

def split_short_chains(short_chains, list_nt):
    array_c4s_c4s_dis = np.zeros((len(list_nt), len(list_nt)))
    array_p_p_dis = np.zeros((len(list_nt), len(list_nt))) - 1.
    for i in range(len(list_nt)):
        r_c4s1 = list_nt[i][2]["C4'"]
        if "P" in list_nt[i][2]:
            r_p1 = list_nt[i][2]["P"]
        else:
            r_p1 = None
        for j in range(i+1,len(list_nt)):
            r_c4s2 = list_nt[j][2]["C4'"]
            dis_c4s_c4s = np.linalg.norm(r_c4s1 - r_c4s2)
            array_c4s_c4s_dis[i][j] = dis_c4s_c4s
            array_c4s_c4s_dis[j][i] = dis_c4s_c4s
            if "P" in list_nt[j][2]:
                r_p2 = list_nt[j][2]["P"]
            else:
                r_p2 = None
            if r_p1 is not None and r_p2 is not None:
                dis_p_p = np.linalg.norm(r_p1 - r_p2)
                array_p_p_dis[i][j] = dis_p_p
                array_p_p_dis[j][i] = dis_p_p

    dict_split_chain_info = {}
    list_nt_idx_to_remove = []
    min_size = 3

    #-------------------- split chains involving large P-P distance in a circular chain-------------
    for i in range(len(short_chains)):
        head_nt_idx = short_chains[i][0]
        tail_nt_idx = short_chains[i][-1]
        if array_c4s_c4s_dis[head_nt_idx][tail_nt_idx] > 8.0:
            continue
        for j in range(min_size-1, len(short_chains[i])-min_size):
            nt1_idx = short_chains[i][j]
            nt2_idx = short_chains[i][j+1]
            if array_p_p_dis[nt1_idx][nt2_idx] > 7.:
                atoms1 = list_nt[nt1_idx][2]
                atoms2 = list_nt[nt2_idx][2]
                rmsd = calc_rmsd_between_stacked_sugars(atoms1, atoms2)
                if rmsd < 1.5:
                    if i not in dict_split_chain_info:
                        dict_split_chain_info[i] = [0, j+1, len(short_chains[i])]
                    else:
                        if j+1 not in dict_split_chain_info[i]:
                            dict_split_chain_info[i].append(j+1)

    #-------------------- split chains involving hairpin--------------------------------------------
    for i in range(len(short_chains)):
        head_nt_idx_i = short_chains[i][0]
        tail_nt_idx_i = short_chains[i][-1]
        head2_nt_idx_i = short_chains[i][1]
        tail2_nt_idx_i = short_chains[i][-2]
        for j in range(len(short_chains)):
            if i == j:
                continue
            head_nt_idx_j = short_chains[j][0]
            tail_nt_idx_j = short_chains[j][-1]
            if array_c4s_c4s_dis[head_nt_idx_i][tail_nt_idx_j] <= 15.0:
                continue
            if array_c4s_c4s_dis[head_nt_idx_j][tail_nt_idx_i] <= 15.0:
                continue

            dis1_ij = 1000.
            potential_split_chain_info1 = []
            for k in range(1,len(short_chains[j])-1):
                nt_idx = short_chains[j][k]
                if array_c4s_c4s_dis[head_nt_idx_i][nt_idx] <= 10.0:
                    if k+1 >= min_size and len(short_chains[j]) - k - 1 >= min_size:
                        if determine_if_forming_hp_loop_for_two_loops(list_nt, array_c4s_c4s_dis, short_chains, i, short_chains[i], j, short_chains[j][0:(k+1)], "head"):
                            dis1_ij = array_c4s_c4s_dis[head_nt_idx_i][nt_idx]
                            if j not in dict_split_chain_info.keys():
                                potential_split_chain_info1 = [0, k+1, len(short_chains[j])]
                            else:
                                potential_split_chain_info1 = [k+1]
                            break
                elif array_c4s_c4s_dis[tail_nt_idx_i][nt_idx] <= 10.0:
                    if k >= min_size and len(short_chains[j]) - k >= min_size:
                        if determine_if_forming_hp_loop_for_two_loops(list_nt, array_c4s_c4s_dis, short_chains, i, short_chains[i], j, short_chains[j][k:], "tail"):
                            dis1_ij = array_c4s_c4s_dis[tail_nt_idx_i][nt_idx]
                            if j not in dict_split_chain_info.keys():
                                potential_split_chain_info1 = [0, k, len(short_chains[j])]
                            else:
                                potential_split_chain_info1 = [k]
                            break

            dis2_ij = 1000.
            potential_split_chain_info2 = []
            for k in range(1,len(short_chains[j])-1):
                nt_idx = short_chains[j][k]
                if array_c4s_c4s_dis[head2_nt_idx_i][nt_idx] <= 10.0:
                    if k+1 >= min_size and len(short_chains[j]) - k - 1 >= min_size:
                        if determine_if_forming_hp_loop_for_two_loops(list_nt, array_c4s_c4s_dis, short_chains, i, short_chains[i][1:], j, short_chains[j][0:(k+1)], "head"):
                            dis2_ij = array_c4s_c4s_dis[head2_nt_idx_i][nt_idx]
                            if j not in dict_split_chain_info.keys():
                                potential_split_chain_info2 = [0, k+1, len(short_chains[j])]
                            else:
                                potential_split_chain_info2 = [k+1]
                            potential_nt_idx_to_remove = head_nt_idx_i
                            break

                elif array_c4s_c4s_dis[tail2_nt_idx_i][nt_idx] <= 10.0:
                    if k >= min_size and len(short_chains[j]) - k >= min_size:
                        if determine_if_forming_hp_loop_for_two_loops(list_nt, array_c4s_c4s_dis, short_chains, i, short_chains[i][0:-1], j, short_chains[j][k:], "tail"):
                            dis2_ij = array_c4s_c4s_dis[tail2_nt_idx_i][nt_idx]
                            if j not in dict_split_chain_info.keys():
                                potential_split_chain_info2 = [0, k, len(short_chains[j])]
                            else:
                                potential_split_chain_info2 = [k]
                            potential_nt_idx_to_remove = tail_nt_idx_i
                            break

            if dis1_ij < dis2_ij and dis1_ij < 1000.:
                if len(potential_split_chain_info1) == 1 and potential_split_chain_info1[0] not in dict_split_chain_info[j]:
                    dict_split_chain_info[j].extend(potential_split_chain_info1)
                else:
                    dict_split_chain_info[j] = potential_split_chain_info1
            elif dis2_ij < dis1_ij and dis2_ij < 1000.:
                if len(potential_split_chain_info2) == 1 and potential_split_chain_info2[0] not in dict_split_chain_info[j]:
                    dict_split_chain_info[j].extend(potential_split_chain_info2)
                else:
                    dict_split_chain_info[j] = potential_split_chain_info2
                if potential_nt_idx_to_remove not in list_nt_idx_to_remove:
                    list_nt_idx_to_remove.append(potential_nt_idx_to_remove)
   
    if not dict_split_chain_info:
        return short_chains, []
    else:
        new_short_chains = []
        new_split_chains_idx = []
        for i in range(len(short_chains)):
            if i not in dict_split_chain_info.keys():
                new_short_chains.append(short_chains[i])
            else:
                split_chain_index = dict_split_chain_info[i]
                split_chain_index.sort()
                for j in range(len(split_chain_index)-1):
                    kb = split_chain_index[j]
                    ke = split_chain_index[j+1]
                    if ke - kb <= 1:
                        if kb-1 >=0 and short_chains[i][kb-1] in list_nt_idx_to_remove:
                            list_nt_idx_to_remove.remove(short_chains[i][kb-1])
                        if ke < len(short_chains[i]) and short_chains[i][ke] in list_nt_idx_to_remove:
                            list_nt_idx_to_remove.remove(short_chains[i][ke])
                        continue
                    new_short_chains.append(short_chains[i][kb:ke])
                    new_split_chains_idx.append(len(new_short_chains)-1)
        for nt_idx_to_remove in list_nt_idx_to_remove:
            for i in range(len(new_short_chains)):
                if nt_idx_to_remove in new_short_chains[i]:
                    if len(new_short_chains[i]) > 2:
                        new_short_chains[i].remove(nt_idx_to_remove)
        return new_short_chains, new_split_chains_idx

def split_all_short_chains(list_all_short_chains, list_nt):
    new_list_all_short_chains = []
    list_split_chains_idx = []
    for short_chains in list_all_short_chains:
        new_short_chains, split_chains_idx = split_short_chains(short_chains, list_nt)
        new_list_all_short_chains.append(new_short_chains)
        list_split_chains_idx.append(split_chains_idx)
    return new_list_all_short_chains, list_split_chains_idx

def thread_all_short_chains_into_long_chains(list_nt,list_all_short_chains,num_native_chain,list_split_chains_idx):
    assert len(list_all_short_chains) == len(list_split_chains_idx)

    list_all_long_chains = []
    list_all_long_chain_num = []
    for short_chains, split_chains_idx in zip(list_all_short_chains, list_split_chains_idx):
        list_long_chains, list_long_chain_num = thread_one_set_of_short_chains_into_long_chains(short_chains,list_nt,num_native_chain,split_chains_idx,adjust_cutoff=False)
        list_all_long_chains.extend(list_long_chains)
        list_all_long_chain_num.extend(list_long_chain_num)
        if len(list_all_short_chains) <= 5:
            list_long_chains2, list_long_chain_num2 = thread_one_set_of_short_chains_into_long_chains(short_chains,list_nt,num_native_chain,split_chains_idx,adjust_cutoff=True)
            list_idx_to_remove = []
            for i in range(len(list_long_chains2)):
                if list_long_chains2[i] in list_long_chains:
                    list_idx_to_remove.append(i)
            for i in reversed(list_idx_to_remove):
                list_long_chains2.pop(i)
                list_long_chain_num2.pop(i)
            list_all_long_chains.extend(list_long_chains2)
            list_all_long_chain_num.extend(list_long_chain_num2)
    return list_all_long_chains, list_all_long_chain_num

def get_rotran(reference_coords, coords):
    n = reference_coords.shape[0]
    av1 = sum(coords) / n
    av2 = sum(reference_coords) / n
    coords = coords - av1
    reference_coords = reference_coords - av2
    a = np.dot(np.transpose(coords), reference_coords)
    u, d, vt = np.linalg.svd(a)
    rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))
    if np.linalg.det(rot) < 0:
        vt[2] = -vt[2]
        rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))
    tran = av2 - np.dot(av1, rot)
    return rot, tran

def calc_rmsd(ref_atom_names, reference_coords, coords):
    assert reference_coords.shape == coords.shape, f"Different shapes in calc_rmsd() {reference_coords.shape} {coords.shape}"
    rot, tran = get_rotran(reference_coords, coords)
    transformed_coords = np.dot(coords, rot) + tran

    diff = transformed_coords - reference_coords

    for i, atom_name in enumerate(ref_atom_names):
        if atom_name in ["P","O3'"]:
            diff[i] = diff[i] * 5.0
    
    rmsd = np.sqrt(sum(sum(diff * diff)) / reference_coords.shape[0])
    return rmsd, rot, tran

def get_transformed(rot, tran, coords):
    transformed_coords = np.dot(coords, rot) + tran
    return transformed_coords

def rebuild_base(dict_base_templates,nt_name,ref_atom_names,ref_atom_coords):
    rebuilt_base_atoms = {}
    coords = np.asarray([dict_base_templates[nt_name+"base"][ref_atom_names[0]], dict_base_templates[nt_name+"base"][ref_atom_names[1]],dict_base_templates[nt_name+"base"][ref_atom_names[2]]])
    rot, tran = get_rotran(ref_atom_coords, coords)
    for atom_name in dict_base_templates[nt_name+"base"]:
        vec = dict_base_templates[nt_name+"base"][atom_name]
        transformed_vec = get_transformed(rot,tran,vec)
        rebuilt_base_atoms[atom_name] = transformed_vec
    return rebuilt_base_atoms

def rebuild_atom(list_templates,ref_atom_names,ref_atom_coords,rebuilt_atom_names):
    best_rmsd = 10000.
    for tp in list_templates:
        coords = np.asarray([tp[name] for name in ref_atom_names])
        rmsd, rot, tran = calc_rmsd(ref_atom_names, ref_atom_coords, coords)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_rot = rot
            best_tran = tran
            best_template = tp
    rebuilt_atom_coords = []
    for atom_name in rebuilt_atom_names:
        vec = best_template[atom_name]
        rebuilt_atom_coords.append(get_transformed(best_rot, best_tran, vec))
    return rebuilt_atom_coords

def calc_rmsd_between_paired_sugars(atoms1, atoms2):
    base_paired_sugar_coord = np.asarray([[6.719, 6.889, -3.258], [6.776, 5.642, -4.140], [7.567, 4.725, -3.208], [6.874, 4.999, -1.877], [6.486, 6.364, -1.919],
            [6.719, -6.889, 3.258], [6.776, -5.642, 4.140], [7.567, -4.725, 3.208], [6.874, -4.999, 1.877], [6.486, -6.364, 1.919]])

    sugar_coord = np.asarray([atoms1["C4'"], atoms1["C3'"], atoms1["C2'"], atoms1["C1'"], atoms1["O4'"], 
        atoms2["C4'"], atoms2["C3'"], atoms2["C2'"], atoms2["C1'"], atoms2["O4'"]])
    
    rmsd, rot, tran = calc_rmsd([], sugar_coord, base_paired_sugar_coord)
    return rmsd

def calc_rmsd_between_stacked_sugars(atoms1, atoms2):
    base_paired_sugar_coord = np.asarray([[6.72, 6.89, -3.26], [6.78, 5.64, -4.14], [7.57, 4.72, -3.21], [6.87, 5.00, -1.88], [6.49, 6.36, -1.92],
        [9.38, 2.17, -5.81], [8.75, 1.09, -6.69], [8.92, -0.11, -5.76], [8.49, 0.49, -4.43], [8.90, 1.85, -4.47]])

    sugar_coord = np.asarray([atoms1["C4'"], atoms1["C3'"], atoms1["C2'"], atoms1["C1'"], atoms1["O4'"], 
        atoms2["C4'"], atoms2["C3'"], atoms2["C2'"], atoms2["C1'"], atoms2["O4'"]])

    rmsd, rot, tran = calc_rmsd([], sugar_coord, base_paired_sugar_coord)
    return rmsd

def calc_2D_score(nt_chains_raw):
    sugar_atom_names = ["C4'","C3'","C2'","C1'","O4'"]

    nt_chains = copy.deepcopy(nt_chains_raw)

    score = 0
    for i in range(len(nt_chains)):
        nt1 = nt_chains[i]
        if len(nt1) == 4:
            nt_idx1, nt_name1, atoms1, chain_idx1 = nt1
        else:
            raise ValueError(f"wrong nt")
        for j in range(i+1,len(nt_chains)):
            nt2 = nt_chains[j]
            if len(nt2) == 4:
                nt_idx2, nt_name2, atoms2, chain_idx2 = nt2
            else:
                raise ValueError(f"wrong nt")

            dis_c4s_c4s = np.linalg.norm(atoms1["C4'"] - atoms2["C4'"])
            if dis_c4s_c4s > 18.0:
                continue

            if nt_name1+nt_name2 not in ["AU","UA","CG","GC","GU","UG"]:
                continue

            rmsd = calc_rmsd_between_paired_sugars(atoms1, atoms2)
            if rmsd <= 0.8:
                score += 1
    return score

def write_PDB_short_chains(DEEPCRYORNA_HOME,map_origin,short_chains_raw,outpdbfile,list_nt_raw):
    list_backbone_templates, list_sugar_base_AG_templates, list_sugar_base_CU_templates, dict_base_templates = get_template_nucleotide(DEEPCRYORNA_HOME)
    backbone_ref_atom_names = ["P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'"]
    sugar_ref_atom_names = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'"]

    short_chains = copy.deepcopy(short_chains_raw)
    list_nt = copy.deepcopy(list_nt_raw)

    f = open(outpdbfile,"w")
    atom_id = 1
    for num_chain, short_chain in enumerate(short_chains):
        for nt_idx, idx in enumerate(short_chain):
            _, nttype, atoms = list_nt[idx]
            if num_chain >= 62:
                num_chain -= 62

            if num_chain <= 25:
                chain_name = chr(ord("A")+num_chain)
            elif num_chain <= 51:
                chain_name = chr(ord("a")+num_chain-26)
            elif num_chain <= 61:
                chain_name = chr(ord("0")+num_chain-52)

            if nttype == 1:
                nt_name = "A"
            elif nttype == 2:
                nt_name = "C"
            else:
                nt_name = "U"

            if nt_name in ["A","G"]:
                base_ref_atom_names = ["N9","C4","C8"]
                if "N1" in atoms.keys() and "N9" not in atoms.keys():
                    atoms["N9"] = atoms["N1"]
                    atoms.pop("N1")
                if "C2" in atoms.keys() and "C4" not in atoms.keys():
                    atoms["C4"] = atoms["C2"]
                    atoms.pop("C2")
                if "C6" in atoms.keys() and "C8" not in atoms.keys():
                    atoms["C8"] = atoms["C6"]
                    atoms.pop("C6")
            elif nt_name in ["C","U"]:
                base_ref_atom_names = ["N1","C2","C6"]
                if "N9" in atoms.keys() and "N1" not in atoms.keys():
                    atoms["N1"] = atoms["N9"]
                    atoms.pop("N9")
                if "C4" in atoms.keys() and "C2" not in atoms.keys():
                    atoms["C2"] = atoms["C4"]
                    atoms.pop("C4")
                if "C8" in atoms.keys() and "C6" not in atoms.keys():
                    atoms["C6"] = atoms["C8"]
                    atoms.pop("C8")
            else:
                raise ValueError(f"Invalid nucleotide name: {nt_name}")
       
            bb_ref_atom_names = []
            bb_rebuilt_atom_names = []
            for atom_name in backbone_ref_atom_names:
                bb_rebuilt_atom_names.append(atom_name)
                if atom_name in atoms.keys():
                    bb_ref_atom_names.append(atom_name)
            if len(bb_rebuilt_atom_names) > 0:
                bb_ref_atom_coords = [atoms[atom_name] for atom_name in bb_ref_atom_names]
                bb_ref_atom_coords = np.asarray(bb_ref_atom_coords)
                bb_rebuilt_atom_coords = rebuild_atom(list_backbone_templates,bb_ref_atom_names,bb_ref_atom_coords,bb_rebuilt_atom_names)
                for tag, atom_name in enumerate(bb_rebuilt_atom_names):
                    atoms[atom_name] = bb_rebuilt_atom_coords[tag]
           
            if True:
                sugar_ref_atom_coords = [atoms[atom_name] for atom_name in sugar_ref_atom_names]
                sugar_ref_atom_coords = np.asarray(sugar_ref_atom_coords)
                if nt_name in ["A","G"]:
                    rebuilt_atom_coords = rebuild_atom(list_sugar_base_AG_templates,sugar_ref_atom_names,sugar_ref_atom_coords,base_ref_atom_names)
                else:
                    rebuilt_atom_coords = rebuild_atom(list_sugar_base_CU_templates,sugar_ref_atom_names,sugar_ref_atom_coords,base_ref_atom_names)
                for tag, atom_name in enumerate(base_ref_atom_names):
                    atoms[atom_name] = rebuilt_atom_coords[tag]

            base_ref_atom_coords = []
            for name in base_ref_atom_names:
                if name in atoms.keys():
                    base_ref_atom_coords.append(atoms[name])
            if len(base_ref_atom_coords) == 3:
                base_ref_atom_coords = np.asarray(base_ref_atom_coords)
                full_base = rebuild_base(dict_base_templates,nt_name,base_ref_atom_names,base_ref_atom_coords)
                for atom_name in full_base:
                    atoms[atom_name] = full_base[atom_name]
    
            for atom_name in atoms:
                coord = atoms[atom_name]
                coord += map_origin
                coord[0] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) 
                coord[1] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) 
                coord[2] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) # alter the same atoms by adding a small shift
                coordx = f"{coord[0]:.3f}"
                if len(coordx) > 8:
                    coordx = coordx[:8]
                coordy = f"{coord[1]:.3f}"
                if len(coordy) > 8:
                    coordy = coordy[:8]
                coordz = f"{coord[2]:.3f}"
                if len(coordz) > 8:
                    coordz = coordz[:8]
                f.write(f"ATOM  {atom_id:>5}  {atom_name:<3} {nt_name:>3} {chain_name}{nt_idx:>4}    {coordx:>8}{coordy:>8}{coordz:>8}\n")
                atom_id += 1
    f.write("END")
    f.close()

def write_PDB_all_nts(DEEPCRYORNA_HOME,map_origin,nt_chains_raw,outpdbfile):
    list_backbone_templates, list_sugar_base_AG_templates, list_sugar_base_CU_templates, dict_base_templates = get_template_nucleotide(DEEPCRYORNA_HOME)
    backbone_ref_atom_names = ["P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'"]
    sugar_ref_atom_names = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'"]

    nt_chains = copy.deepcopy(nt_chains_raw)

    f = open(outpdbfile,"w")
    atom_id = 1
    for nt_num, nt in enumerate(nt_chains): 
        if len(nt) == 3:
            nt_idx, nttype, atoms = nt
        else:
            raise ValueError(f"wrong nt")
        chain_name = "A"
        if nttype == 1:
            nt_name = "A"
        elif nttype == 2:
            nt_name = "C"
        else:
            nt_name = "U"

        if nt_name in ["A","G"]:
            base_ref_atom_names = ["N9","C4","C8"]
            if "N1" in atoms.keys() and "N9" not in atoms.keys():
                atoms["N9"] = atoms["N1"]
                atoms.pop("N1")
            if "C2" in atoms.keys() and "C4" not in atoms.keys():
                atoms["C4"] = atoms["C2"]
                atoms.pop("C2")
            if "C6" in atoms.keys() and "C8" not in atoms.keys():
                atoms["C8"] = atoms["C6"]
                atoms.pop("C6")
        elif nt_name in ["C","U"]:
            base_ref_atom_names = ["N1","C2","C6"]
            if "N9" in atoms.keys() and "N1" not in atoms.keys():
                atoms["N1"] = atoms["N9"]
                atoms.pop("N9")
            if "C4" in atoms.keys() and "C2" not in atoms.keys():
                atoms["C2"] = atoms["C4"]
                atoms.pop("C4")
            if "C8" in atoms.keys() and "C6" not in atoms.keys():
                atoms["C6"] = atoms["C8"]
                atoms.pop("C8")
        else:
            raise ValueError(f"Invalid nucleotide name: {nt_name}")
       
        bb_ref_atom_names = []
        bb_rebuilt_atom_names = []
        for atom_name in backbone_ref_atom_names:
            bb_rebuilt_atom_names.append(atom_name)
            if atom_name in atoms.keys():
                bb_ref_atom_names.append(atom_name)
        if len(bb_rebuilt_atom_names) > 0:
            bb_ref_atom_coords = [atoms[atom_name] for atom_name in bb_ref_atom_names]
            bb_ref_atom_coords = np.asarray(bb_ref_atom_coords)
            bb_rebuilt_atom_coords = rebuild_atom(list_backbone_templates,bb_ref_atom_names,bb_ref_atom_coords,bb_rebuilt_atom_names)
            for tag, atom_name in enumerate(bb_rebuilt_atom_names):
                atoms[atom_name] = bb_rebuilt_atom_coords[tag]
           
        if True:
            sugar_ref_atom_coords = [atoms[atom_name] for atom_name in sugar_ref_atom_names]
            sugar_ref_atom_coords = np.asarray(sugar_ref_atom_coords)
            if nt_name in ["A","G"]:
                rebuilt_atom_coords = rebuild_atom(list_sugar_base_AG_templates,sugar_ref_atom_names,sugar_ref_atom_coords,base_ref_atom_names)
            else:
                rebuilt_atom_coords = rebuild_atom(list_sugar_base_CU_templates,sugar_ref_atom_names,sugar_ref_atom_coords,base_ref_atom_names)
            for tag, atom_name in enumerate(base_ref_atom_names):
                atoms[atom_name] = rebuilt_atom_coords[tag]

        base_ref_atom_coords = []
        for name in base_ref_atom_names:
            if name in atoms.keys():
                base_ref_atom_coords.append(atoms[name])
        if len(base_ref_atom_coords) == 3:
            base_ref_atom_coords = np.asarray(base_ref_atom_coords)
            full_base = rebuild_base(dict_base_templates,nt_name,base_ref_atom_names,base_ref_atom_coords)
            for atom_name in full_base:
                atoms[atom_name] = full_base[atom_name]

        for atom_name in atoms:
            coord = atoms[atom_name]
            coord += map_origin
            coord[0] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) 
            coord[1] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) 
            coord[2] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) # alter the same atoms by adding a small shift
            coordx = f"{coord[0]:.3f}"
            if len(coordx) > 8:
                coordx = coordx[:8]
            coordy = f"{coord[1]:.3f}"
            if len(coordy) > 8:
                coordy = coordy[:8]
            coordz = f"{coord[2]:.3f}"
            if len(coordz) > 8:
                coordz = coordz[:8]
            f.write(f"ATOM  {atom_id:>5}  {atom_name:<3} {nt_name:>3} {chain_name}{nt_num:>4}    {coordx:>8}{coordy:>8}{coordz:>8}\n")
            atom_id += 1
    f.write("END")
    f.close()

def write_PDB(DEEPCRYORNA_HOME,map_origin,dict_right_chain_order,nt_chains_raw_and_outpdbfile):
    list_backbone_templates, list_sugar_base_AG_templates, list_sugar_base_CU_templates, dict_base_templates = get_template_nucleotide(DEEPCRYORNA_HOME)
    backbone_ref_atom_names = ["P","OP1","OP2","O5'","C5'","C4'","C3'","O3'","C2'","O2'","C1'","O4'"]
    sugar_ref_atom_names = ["C4'","C3'","O3'","C2'","O2'","C1'","O4'"]

    nt_chains_raw, outpdbfile = nt_chains_raw_and_outpdbfile
    nt_chains = copy.deepcopy(nt_chains_raw)

    f = open(outpdbfile,"w")
    atom_id = 1
    old_chain_idx = None
    for nt_num, nt in enumerate(nt_chains): 
        if len(nt) == 4:
            nt_idx, nt_name, atoms, chain_idx = nt
        else:
            raise ValueError(f"wrong nt")

        chain_name = chr(ord("A")+dict_right_chain_order[chain_idx])

        if nt_name in ["A","G"]:
            base_ref_atom_names = ["N9","C4","C8"]
            if "N1" in atoms.keys() and "N9" not in atoms.keys():
                atoms["N9"] = atoms["N1"]
                atoms.pop("N1")
            if "C2" in atoms.keys() and "C4" not in atoms.keys():
                atoms["C4"] = atoms["C2"]
                atoms.pop("C2")
            if "C6" in atoms.keys() and "C8" not in atoms.keys():
                atoms["C8"] = atoms["C6"]
                atoms.pop("C6")
        elif nt_name in ["C","U"]:
            base_ref_atom_names = ["N1","C2","C6"]
            if "N9" in atoms.keys() and "N1" not in atoms.keys():
                atoms["N1"] = atoms["N9"]
                atoms.pop("N9")
            if "C4" in atoms.keys() and "C2" not in atoms.keys():
                atoms["C2"] = atoms["C4"]
                atoms.pop("C4")
            if "C8" in atoms.keys() and "C6" not in atoms.keys():
                atoms["C6"] = atoms["C8"]
                atoms.pop("C8")
        else:
            raise ValueError(f"Invalid nucleotide name: {nt_name}")
      
        has_complete_ref_base_atoms = True
        for name in base_ref_atom_names:
            if name not in atoms.keys():
                has_complete_ref_base_atoms = False
                break
        if not has_complete_ref_base_atoms:
            sugar_ref_atom_coords = [atoms[atom_name] for atom_name in sugar_ref_atom_names]
            sugar_ref_atom_coords = np.asarray(sugar_ref_atom_coords)
            if nt_name in ["A","G"]:
                rebuilt_atom_coords = rebuild_atom(list_sugar_base_AG_templates,sugar_ref_atom_names,sugar_ref_atom_coords,base_ref_atom_names)
            else:
                rebuilt_atom_coords = rebuild_atom(list_sugar_base_CU_templates,sugar_ref_atom_names,sugar_ref_atom_coords,base_ref_atom_names)
            for tag, atom_name in enumerate(base_ref_atom_names):
                atoms[atom_name] = rebuilt_atom_coords[tag]

        base_ref_atom_coords = []
        for name in base_ref_atom_names:
            if name in atoms.keys():
                base_ref_atom_coords.append(atoms[name])
        if len(base_ref_atom_coords) == 3:
            base_ref_atom_coords = np.asarray(base_ref_atom_coords)
            full_base = rebuild_base(dict_base_templates,nt_name,base_ref_atom_names,base_ref_atom_coords)
            for atom_name in full_base:
                atoms[atom_name] = full_base[atom_name]
           
        for atom_name in atoms:
            coord = atoms[atom_name]
            coord += map_origin
            coord[0] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) 
            coord[1] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) 
            coord[2] += np.random.uniform(0.01,0.05) * np.random.choice([-1.,1.]) # alter the same atoms by adding a small shift
            coordx = f"{coord[0]:.3f}"
            if len(coordx) > 8:
                coordx = coordx[:8]
            coordy = f"{coord[1]:.3f}"
            if len(coordy) > 8:
                coordy = coordy[:8]
            coordz = f"{coord[2]:.3f}"
            if len(coordz) > 8:
                coordz = coordz[:8]
            f.write(f"ATOM  {atom_id:>5}  {atom_name:<3} {nt_name:>3} {chain_name}{nt_idx:>4}    {coordx:>8}{coordy:>8}{coordz:>8}\n")
            atom_id += 1
        old_chain_idx = chain_idx
    f.write("END\n")
    f.close()

    return outpdbfile

def generate_structure_files(DEEPCRYORNA_HOME,map_origin,dict_right_chain_order,list_processed_nt_chains,filename,ncpu):
    ncpu_available = multiprocessing.cpu_count()
    if ncpu > ncpu_available:
        ncpu = ncpu_available

    pool = multiprocessing.Pool(processes=ncpu)
    partial_function = partial(write_PDB,DEEPCRYORNA_HOME,map_origin,dict_right_chain_order)
    
    list_nt_chains_raw_and_outpdbfile = []
    for i, nt_chains in enumerate(list_processed_nt_chains):
        list_nt_chains_raw_and_outpdbfile.append((nt_chains, f"{filename}-{i+1}.pdb"))

    list_outpdbfile_names = pool.map(partial_function, list_nt_chains_raw_and_outpdbfile)

    pool.close()
    pool.join()

    return list_outpdbfile_names

def filter_alignment(alignment):
    seqA, seqB, score, long_chain, list_index = alignment
    for c1, c2 in zip(seqA,seqB):
        if c1 == "T" and c2 not in ["-","N"]:
            return False
    return True

def map_best_alignments_to_nt_chains(list_nt, list_alignments, num_best_alignment):
    list_alignments.sort(key=lambda x:x[2],reverse=True)

    list_best_alignments = []
    list_mapped_nt_chains = []
    list_mapped_nt_chains_no_atoms = []
    best_score = None
    list_scores = []
    for alignment in list_alignments:
        if filter_alignment(alignment) and alignment not in list_best_alignments:
            mapped_nt_chain, mapped_nt_chain_no_atoms = map_one_alignment_to_nt_chain(list_nt,alignment)
            if best_score is None:
                best_score = alignment[2]
            if not mapped_nt_chain:
                continue
            if mapped_nt_chain_no_atoms in list_mapped_nt_chains_no_atoms:
                continue
            list_mapped_nt_chains.append(mapped_nt_chain)
            list_mapped_nt_chains_no_atoms.append(mapped_nt_chain_no_atoms)
            list_best_alignments.append(alignment)
            list_scores.append(alignment[2])
            if alignment[2] < best_score and len(list_best_alignments) >= num_best_alignment:
                break

    if len(list_best_alignments) == 0:
        for alignment in list_alignments:
            if alignment not in list_best_alignments:
                mapped_nt_chain, mapped_nt_chain_no_atoms = map_one_alignment_to_nt_chain(list_nt,alignment)
                if best_score is None:
                    best_score = alignment[2]
                if not mapped_nt_chain:
                    continue
                if mapped_nt_chain_no_atoms in list_mapped_nt_chains_no_atoms:
                    continue
                list_mapped_nt_chains.append(mapped_nt_chain)
                list_mapped_nt_chains_no_atoms.append(mapped_nt_chain_no_atoms)
                list_best_alignments.append(alignment)
                list_scores.append(alignment[2])
                if alignment[2] < best_score and len(list_best_alignments) >= num_best_alignment:
                    break
    
    num_best_score = list_scores.count(best_score)
    if num_best_score > 1:
        list_2D_scores = []
        for i in range(num_best_score):
            score_2D = calc_2D_score(list_mapped_nt_chains[i])
            list_2D_scores.append(score_2D)
        index_ranked_by_2D_score = sorted(range(num_best_score), key=lambda k:list_2D_scores[k], reverse=True)

        new_list_mapped_nt_chains = []
        new_list_best_alignments = []
        for idx in index_ranked_by_2D_score:
            new_list_mapped_nt_chains.append(list_mapped_nt_chains[idx])
            new_list_best_alignments.append(list_best_alignments[idx])

        if num_best_score < num_best_alignment:
            new_list_mapped_nt_chains.extend(list_mapped_nt_chains[num_best_score:])
            new_list_best_alignments.extend(list_best_alignments[num_best_score:])
            return new_list_best_alignments, new_list_mapped_nt_chains
        else:
            return new_list_best_alignments[0:num_best_alignment], new_list_mapped_nt_chains[0:num_best_alignment]
    else:
        return list_best_alignments, list_mapped_nt_chains

def map_one_alignment_to_nt_chain(list_nt0,alignment):
    list_nt = copy.deepcopy(list_nt0)
    seqA, seqB, score, long_chain, list_chain_order = alignment

    assert len(seqA) == len(seqB)
    chain_idx = 0
    nt_idx = 0
    mapped_nt_chain = []
    mapped_nt_chain_no_atoms = []
    count = 0
    for c1, c2 in zip(seqA,seqB):
        if c1 == "-" and c2 not in ["N","P"]:
            count += 1
            continue
        if c1 == "-" and c2 in ["N", "P"]:
            count += 1
            continue
        if c1 == "T" and c2 in ["N","P"]:
            chain_idx += 1
            nt_idx = 0
            count += 1
            continue
        if c1 == "T" and c2 == "-":
            chain_idx += 1
            nt_idx = 0
            continue
        if c1 == "T" and c2 in ["A", "G", "C", "U", "X"]:
            chain_idx += 1
            nt_idx = 0
            count += 1
            continue
        if c2 == "-":
            if c1 in ["A","G","C","U"]:
                nt_idx += 1
            continue
        if c2 in ["N","P"]:
            if c1 in ["A","G","C","U"]:
                nt_idx += 1
                count += 1
            continue
        if long_chain[count] is not None:
            nt_idx += 1
            nt = list_nt[ long_chain[count] ]
            nt.append(list_chain_order[chain_idx])
            nt[0] = nt_idx
            nt[1] = c1.upper()
            mapped_nt_chain.append(nt)
            mapped_nt_chain_no_atoms.append([nt_idx,c1.upper(),long_chain[count],list_chain_order[chain_idx]])
        count += 1
    mapped_nt_chain = sorted(mapped_nt_chain, key = lambda x: (x[3], x[0]))
    mapped_nt_chain_no_atoms = sorted(mapped_nt_chain_no_atoms, key = lambda x: (x[3], x[0]))
    return mapped_nt_chain, mapped_nt_chain_no_atoms

def map_all_alignments_to_nt_chains(list_nt,list_alignments):
    list_mapped_nt_chains = []
    for alignment in list_alignments:
        mapped_nt_chain = map_one_alignment_to_nt_chain(list_nt,alignment)
        if mapped_nt_chain:
            list_mapped_nt_chains.append(mapped_nt_chain)
    return list_mapped_nt_chains

def filter_long_chains_by_chain_num(list_all_long_chains, list_long_chain_num, num_native_chain, max_num):
    list_idx_chain_num = []
    for i, long_chain_num in enumerate(list_long_chain_num):
        list_idx_chain_num.append([i,long_chain_num])
    list_idx_chain_num.sort(key=lambda x:x[1])
    new_list_all_long_chains = []
    largest_chain_num = -1
    for v in list_idx_chain_num:
        idx, chain_num = v
        if len(new_list_all_long_chains) < max_num:
            new_list_all_long_chains.append(list_all_long_chains[idx])
            if chain_num > largest_chain_num:
                largest_chain_num = chain_num
        elif len(new_list_all_long_chains) >= 100000:
            break
        elif chain_num <= num_native_chain + 2:
            new_list_all_long_chains.append(list_all_long_chains[idx])
            if chain_num > largest_chain_num:
                largest_chain_num = chain_num
        elif chain_num == largest_chain_num:
            new_list_all_long_chains.append(list_all_long_chains[idx])
        else:
            break
    return new_list_all_long_chains

def remove_short_chains_from_native_seq(list_native_seq_raw):
    list_native_seq = copy.deepcopy(list_native_seq_raw)
    to_remove_idx = []
    dict_right_chain_order = {}
    count = 0
    for i, seq in enumerate(list_native_seq):
        if len(seq) < 6:
            to_remove_idx.append(i)
            count += 1
        else:
            dict_right_chain_order[i-count] = i

    for idx in sorted(to_remove_idx,reverse=True):
        list_native_seq.pop(idx)

    return list_native_seq, dict_right_chain_order

def convert_atoms_to_structures(DEEPCRYORNA_HOME,list_native_seq_raw,map_origin,atomseg,ncpu,filename):
    if "128" in filename:
        patch_size = 128
    else:
        patch_size = 64
    print(f"      For patch size {patch_size}:", flush=True)

    list_native_seq, dict_right_chain_order = remove_short_chains_from_native_seq(list_native_seq_raw)

    list_nt = assign_atom_to_nt(DEEPCRYORNA_HOME,atomseg=atomseg,apix=0.5)
    write_PDB_all_nts(DEEPCRYORNA_HOME,map_origin,list_nt,f"tmp/all_nts_ps{patch_size}.pdb")

    list_neighbors_candidates = sort_list_nt(list_native_seq,list_nt)

    list_all_short_chains = get_all_short_chains(list_native_seq,list_neighbors_candidates)
    list_all_short_chains, list_split_chains_idx = split_all_short_chains(list_all_short_chains, list_nt)
    max_num_short_chains = 32
    if len(list_all_short_chains) > max_num_short_chains:
        list_all_short_chains = list_all_short_chains[0:max_num_short_chains]
        list_split_chains_idx = list_split_chains_idx[0:max_num_short_chains]
    print(f"        Got {len(list_all_short_chains)} sets of short chains.", flush=True)

    for i, short_chains in enumerate(list_all_short_chains):
        outpdbfile = f"tmp/short-chain-ps{patch_size}-{i+1}.pdb"
        write_PDB_short_chains(DEEPCRYORNA_HOME,map_origin,short_chains,outpdbfile,list_nt)

    num_native_chain = len(list_native_seq)
    list_all_long_chains, list_long_chain_num = thread_all_short_chains_into_long_chains(list_nt,list_all_short_chains,num_native_chain,list_split_chains_idx)

    max_num_long_chains = 2000 
    list_all_long_chains =  filter_long_chains_by_chain_num(list_all_long_chains, list_long_chain_num, num_native_chain, max_num_long_chains)
    print(f"        Got {len(list_all_long_chains)} sets of long chains.", flush=True)

    list_alignments, list_alignments_cluster = align_sequence_for_all_long_chains_efficiently(list_native_seq,list_nt,list_all_long_chains,DEEPCRYORNA_HOME,ncpu)

    num_best_alignment = 10
    list_best_alignments, list_mapped_nt_chains = map_best_alignments_to_nt_chains(list_nt, list_alignments, num_best_alignment)
    alignment_result_file = f"ps{patch_size}/best_alignment_results_ps{patch_size}.txt"
    print(f"        Got {len(list_best_alignments)} best alignments.", flush=True)
    if len(list_best_alignments) > 0:
        print(f"        The best alignment results are stored in '{alignment_result_file}'.", flush=True)
        with open(alignment_result_file, "w") as f:
            for i, alignment in enumerate(list_best_alignments):
                f.write(f"best alignment {i+1}\n")
                f.write(f"alignment score: {alignment[2]}\n")
                f.write(f"native seq: {alignment[0]}\n")
                f.write(f"  pred seq: {alignment[1]}\n\n")

    list_outpdbfile_names = generate_structure_files(DEEPCRYORNA_HOME,map_origin,dict_right_chain_order,list_mapped_nt_chains,filename,ncpu)
    if len(list_outpdbfile_names) == 1:
        print(f"        {len(list_outpdbfile_names)} structure has been generated in the following file:")
        print(f"          {list_outpdbfile_names[0]}\n")
    elif len(list_outpdbfile_names) > 1:
        print(f"        {len(list_outpdbfile_names)} structures have been generated in the following files:")
        for pdbfile_name in list_outpdbfile_names:
            print(f"          {pdbfile_name}")
        print()
    else:
        print(f"        Failed in predicting the RNA structures from the given cryo-EM map for patch size {patch_size}.\n")

    return list_best_alignments, list_mapped_nt_chains, dict_right_chain_order, list_outpdbfile_names

def cluster_atoms(pred_atoms):
    clustered_atoms = np.zeros(pred_atoms.shape,dtype=np.int8)
    for i in range(1,16):
        atom_image = np.zeros(pred_atoms.shape,dtype=np.int8)
        atom_image[pred_atoms==i] = 1
        if i >= 13:
            atom_image[pred_atoms==(i+3)] = 1

        kernel = np.ones((3,3,3))
        distance = convolve(atom_image,kernel,mode="constant")

        coords = peak_local_max(distance, min_distance=5, threshold_abs=5, p_norm=2)
        assert coords.ndim == 2 and coords.shape[1] == 3, f"coords are not 3D {coords.shape}"

        if i < 13:
            clustered_atoms[coords[:,0],coords[:,1],coords[:,2]] = i
        else:
            for coord in coords:
                pointnum_AG = 0
                pointnum_CU = 0
                x = coord[0]
                y = coord[1]
                z = coord[2]
                for m in range(-5,6):
                    for n in range(-5,6):
                        for p in range(-5,6):
                            new_x = x + m
                            new_y = y + n
                            new_z = z + p
                            if pred_atoms[new_x,new_y,new_z] == i:
                                pointnum_AG += 1
                            if pred_atoms[new_x,new_y,new_z] == i+3:
                                pointnum_CU += 1
                if pointnum_AG >= pointnum_CU:
                    clustered_atoms[x,y,z] = i
                else:
                    clustered_atoms[x,y,z] = i+3
    return clustered_atoms
