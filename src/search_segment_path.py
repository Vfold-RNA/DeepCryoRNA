import random
import math
from predict_structures_from_atoms import *
import time


def build_chain_neighbor_graph(list_nt, short_chains):
    dict_next_neighbor = {}
    dict_next_neighbor_dis = {}
    dict_next_neighbor_helix_rmsd = {}

    dict_last_neighbor = {}

    for i in range(len(short_chains)):
        dict_next_neighbor[i] = []
        dict_next_neighbor_dis[i] = []
        dict_next_neighbor_helix_rmsd[i] = []
        dict_last_neighbor[i] = []

    short_chain_distance_matrix = np.zeros((len(short_chains),len(short_chains)))
    short_chain_helix_rmsd_matrix = np.zeros((len(short_chains), len(short_chains)))

    for i in range(len(short_chains)):
        for j in range(i+1,len(short_chains)):
            neighbor_type, chain_dis, helix_rmsd = determine_if_two_short_chains_are_neighbors(short_chains[i],short_chains[j],list_nt,cutoff=20.0)
            short_chain_distance_matrix[i][j] = chain_dis[0]
            short_chain_distance_matrix[j][i] = chain_dis[1]
            short_chain_helix_rmsd_matrix[i][j] = helix_rmsd[0]
            short_chain_helix_rmsd_matrix[j][i] = helix_rmsd[1]

            if neighbor_type[0] == 1 and chain_dis[0] <= 20.0:
                dict_next_neighbor[i].append(j)
                dict_next_neighbor_helix_rmsd[i].append(helix_rmsd[0])
                dict_last_neighbor[j].append(i)

            if neighbor_type[1] == 1 and chain_dis[1] <= 20.0:
                dict_next_neighbor[j].append(i)
                dict_next_neighbor_helix_rmsd[j].append(helix_rmsd[1])
                dict_last_neighbor[i].append(j)

    return  dict_next_neighbor, dict_last_neighbor, short_chain_distance_matrix, short_chain_helix_rmsd_matrix


def build_linking_nts(list_nt, short_chains, short_chain_distance_matrix, chain_idx_1, chain_idx_2, dis_threshold, force_insert=False):
    list_nts_in_helix = [
        {"P": [0.3, -3.6, 2.2], "C5'": [0.3, -3.4, 4.9], "O5'": [-0.4, -3.4, 3.6], "C4'": [-0.6, -3.3, 6.0],
         "O4'": [-1.2, -2.0, 6.1], "C3'": [-1.8, -4.3, 6.0], "O3'": [-1.5, -5.6, 6.4], "C2'": [-2.8, -3.6, 6.9],
         "O2'": [-2.5, -3.8, 8.3], "C1'": [-2.6, -2.1, 6.6], "C4": [-4.8, -1.3, 5.6], "C8": [-3.1, -1.4, 4.2],
         "N9": [-3.4, -1.6, 5.5], "OP1": [1.2, -4.7, 2.3], "OP2": [-0.7, -3.5, 1.2]},
        {"P": [-2.4, -6.9, 6.0], "C5'": [-3.9, -6.6, 8.1], "O5'": [-3.8, -6.6, 6.6], "C4'": [-5.4, -6.4, 8.4],
         "O4'": [-5.8, -5.1, 8.1], "C3'": [-6.4, -7.3, 7.7], "O3'": [-6.5, -8.6, 8.2], "C2'": [-7.7, -6.5, 7.9],
         "O2'": [-8.2, -6.7, 9.1], "C1'": [-7.2, -5.1, 7.7], "C4": [-8.5, -4.0, 5.8], "C8": [-6.3, -4.3, 5.5],
         "N9": [-7.3, -4.5, 6.4], "OP1": [-1.8, -8.1, 6.6], "OP2": [-2.5, -6.8, 4.5]},
        {"P": [-7.1, -9.8, 7.3], "C5'": [-9.5, -9.5, 8.2], "O5'": [-8.6, -9.5, 7.1], "C4'": [-10.9, -9.2, 7.8],
         "O4'": [-10.9, -7.8, 7.3], "C3'": [-11.5, -9.9, 6.6], "O3'": [-12.0, -11.2, 7.0], "C2'": [-12.6, -9.0, 6.1],
         "O2'": [-13.7, -9.1, 6.9], "C1'": [-11.9, -7.7, 6.3], "C4": [-11.9, -6.6, 4.0], "C8": [-9.9, -7.1, 4.8],
         "N9": [-11.3, -7.1, 5.1], "OP1": [-7.0, -11.1, 8.1], "OP2": [-6.4, -9.7, 6.0]},
        {"P": [-12.1, -12.4, 5.9], "C5'": [-14.6, -11.7, 5.4], "O5'": [-13.2, -11.9, 4.9], "C4'": [-15.5, -11.3, 4.2],
         "O4'": [-15.1, -9.9, 3.8], "C3'": [-15.4, -12.1, 2.9], "O3'": [-16.1, -13.3, 3.0], "C2'": [-15.9, -11.1, 1.9],
         "O2'": [-17.4, -11.1, 2.0], "C1'": [-15.4, -9.8, 2.5], "C4": [-14.0, -8.8, 0.6], "C8": [-12.9, -9.5, 2.3],
         "N9": [-14.1, -9.4, 1.8], "OP1": [-12.6, -13.6, 6.6], "OP2": [-10.8, -12.5, 5.1]},
        {"P": [-15.7, -14.5, 2.0], "C5'": [-17.5, -13.7, 0.2], "O5'": [-16.1, -14.0, 0.6], "C4'": [-17.6, -13.2, -1.2],
         "O4'": [-17.0, -11.9, -1.3], "C3'": [-16.9, -14.1, -2.3], "O3'": [-17.7, -15.2, -2.6],
         "C2'": [-16.8, -13.1, -3.4], "O2'": [-18.0, -12.9, -4.1], "C1'": [-16.5, -11.8, -2.6],
         "C4": [-14.2, -11.0, -3.5], "C8": [-14.2, -11.7, -1.4], "N9": [-15.0, -11.5, -2.5], "OP1": [-16.6, -15.7, 2.3],
         "OP2": [-14.2, -14.7, 2.0]},
        {"P": [-17.0, -16.5, -3.3], "C5'": [-17.5, -15.5, -5.7], "O5'": [-16.5, -15.9, -4.7],
         "C4'": [-16.8, -15.1, -7.0], "O4'": [-16.0, -13.9, -6.7], "C3'": [-15.7, -16.0, -7.5],
         "O3'": [-16.2, -17.1, -8.3], "C2'": [-14.8, -15.1, -8.4], "O2'": [-15.5, -14.8, -9.6],
         "C1'": [-14.9, -13.8, -7.5], "C4": [-12.4, -13.2, -7.1], "C8": [-13.6, -13.9, -5.3],
         "N9": [-13.7, -13.6, -6.7], "OP1": [-18.0, -17.5, -3.5], "OP2": [-15.8, -16.8, -2.5]}
    ]

    if force_insert:
        list_tail_head_nts = [
            list_nt[short_chains[chain_idx_1][-1]],
            list_nt[short_chains[chain_idx_2][0]]
        ]
        dis = short_chain_distance_matrix[chain_idx_1][chain_idx_2]
        if dis <= dis_threshold and dis_threshold <= 14.0:
            list_seq_gap = [1] # insert a nt in the pred seq
        elif dis <= dis_threshold and dis_threshold > 14.0:
            list_seq_gap = [2]  # insert two nts in the pred seq
        else:
            return False, 1000., [] # do not merge if dis > 8.0
    else:
        list_tail_head_nts = [
            list_nt[short_chains[chain_idx_1][-2]],
            list_nt[short_chains[chain_idx_1][-1]],
            list_nt[short_chains[chain_idx_2][0]],
            list_nt[short_chains[chain_idx_2][1]]
        ]
        list_seq_gap = [0, 1, 2] # consider seq gap of 0, 1, 2

    best_rmsd = 10000.
    atom_names = ["P", "C5'", "C4'", "C3'", "O3'", "C2'", "C1'"]
    for i in list_seq_gap:
        if not force_insert:
            template_nts = list_nts_in_helix[0:2] + list_nts_in_helix[(2+i):(4+i)]
        else:
            template_nts = list_nts_in_helix[0:1] + list_nts_in_helix[(1+i):(2+i)]
        assert len(list_tail_head_nts) == len(template_nts)

        ref_coords = []
        template_coords = []
        for nt1, nt2 in zip(list_tail_head_nts, template_nts):
            for name in atom_names:
                if name in nt1[2]:
                    ref_coords.append(nt1[2][name])
                    template_coords.append(nt2[name])
        ref_coords = np.asarray(ref_coords)
        template_coords = np.asarray(template_coords)

        rmsd, rot, tran = calc_rmsd([], ref_coords, template_coords)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_seq_gap = i
            best_rot = rot
            best_tran = tran

        '''
        if chain_idx_1 == 20 and chain_idx_2 == 14:
            print()
            print(f"seq gap {i}")
            print(ref_coords)
            print(template_coords)
            print(rmsd)
        '''

    list_added_nts = []
    is_merged = False

    if (not force_insert) and (best_rmsd > 2.0):
        return is_merged, best_rmsd, list_added_nts
    else:
        is_merged = True

    for i in range(best_seq_gap):
        nt = []
        nt.append(1) # nt idx
        nt.append(1) # nt type 1: A; 2: C
        nt.append({}) # keep atoms
        if force_insert:
            nt_in_helix = list_nts_in_helix[i+1]
        else:
            nt_in_helix = list_nts_in_helix[i+2]
        for name in nt_in_helix:
            vec = np.asarray(nt_in_helix[name])
            transformed_vec = get_transformed(best_rot, best_tran, vec)
            nt[2][name] = transformed_vec
        list_added_nts.append(nt)

    return is_merged, best_rmsd, list_added_nts


def merge_chains_directional(short_chains, list_merged_chains):
    """
    Merge chains according to directional rules:
    chain_id_1's tail connects to chain_id_2's head via added_nts.
    """
    next_map = {}  # chain_id_1 -> (chain_id_2, added_nts)
    prev_map = {}  # chain_id_2 -> (chain_id_1, added_nts)

    # Build directional mapping
    for c1, c2, nts in list_merged_chains:
        if c1 in next_map:
            raise ValueError(f"Chain {c1} has multiple outgoing connections")
        if c2 in prev_map:
            raise ValueError(f"Chain {c2} has multiple incoming connections")
        next_map[c1] = (c2, nts)
        prev_map[c2] = (c1, nts)

    visited = set()
    merged = []

    # Start from all head chains (not in prev_map)
    for idx in range(len(short_chains)):
        if idx in visited or idx in prev_map:
            continue
        chain = copy.deepcopy(short_chains[idx])
        visited.add(idx)
        curr = idx
        while curr in next_map:
            nxt, nts = next_map[curr]
            chain += nts + copy.deepcopy(short_chains[nxt])
            visited.add(nxt)
            curr = nxt
        merged.append(chain)

    # Add unvisited chains (not part of any path)
    for idx in range(len(short_chains)):
        if idx not in visited:
            merged.append(copy.deepcopy(short_chains[idx]))

    return merged


def add_nts_to_merge_short_chains(list_nt, short_chains, dict_next_neighbor, dict_last_neighbor, short_chain_distance_matrix):
    list_merged_chains = []

    list_nt_new = copy.deepcopy(list_nt)

    merged_chain_j = []
    for i in range(len(short_chains)):
        next_neighbor = dict_next_neighbor[i]
        min_dis = 100000.
        best_added_nts = None
        best_rmsd = None
        best_j = None
        for j in next_neighbor:
            if j in merged_chain_j:
                continue
            is_merged, rmsd, added_nts = build_linking_nts(list_nt, short_chains, short_chain_distance_matrix, i, j, dis_threshold=None, force_insert=False)
            if is_merged:
                if short_chain_distance_matrix[i][j] < min_dis:
                    min_dis = short_chain_distance_matrix[i][j]
                    best_added_nts = added_nts
                    best_rmsd = rmsd
                    best_j = j
        if best_added_nts is not None:
            merged_chain_j.append(best_j)
            list_merged_chains.append((i, best_j, [len(list_nt_new)+m for m in range(len(best_added_nts))]))
            list_nt_new.extend(best_added_nts)
            continue
            #print(f"chain {i} and {best_j} are merged with {len(best_added_nts)} nts. rmsd: {best_rmsd}", flush=True)

        #merge if chain i has only one next neighbor chain j
        if best_added_nts is None and len(next_neighbor) == 1:
            j = next_neighbor[0]
            if dict_next_neighbor[j] == i:
                continue
            if j in merged_chain_j:
                continue
            if len(dict_last_neighbor[j]) > 1:
                dis_threshold = 8.0
            else:
                dis_threshold = 10.0
            is_merged, rmsd, added_nts = build_linking_nts(list_nt, short_chains, short_chain_distance_matrix, i, j, dis_threshold=dis_threshold, force_insert=True)
            if is_merged:
                merged_chain_j.append(j)
                list_merged_chains.append((i, j, [len(list_nt_new) + m for m in range(len(added_nts))]))
                list_nt_new.extend(added_nts)
                continue
                #print(f"chain {i} and {j} are merged with {len(added_nts)} nts. rmsd: {rmsd}", flush=True)

    to_remove_idx = []
    for i in range(len(list_merged_chains)):
        c1, c2, _ = list_merged_chains[i]
        for j in range(i+1, len(list_merged_chains)):
            c3, c4, _ = list_merged_chains[j]
            if c1 == c4 and c2 == c3:
                to_remove_idx.append(i)
                to_remove_idx.append(j)
                #raise ValueError(f"cannot merge two chains {c1} and {c2} into a circle!")
    for i in sorted(set(to_remove_idx), reverse=True):
        del list_merged_chains[i]

    #re-check
    for i in range(len(list_merged_chains)):
        c1, c2, _ = list_merged_chains[i]
        for j in range(i+1, len(list_merged_chains)):
            c3, c4, _ = list_merged_chains[j]
            if c1 == c4 and c2 == c3:
                raise ValueError(f"cannot merge two chains {c1} and {c2} into a circle!")

    short_chains_new = merge_chains_directional(short_chains, list_merged_chains)

    return list_nt_new, short_chains_new


def calc_score(score_cpp_function, list_nt, native_seq, nt_chain):
    decoy_seq, closeness_matrix, new_nt_chain, fuse_bonus = get_decoy_seq_and_closeness_matrix(list_nt, nt_chain, calc_close_mat=False)
    score = score_cpp_function(native_seq.encode(), decoy_seq.encode())

    return score + fuse_bonus


def get_alignments(list_nt, list_native_seq, long_chain):
    decoy_seq, closeness_matrix, new_long_chain, fuse_bonus = get_decoy_seq_and_closeness_matrix(list_nt, long_chain)

    native_seq = "T".join(list_native_seq)

    scores, alignments = global_alignment(native_seq, decoy_seq, closeness_matrix)

    list_native_chain_index = list(range(len(list_native_seq)))
    list_alignments = []
    for score, alignment in zip(scores, alignments):
        seqA, seqB = alignment
        list_alignments.append([seqA, seqB, score+fuse_bonus, new_long_chain, list_native_chain_index])

    return list_alignments


def convert_final_path_to_nt_chain(list_native_seq, list_nt, short_chains, final_path):
    list_nt_copy = copy.deepcopy(list_nt)
    total_score, path = final_path

    # sort by (chain_idx, start_pos)
    sorted_path = sorted(path, key=lambda x: (x[1][0], x[1][1]))

    mapped_nt_chain = []

    seen_nts = []
    aligned_native_seq = "T".join(list_native_seq)
    aligned_pred_seq = ["-"] * len(aligned_native_seq)

    used_nts = []

    for seg_id, match in sorted_path: # seg_id is short chain index
        nc_id, start_pos, end_pos, score = match # nc_id is native chain index
        short_chain = short_chains[seg_id]
        native_chain_seq = list_native_seq[nc_id]
        assert len(short_chain) == end_pos - start_pos
        for i in range(len(short_chain)):
            nt_id = short_chain[i]
            nt = list_nt_copy[nt_id]

            if nt_id in used_nts:
                print("use one nt twice")
                exit()
            used_nts.append(nt_id)

            real_nt_id = start_pos + i + 1
            if (nc_id, real_nt_id) not in seen_nts:
                seen_nts.append((nc_id, real_nt_id))
            else:
                continue
            num_nt = len("T".join(list_native_seq[0:nc_id]) + list_native_seq[nc_id][0:(start_pos+i+1)])
            if nt[1] == 1:
                aligned_pred_seq[num_nt-1] = "A"
            elif nt[1] == 2:
                aligned_pred_seq[num_nt-1] = "C"
            else:
                aligned_pred_seq[num_nt - 1] = "X"
            nt[0] = real_nt_id # real nt index in a chain
            nt[1] = native_chain_seq[start_pos+i] # nt[2] stores atom info. and do not change
            nt.append(nc_id) # nt[3] is the native chain index
            mapped_nt_chain.append(nt)

    aligned_pred_seq = "".join(aligned_pred_seq)

    alignment = (aligned_native_seq, aligned_pred_seq, total_score)

    return mapped_nt_chain, alignment


def convert_one_path_to_global_alignment(list_native_seq, result):
    list_nt, short_chains, final_path = result
    score, path = final_path
    sorted_path = sorted(path, key=lambda x: (x[1][0], x[1][1]))  # sort by (chain_idx, start_pos)
    long_chain = []
    for seg_id, matches in sorted_path:
        for nt_idx in short_chains[seg_id]:
            long_chain.append(nt_idx)
        long_chain.append(None)
    long_chain.pop()
    alignments = get_alignments(list_nt, list_native_seq, long_chain)
    #print(f"get {len(alignments)} alignments", flush=True)
    num_best_alignment = 1
    best_alignments, mapped_nt_chains = map_best_alignments_to_nt_chains(list_nt, alignments, num_best_alignment)
    best_alignments = best_alignments[0:num_best_alignment]
    mapped_nt_chains = mapped_nt_chains[0:num_best_alignment]
    #print(f"get {len(best_alignments)} best alignments", flush=True)

    return [best_alignments, mapped_nt_chains]


def write_structures_from_final_paths_global_alignment(DEEPCRYORNA_HOME, map_origin, dict_right_chain_order, patch_size, list_native_seq, ncpu, filename, list_results):
    max_result = 10
    partial_function = partial(convert_one_path_to_global_alignment, list_native_seq)
    with multiprocessing.Pool(processes=ncpu) as pool:
        list_alignments_and_mapped_nt_chains = pool.map(partial_function, list_results[0:max_result])

    list_best_alignments =[]
    list_mapped_nt_chains = []
    for best_alignments, mapped_nt_chains in list_alignments_and_mapped_nt_chains:
        list_best_alignments.extend(best_alignments)
        list_mapped_nt_chains.extend(mapped_nt_chains)

    alignment_result_file = f"ps{patch_size}/best_alignment_results_ps{patch_size}.txt"
    print(f"        Got {len(list_best_alignments)} best alignments.", flush=True)
    if len(list_best_alignments) > 0:
        print(f"        The best alignment results are stored in '{alignment_result_file}'.", flush=True)
        with open(alignment_result_file, "w") as f:
            for i, alignment in enumerate(list_best_alignments):
                f.write(f"best alignment {i+1}\n")
                f.write(f"alignment score: {alignment[2]}\n")
                f.write(f"native seq: {alignment[0]}\n")
                f.write(f"  pred seq: {alignment[1]}\n")
                norm_score = calc_per_chain_length_normalized_alignment_score(alignment[0], alignment[1])
                f.write(f"per-chain length-normalized alignment score: {norm_score}\n\n")

    list_outfile_names = generate_structure_files(DEEPCRYORNA_HOME,map_origin,dict_right_chain_order,list_mapped_nt_chains,filename,ncpu)
    if len(list_outfile_names) == 1:
        print(f"        {len(list_outfile_names)} structure has been generated in the following file:")
        print(f"          {list_outfile_names[0]}\n")
    elif len(list_outfile_names) > 1:
        print(f"        {len(list_outfile_names)} structures have been generated in the following files:")
        for file_name in list_outfile_names:
            print(f"          {file_name}")
        print()
    else:
        print(f"        Failed in predicting the RNA structures from the given cryo-EM map for patch size {patch_size}.\n")

    return list_best_alignments, list_mapped_nt_chains, dict_right_chain_order, list_outfile_names


def write_structures_from_final_paths_local_alignment(DEEPCRYORNA_HOME, map_origin, dict_right_chain_order, patch_size, list_native_seq, list_nt, short_chains, ncpu, filename, list_final_paths):
    list_mapped_nt_chains = []
    list_alignments = []

    for final_path in list_final_paths:
        mapped_nt_chain, alignment = convert_final_path_to_nt_chain(list_native_seq, list_nt, short_chains, final_path)
        list_mapped_nt_chains.append(mapped_nt_chain)
        list_alignments.append(alignment)

    alignment_result_file = f"ps{patch_size}/best_alignment_results_ps{patch_size}.txt"
    print(f"        Got {len(list_alignments)} best alignments.", flush=True)
    if len(list_alignments) > 0:
        print(f"        The best alignment results are stored in '{alignment_result_file}'.", flush=True)
        with open(alignment_result_file, "w") as f:
            for i, alignment in enumerate(list_alignments):
                f.write(f"best alignment {i+1}\n")
                f.write(f"alignment score: {alignment[2]}\n")
                f.write(f"native seq: {alignment[0]}\n")
                f.write(f"  pred seq: {alignment[1]}\n\n")

    list_outfile_names = generate_structure_files(DEEPCRYORNA_HOME,map_origin,dict_right_chain_order,list_mapped_nt_chains,filename,ncpu)
    if len(list_outfile_names) == 1:
        print(f"        {len(list_outfile_names)} structure has been generated in the following file:")
        print(f"          {list_outfile_names[0]}\n")
    elif len(list_outfile_names) > 1:
        print(f"        {len(list_outfile_names)} structures have been generated in the following files:")
        for file_name in list_outfile_names:
            print(f"          {file_name}")
        print()
    else:
        print(f"        Failed in predicting the RNA structures from the given cryo-EM map for patch size {patch_size}.\n")

    return list_alignments, list_mapped_nt_chains, dict_right_chain_order, list_outfile_names


def is_overlap(seg_id_a, match_a, seg_id_b, match_b, distance_matrix):
    native_chain_idx_a, ib_a, ie_a, score_a = match_a
    native_chain_idx_b, ib_b, ie_b, score_b = match_b

    if native_chain_idx_a != native_chain_idx_b:
        return False

    redundant_size = 2
    if ie_a <= ib_b + redundant_size:
        seq_gap = ib_b - ie_a
        if distance_matrix[seg_id_a][seg_id_b]  > max_distance(seq_gap):
            return True
        else:
            return False

    if ie_b <= ib_a + redundant_size:
        seq_gap = ib_a - ie_b
        if distance_matrix[seg_id_b][seg_id_a] > max_distance(seq_gap):
            return True
        else:
            return False

    return True


def max_distance(seq_gap):
    if seq_gap <= 2:
        return 20.0
    elif seq_gap == 3:
        return 24.0
    elif seq_gap == 4:
        return 28.0
    elif seq_gap > 4 and seq_gap <= 7:
        return 30.0
    else:
        return 5. * seq_gap


def calc_path_score_with_dis_penalty(short_chain_distance_matrix, path):
    # path has been sorted

    # Start with sum of all alignment scores
    total_score = sum(match[3] for _, match in path)

    max_bonus = 10
    min_dis = 8.0
    max_dis = 40.0

    for i in range(0, len(path)-1):
        seg_a, match_a = path[i]
        seg_b, match_b = path[i+1]
        if match_a[0] != match_b[0]:
            continue
        dis = short_chain_distance_matrix[seg_a][seg_b]

        if dis <= min_dis:
            bonus = max_bonus
        elif dis >= max_dis:
            bonus = 0
        else:
            bonus = round(max_bonus * (max_dis - dis) / (max_dis - min_dis))

        total_score += bonus

    return total_score


def generate_segment_matches(DEEPCRYORNA_HOME, list_native_seq, list_nt, short_chains, min_seg_len, top_n_per_chain=10):
    # Here segments == short chains

    cpp_library = ctypes.CDLL(f'{DEEPCRYORNA_HOME}/alignment_score.so')
    score_cpp_function = cpp_library.get_alignment_score
    score_cpp_function.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    score_cpp_function.restype = ctypes.c_int

    segment_matches = {}

    for seg_idx in range(len(short_chains)):
        short_chain = short_chains[seg_idx]

        seg_len = len(short_chain)
        if seg_len < min_seg_len:
            continue

        matches = []

        for chain_idx, native_seq in enumerate(list_native_seq):
            chain_len = len(native_seq)
            chain_matches = []

            for i in range(chain_len - seg_len + 1):
                native_subseq = native_seq[i:i + seg_len]
                score = calc_score(score_cpp_function, list_nt, native_subseq, short_chain)
                chain_matches.append((chain_idx, i, i + seg_len, score))

            if len(chain_matches) == 0:
                continue

            sorted_chain_matches = sorted(chain_matches, key=lambda x: -x[3])

            filtered_chain_matches = [] # remove too similar matches
            for match in sorted_chain_matches:
                too_close = False
                for kept in filtered_chain_matches:
                    if abs(kept[1] - match[1]) <= 2: # 2 is the position_tolerance
                        too_close = True
                        break
                if not too_close:
                    filtered_chain_matches.append(match)

            max_score = filtered_chain_matches[0][-1]
            if max_score <= 30:
                for i in range(1, len(filtered_chain_matches)):
                    m = filtered_chain_matches[i]
                    if max_score - m[-1] <= 10:
                        filtered_chain_matches[i] = (m[0], m[1], m[2], max_score)

            if len(filtered_chain_matches) <= top_n_per_chain:
                chain_matches = filtered_chain_matches
            else:
                cutoff_score = filtered_chain_matches[top_n_per_chain - 1][3]
                #if seg_len >= min_seg_len and seg_len <= 20:
                    #cutoff_score2 = int(seg_len * 0.2) # at least 60% same nt type for segments shorter than 20 nts
                    #cutoff_score = min(cutoff_score, cutoff_score2)
                chain_matches = [m for m in filtered_chain_matches if m[3] >= cutoff_score]
            matches.extend(chain_matches)

        segment_matches[seg_idx] = matches

        '''
        sorted_matches = sorted(matches, key=lambda x: -x[3])
        print(f"\nmerged short chain {seg_idx} matches")
        for m in sorted_matches:
            print(m, flush=True)
        print(f"select t, chain {seg_idx} in em-short-chain1; center t")
        print(f"select p, resi {sorted_matches[0][1]+1}-{sorted_matches[0][2]} in c{sorted_matches[0][0]}; show cartoon, t or p")
        if seg_idx in [51,56,58]:
            print(f"seg {seg_idx} matches:")
            for m in sorted_matches:
                print(m, flush=True)
            print()
            #exit()
        '''

    return segment_matches


def generate_segment_matches_in_current_path(list_native_seq, short_chains, min_seg_len, path):
    # sort by (chain_idx, start_pos)
    sorted_path = sorted(path, key=lambda x: (x[1][0], x[1][1]))

    segment_matches = {}

    used_segments = []
    for s, m in sorted_path:
        used_segments.append(s)

    for seg_idx in range(len(short_chains)):
        if seg_idx in used_segments:
            continue

        short_chain = short_chains[seg_idx]

        seg_len = len(short_chain)
        #if seg_len >= 2*min_seg_len:
        if seg_len >= min_seg_len:
            continue

        matches = []

        for chain_idx in range(len(list_native_seq)):
            list_occupied_pos_in_native_seq = []

            list_occupied_pos_in_native_seq.append((-10, 0))

            for s, m in sorted_path:
                if m[0] == chain_idx:
                    list_occupied_pos_in_native_seq.append((m[1], m[2])) # m[1]: start_pos, m[2]: end_pos

            native_seq_len = len(list_native_seq[chain_idx])
            list_occupied_pos_in_native_seq.append((native_seq_len, native_seq_len + 10))

            is_sorted = all(a < b and (i == 0 or list_occupied_pos_in_native_seq[i - 1][1] <= a+2) for i, (a, b) in enumerate(list_occupied_pos_in_native_seq))
            if not is_sorted:
                raise ValueError(f"native sequence alignment is wrong! {list_occupied_pos_in_native_seq}")

            for i in range(len(list_occupied_pos_in_native_seq)-1):
                end_pos = list_occupied_pos_in_native_seq[i][1]
                start_pos = list_occupied_pos_in_native_seq[i+1][0]
                gap_len = start_pos - end_pos
                if gap_len + 3 < seg_len: # 3 is tolerant gap
                    continue
                m1 = (chain_idx, end_pos, min(end_pos+seg_len, start_pos), 0)
                matches.append(m1)
                m2 = (chain_idx, max(start_pos-seg_len, end_pos), start_pos, 0)
                if abs(m2[1] - m1[1]) >= 5:
                    matches.append(m2)

        if matches:
            segment_matches[seg_idx] = matches

    '''
    print()
    print("".join(["*"]*80))
    print(path)
    for sid in segment_matches:
        print(f"sid: {sid}, length: {len(short_chains[sid])}", flush=True)
        print(segment_matches[sid], flush=True)
    print("".join(["*"] * 80), flush=True)
    '''

    return segment_matches


def find_best_segment_paths(DEEPCRYORNA_HOME, list_native_seq, list_nt, top_k_paths, beam_width, short_chains, min_seg_len):
    dict_next_neighbor, dict_last_neighbor, short_chain_distance_matrix, short_chain_helix_rmsd_matrix = build_chain_neighbor_graph(list_nt, short_chains)
   
    '''
    print(f"merged short chains' next neighbors:")
    for i in dict_next_neighbor:
        print(i, dict_next_neighbor[i])
    print()
    print(f"merged short chains' last neighbors:")
    for i in dict_last_neighbor:
        print(i, dict_last_neighbor[i])
    '''

    segment_matches = generate_segment_matches(DEEPCRYORNA_HOME, list_native_seq, list_nt, short_chains, min_seg_len, top_n_per_chain=50)

    initial_paths = []
    for seg_id, matches in segment_matches.items():
        for match in matches:
            score = match[3]
            initial_paths.append((score, [(seg_id, match)]))

    beam = sorted(initial_paths, key=lambda x: -x[0])[:beam_width]
    final_paths = []

    '''
    step = 0
    print(f"step {step}:")

    for s, p in beam:
        print(s, p, flush=True)
    print()
    '''

    while beam:
        new_beam = []
        seen_paths = set()

        for total_score, path in beam:
            used_segments = {sid for sid, _ in path}

            extended = False

            for seg_id, matches in segment_matches.items():
                if seg_id in used_segments:
                    continue

                for match in matches:
                    is_overlapped = any(is_overlap(seg_id, match, s, m, short_chain_distance_matrix) for s, m in path)
                    if is_overlapped:
                        continue

                    new_path = path + [(seg_id, match)]

                    path_signature = frozenset((sid, m) for sid, m in new_path)
                    if path_signature in seen_paths:
                        continue

                    new_path = sorted(new_path, key=lambda x: (x[1][0], x[1][1]))

                    new_score = calc_path_score_with_dis_penalty(short_chain_distance_matrix, new_path)

                    seen_paths.add(path_signature)
                    new_beam.append((new_score, new_path))
                    extended = True
                    if len(new_beam) > 2*beam_width:
                        new_beam = sorted(new_beam, key=lambda x: -x[0])[:beam_width]

            # get available segment matches for segments with length < min_seg_len in current path
            segment_matches_remaining = generate_segment_matches_in_current_path(list_native_seq, short_chains, min_seg_len, path)

            for seg_id, matches in segment_matches_remaining.items():
                if seg_id in used_segments:
                    raise ValueError(f"segment {seg_id} should not be used.")

                for match in matches:
                    new_path = path + [(seg_id, match)]

                    path_signature = frozenset((sid, m) for sid, m in new_path)
                    if path_signature in seen_paths:
                        continue

                    new_path = sorted(new_path, key=lambda x: (x[1][0], x[1][1]))

                    new_score = calc_path_score_with_dis_penalty(short_chain_distance_matrix, new_path)

                    seen_paths.add(path_signature)
                    new_beam.append((new_score, new_path))
                    extended = True
                    if len(new_beam) > 2*beam_width:
                        new_beam = sorted(new_beam, key=lambda x: -x[0])[:beam_width]

            if not extended:
                final_paths.append((total_score, path))

        beam = sorted(new_beam, key=lambda x: -x[0])[:beam_width]

        '''
        step += 1
        print(f"step {step}:", flush=True)

        for s, p in beam:
            print(s, p, flush=True)
        print()
        '''

    final_paths = sorted(final_paths, key=lambda x: -x[0])[:top_k_paths]

    return final_paths


def thread_short_chains_into_paths(DEEPCRYORNA_HOME, map_origin, list_native_seq, list_nt, min_seg_len, beam_width, top_k_paths, short_chains):
    dict_next_neighbor, dict_last_neighbor, short_chain_distance_matrix, short_chain_helix_rmsd_matrix = build_chain_neighbor_graph(list_nt, short_chains)

    '''
    outfile = "short_chains.cif"
    write_PDB_short_chains(DEEPCRYORNA_HOME, map_origin, list_nt, [short_chains, outfile])
    print(f"all short chains' next neighbors:")
    for i in dict_next_neighbor:
        print(i, dict_next_neighbor[i])
    print()
    print(f"all short chains' last neighbors:")
    for i in dict_last_neighbor:
        print(i, dict_last_neighbor[i])
    '''

    list_nt_new, short_chains_new = add_nts_to_merge_short_chains(list_nt, short_chains, dict_next_neighbor, dict_last_neighbor, short_chain_distance_matrix)
    short_chains_new = [chain for chain in short_chains_new if len(chain) >= 8]

    #outfile = "merged_short_chains.cif"
    #write_PDB_short_chains(DEEPCRYORNA_HOME, map_origin, list_nt_new, [short_chains_new, outfile])
    #exit()

    list_paths = find_best_segment_paths(DEEPCRYORNA_HOME, list_native_seq, list_nt_new, top_k_paths, beam_width, short_chains_new, min_seg_len)

    results = []
    for i in range(len(list_paths)):
        result = (copy.deepcopy(list_nt_new), copy.deepcopy(short_chains_new), list_paths[i])
        results.append(result)

    return results


def convert_atoms_to_structures_for_large_rna(DEEPCRYORNA_HOME, list_native_seq_raw, map_origin, atomseg, ncpu, filename):
    import pickle

    if "128" in filename:
        patch_size = 128
    else:
        patch_size = 64
    print(f"      For patch size {patch_size}:", flush=True)

    min_chain_length = 15
    list_native_seq, dict_right_chain_order = remove_short_chains_from_native_seq(list_native_seq_raw, min_chain_length)

    list_nt = assign_atom_to_nt(DEEPCRYORNA_HOME, atomseg=atomseg, apix=0.5)

    write_PDB_all_nts(DEEPCRYORNA_HOME,map_origin,list_nt,f"tmp/all_nts_ps{patch_size}.cif")

    list_neighbors_candidates = sort_list_nt(list_native_seq,list_nt)

    list_all_short_chains = get_all_short_chains(list_native_seq,list_neighbors_candidates,ncpu)

    max_num_short_chains = 10
    if len(list_all_short_chains) > max_num_short_chains:
        list_all_short_chains = list_all_short_chains[0:max_num_short_chains]
    print(f"        Got {len(list_all_short_chains)} sets of short chains.", flush=True)

    short_chains_and_outfile = []
    for i, short_chains in enumerate(list_all_short_chains):
        outfile = f"tmp/short-chain-ps{patch_size}-{i+1}.cif"
        short_chains_and_outfile.append((short_chains, outfile))
    pf= partial(write_PDB_short_chains, DEEPCRYORNA_HOME, map_origin, list_nt)
    with multiprocessing.Pool(processes=ncpu) as pool:
        pool.map(pf, short_chains_and_outfile)

    top_k_paths = 1
    beam_width = 100
    min_seg_len = 20

    partial_function = partial(thread_short_chains_into_paths, DEEPCRYORNA_HOME, map_origin, list_native_seq, list_nt, min_seg_len, beam_width, top_k_paths)
    with multiprocessing.Pool(processes=ncpu) as pool:
        list_results = pool.map(partial_function, list_all_short_chains)

    list_results = [result for results in list_results for result in results]
    list_results = sorted(list_results, key=lambda x: -x[2][0]) # result: (list_nt_new, short_chains_new, path)

    list_alignments, list_mapped_nt_chains, dict_right_chain_order, list_outfile_names = write_structures_from_final_paths_global_alignment(DEEPCRYORNA_HOME, map_origin, dict_right_chain_order, patch_size, list_native_seq, ncpu, filename, list_results)

    return list_alignments, list_mapped_nt_chains, dict_right_chain_order, list_outfile_names
