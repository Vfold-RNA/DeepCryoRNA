import numpy as np
import re

def score_match(a, b, pair_scores, T_N_misalignment_penalty):
    if a == 'T' and b not in ['N','P']:
        return T_N_misalignment_penalty
    pair = a + b
    return pair_scores[pair]

def trace_back(matrix_name, D, P, Q, seq1, seq2, aligned1, aligned2, i, j, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments):
    if len(alignments) >= 100:
        return

    if i == 0 and j == 0:
        alignments.append((aligned1,aligned2))
        return

    if len(N_index) >= 2:
        n_index = [k for k, val in enumerate(aligned2) if val == 'N']
        if len(n_index) >= 1 and len(n_index) < len(N_index):
            gap_N_index = [k for k in range(0,n_index[-1]+1) if k > 0 and aligned2[k] == 'N' and aligned2[k-1] == '-']
            for idx in gap_N_index:
                if aligned1[idx] != 'T':
                    return
        elif len(n_index) == len(N_index):
            gap_N_index = [k for k in range(n_index[0],n_index[-1]+1) if k > 0 and aligned2[k] == 'N' and aligned2[k-1] == '-']
            for idx in gap_N_index:
                if aligned1[idx] != 'T':
                    return

    within_segment = i != len(seq2) and i != 0 and i-1 not in N_index and i not in N_index
    if j == 0 and i > 0:
        within_segment = False
    
    if matrix_name == "D":
        if i > 0 and j > 0:
            match_point = D[i - 1][j - 1] + score_match(seq1[j - 1], seq2[i - 1], pair_scores, T_N_misalignment_penalty)
            if D[i][j] == match_point:
                trace_back("D", D, P, Q, seq1, seq2, seq1[j - 1] + aligned1, seq2[i - 1] + aligned2, i - 1, j - 1, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

        if i == 0 and j > 0:
            trace_back("D", D, P, Q, seq1, seq2, seq1[0:j] + aligned1, "".join(["-"]*j) + aligned2, 0, 0, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

        if i > 0 and j == 0:
            trace_back("D", D, P, Q, seq1, seq2, "".join(["-"]*i) + aligned1, seq2[0:i] + aligned2, 0, 0, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

        if D[i][j] == P[i][j]:
            trace_back("P", D, P, Q, seq1, seq2, aligned1, aligned2, i, j, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

        if D[i][j] == Q[i][j]:
            trace_back("Q", D, P, Q, seq1, seq2, aligned1, aligned2, i, j, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

    if matrix_name == "P" and i > 0:
        if P[i][j] == P[i-1][j] + gap_extend1:
            trace_back("P", D, P, Q, seq1, seq2, "-" + aligned1, seq2[i - 1] + aligned2, i - 1, j, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)
        if P[i][j] == D[i-1][j] + gap_open1 + gap_extend1:
            trace_back("D", D, P, Q, seq1, seq2, "-" + aligned1, seq2[i - 1] + aligned2, i - 1, j, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

    if not within_segment and matrix_name == "Q" and j > 0:
        if Q[i][j] == Q[i][j-1] + gap_extend2:
            trace_back("Q", D, P, Q, seq1, seq2, seq1[j - 1] + aligned1, "-" + aligned2, i, j - 1, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)
        if Q[i][j] == D[i][j-1] + gap_open2 + gap_extend2:
            trace_back("D", D, P, Q, seq1, seq2, seq1[j - 1] + aligned1, "-" + aligned2, i, j - 1, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

    if within_segment and matrix_name == "Q" and j > 0:
        if Q[i][j] == Q[i][j-1] + gap_extend3:
            trace_back("Q", D, P, Q, seq1, seq2, seq1[j - 1] + aligned1, "-" + aligned2, i, j - 1, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)
        if Q[i][j] == D[i][j-1] + gap_open3 + gap_extend3:
            trace_back("D", D, P, Q, seq1, seq2, seq1[j - 1] + aligned1, "-" + aligned2, i, j - 1, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

def gotoh(seq1, seq2, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments, score_only=False):
    m = len(seq1) + 1
    n = len(seq2) + 1

    N_index = [i for i in range(len(seq2)) if seq2[i] == 'N']

    D = np.zeros((n,m))
    P = np.zeros((n,m))
    Q = np.zeros((n,m))

    for i in range(1,n):
        D[i][0] = gap_open1 + i*gap_extend1
        P[i][0] = -np.inf
        Q[i][0] = -np.inf

    for j in range(1,m):
        D[0][j] = gap_open2 + j*gap_extend2
        P[0][j] = -np.inf
        Q[0][j] = -np.inf

    for i in range(1, n):
        for j in range(1, m):
            within_segment = i-1 != n-2 and i > 0 and i-1 not in N_index and i not in N_index

            P[i][j] = max(P[i-1][j]+gap_extend1, D[i-1][j]+gap_open1+gap_extend1)
            if not within_segment:
                Q[i][j] = max(Q[i][j-1]+gap_extend2, D[i][j-1]+gap_open2+gap_extend2)
            else:
                Q[i][j] = max(Q[i][j-1]+gap_extend3, D[i][j-1]+gap_open3+gap_extend3)

            match_point = D[i - 1][j - 1] + score_match(seq1[j-1], seq2[i-1], pair_scores, T_N_misalignment_penalty)
            D[i][j] = max(match_point, P[i][j], Q[i][j])

    if not score_only:
        trace_back("D", D, P, Q, seq1, seq2, "", "", n - 1, m - 1, N_index, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

    return D[n-1][m-1]

def rescore(aligned1, aligned2, pair_scores):
    assert len(aligned1) == len(aligned2)

    score = 0
    for c1, c2 in zip(aligned1, aligned2):
        if c1 != "-" and c2 != "-":
            score += pair_scores[c1+c2]
    return score

def find_first_single_gap(seq):
    first_single_gap = None
    for i in range(len(seq)-1):
        if seq[i] == '-' and seq[i+1] == '-':
            break
        if seq[i] == '-' and seq[i+1] != '-':
            if seq[0] != 'T' and i != 0:
                first_single_gap = i
            if seq[0] == 'T' and i != 1:
                first_single_gap = i
            break
    return first_single_gap

def find_last_single_gap(seq):
    last_single_gap = None
    for i in range(len(seq)-1,0,-1):
        if seq[i] == '-' and seq[i-1] == '-':
            break
        if seq[i] == '-' and seq[i-1] != '-':
            if seq[-1] != 'T' and i != len(seq) - 1:
                last_single_gap = i
            if seq[-1] == 'T' and i != len(seq) - 2:
                last_single_gap = i
            break
    return last_single_gap

def realign_head_tail(alignment, pair_scores):
    aligned1, aligned2 = alignment

    if "-" not in aligned1:
        return (aligned1, aligned2)

    T_index = [0]
    for i, c in enumerate(aligned1):
        if c == 'T':
            T_index.append(i)
    T_index.append(len(aligned1))

    new_aligned1 = []
    for i in range(len(T_index)-1):
        ib = T_index[i]
        ie = T_index[i+1]
        if "-" not in aligned1[ib:ie]:
            new_aligned1.append(aligned1[ib:ie])
            continue

        first_single_gap = find_first_single_gap(aligned1[ib:ie])
        if first_single_gap is not None:
            first_single_gap += ib

        switch_head = False
        new_head = None
        if first_single_gap is not None:
            old_score = rescore(aligned1[ib:(first_single_gap+1)],aligned2[ib:(first_single_gap+1)],pair_scores)
            if ib == 0:
                new_score = rescore("-"+aligned1[ib:first_single_gap],aligned2[ib:(first_single_gap+1)],pair_scores)
                new_head = "-"+aligned1[ib:first_single_gap]
            else:
                new_score = rescore(aligned1[ib]+"-"+aligned1[ib+1:first_single_gap],aligned2[ib:(first_single_gap+1)],pair_scores)
                new_head = aligned1[ib]+"-"+aligned1[ib+1:first_single_gap]
            if new_score >= old_score - 2:
                switch_head = True

        last_single_gap = find_last_single_gap(aligned1[ib:ie])
        if last_single_gap is not None:
            last_single_gap += ib

        switch_tail = False
        new_tail = None
        if last_single_gap is not None and (last_single_gap != first_single_gap or (last_single_gap == first_single_gap and switch_head == False)):
            old_score = rescore(aligned1[last_single_gap:ie],aligned2[last_single_gap:ie],pair_scores)
            new_score = rescore(aligned1[(last_single_gap+1):ie]+"-",aligned2[last_single_gap:ie],pair_scores)
            new_tail = aligned1[(last_single_gap+1):ie]+"-"
            if new_score >= old_score - 2:
                switch_tail = True

        if switch_head and switch_tail:
            new_chain = new_head + aligned1[first_single_gap+1:last_single_gap] + new_tail
        elif switch_head:
            new_chain = new_head + aligned1[first_single_gap+1:ie]
        elif switch_tail:
            new_chain = aligned1[ib:last_single_gap] + new_tail
        else:
            new_chain = aligned1[ib:ie]
        new_aligned1.append(new_chain)
    new_aligned1 = "".join(new_aligned1)

    return (new_aligned1, aligned2)

def punish_by_closeness(alignment, closeness_matrix, verbose=False):
    aligned1, aligned2 = alignment
    penalty = 0.0

    gaps = re.finditer(r'-+', aligned1)
    gap_indices = [(gap.start(), gap.end() - 1) for gap in gaps]
    for index in gap_indices:
        ib, ie = index
        if ib == 0:
            continue
        if ie == len(aligned1) - 1:
            continue
        if aligned1[ib-1] not in ["A","G","C","U"]:
            continue
        if aligned1[ie+1] not in ["A","G","C","U"]:
            continue
        new_ib = None
        for j in range(ib-1,-1,-1):
            if aligned2[j] in ["A", "G", "C", "U", "X"]:
                new_ib = j + 1
                break
        assert new_ib is not None
        new_ie = None
        for j in range(ie+1,len(aligned1)):
            if aligned2[j] in ["A", "G", "C", "U", "X"]:
                new_ie = j - 1
                break
        assert new_ie is not None
        if "T" in aligned1[(new_ib-1):(new_ie+2)]:
            continue
        nt1_idx = new_ib - 1 - aligned2[0:new_ib].count('-')
        nt2_idx = new_ie + 1 - aligned2[0:(new_ie+1)].count('-')
        true_gap_num = new_ie - new_ib + 2 - aligned1[(new_ib-1):(new_ie+2)].count('-')
        least_gap_num = round(closeness_matrix[nt1_idx][nt2_idx]/6.)
        if least_gap_num > true_gap_num:
            penalty += (least_gap_num - true_gap_num) * 2
        if verbose:
            print("closeness penalty",new_ib,new_ie,true_gap_num,least_gap_num,flush=True)

    gaps = re.finditer(r'-+', aligned2)
    gap_indices = [(gap.start(), gap.end() - 1) for gap in gaps]
    for index in gap_indices:
        ib, ie = index
        if ib == 0:
            continue
        if ie == len(aligned2) - 1:
            continue
        new_ib = None
        for j in range(ib-1,-1,-1):
            if aligned2[j] in ["A", "G", "C", "U", "X"]:
                new_ib = j + 1
                break
        assert new_ib is not None
        new_ie = None
        for j in range(ie+1,len(aligned2)):
            if aligned2[j] in ["A", "G", "C", "U", "X"]:
                new_ie = j - 1
                break
        assert new_ie is not None
        if "T" in aligned1[(new_ib-1):(new_ie+2)]:
            continue
        nt1_idx = new_ib - 1 - aligned2[0:new_ib].count('-')
        nt2_idx = new_ie + 1 - aligned2[0:(new_ie+1)].count('-')
        true_gap_num = new_ie - new_ib + 2 - aligned1[(new_ib-1):(new_ie+2)].count('-')
        least_gap_num = round(closeness_matrix[nt1_idx][nt2_idx]/6.)
        if least_gap_num > true_gap_num:
            penalty += (least_gap_num - true_gap_num) * 2
        if verbose:
            print("closeness penalty",new_ib,new_ie,true_gap_num,least_gap_num,flush=True)

    N_index = []
    for i in range(len(aligned2)-1):
        if aligned2[i] == "N" and aligned2[i-1] in "AGCUXP" and aligned2[i+1] in "AGCUXP":
            N_index.append(i)
    for idx in N_index:
        ib = idx
        ie = idx
        new_ib = None
        for j in range(ib-1,-1,-1):
            if aligned2[j] in ["A", "G", "C", "U", "X"]:
                new_ib = j + 1
                break
        assert new_ib is not None
        new_ie = None
        for j in range(ie+1,len(aligned2)):
            if aligned2[j] in ["A", "G", "C", "U", "X"]:
                new_ie = j - 1
                break
        assert new_ie is not None
        if "T" in aligned1[(new_ib-1):(new_ie+2)]:
            continue
        if "-" in aligned1[(new_ib-1):(new_ie+2)]:
            continue
        nt1_idx = new_ib - 1 - aligned2[0:new_ib].count('-')
        nt2_idx = new_ie + 1 - aligned2[0:(new_ie+1)].count('-')
        true_gap_num = new_ie - new_ib + 2 - aligned1[(new_ib-1):(new_ie+2)].count('-')
        least_gap_num = round(closeness_matrix[nt1_idx][nt2_idx]/6.)
        if least_gap_num > true_gap_num:
            penalty += (least_gap_num - true_gap_num) * 2
        if verbose:
            print("closeness penalty",new_ib,new_ie,true_gap_num,least_gap_num,flush=True)

    return penalty

def global_alignment(seq1, seq2, closeness_matrix):
    pair_scores = {"AA": 1, "AG": 1, "AX": -1, "AC": -1, "AU": -1, "AT": -1, "AN": -3, "AP": -2,
            "CA": -1, "CG": -1, "CX": -1, "CC": 1, "CU": 1, "CT": -1, "CN": -3, "CP": -2,
            "UA": -1, "UG": -1, "UX": -1, "UC": 1, "UU": 1, "UT": -1, "UN": -3, "UP": -2,
            "GA": 1, "GG": 1, "GC": -1, "GU": -1, "GX": -1, "GT": -1, "GN": -3, "GP": -2,
            "XA": -1, "XG": -1, "XC": -1, "XU": -1, "XX": -1, "XT": -1, "XN": -3, "XP": -2,
            "TA": -1, "TG": -1, "TC": -1, "TU": -1, "TX": -1, "TT": 2, "TN": 5, "TP": -2,
            "NA": -3, "NG": -3, "NC": -3, "NU": -3, "NX": -3, "NT": 5, "NN": 2,
            "PA": -2, "PG": -2, "PC": -2, "PU": -2, "PT": -2
            }
    T_N_misalignment_penalty = -5

    gap_open1 = -2
    gap_extend1 = -2
    gap_open2 = -2
    gap_extend2 = -2
    gap_open3 = -5
    gap_extend3 = -4

    alignments = []
    scores = []
    
    score = gotoh(seq1, seq2, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments)

    new_alignments = []
    for alignment in alignments:
        new_alignment = realign_head_tail(alignment, pair_scores)
        new_alignments.append(new_alignment)
        if closeness_matrix is not None:
            penalty = punish_by_closeness(new_alignment, closeness_matrix)
        else:
            penalty = 0.0
        bonus = 0
        for c1, c2 in zip(new_alignment[0], new_alignment[1]):
            if c1 == "T" and c2 == "-":
                bonus += -(T_N_misalignment_penalty+gap_open2)
        new_score = score - penalty + bonus
        scores.append(new_score)
    return scores, new_alignments

def get_alignment_score(seq1, seq2, closeness_matrix):
    pair_scores = {"AA": 1, "AG": 1, "AX": -1, "AC": -1, "AU": -1, "AT": -1, "AN": -3, "AP": -2,
            "CA": -1, "CG": -1, "CX": -1, "CC": 1, "CU": 1, "CT": -1, "CN": -3, "CP": -2,
            "UA": -1, "UG": -1, "UX": -1, "UC": 1, "UU": 1, "UT": -1, "UN": -3, "UP": -2,
            "GA": 1, "GG": 1, "GC": -1, "GU": -1, "GX": -1, "GT": -1, "GN": -3, "GP": -2,
            "XA": -1, "XG": -1, "XC": -1, "XU": -1, "XX": -1, "XT": -1, "XN": -3, "XP": -2,
            "TA": -1, "TG": -1, "TC": -1, "TU": -1, "TX": -1, "TT": 2, "TN": 5, "TP": -2,
            "NA": -3, "NG": -3, "NC": -3, "NU": -3, "NX": -3, "NT": 5, "NN": 2,
            "PA": -2, "PG": -2, "PC": -2, "PU": -2, "PT": -2
            }
    T_N_misalignment_penalty = -5

    gap_open1 = -2
    gap_extend1 = -2
    gap_open2 = -2
    gap_extend2 = -2
    gap_open3 = -5
    gap_extend3 = -4

    alignments = []
    score_only = True
    
    score = gotoh(seq1, seq2, pair_scores, gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty, alignments, score_only)

    return score
