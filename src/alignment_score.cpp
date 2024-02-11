#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <string>

using namespace std;

int score_match(char a, char b, map<string, int>& pair_scores, int T_N_misalignment_penalty) {
  if (a == 'T' && b != 'N' && b != 'P') {
    return T_N_misalignment_penalty; 
  }
  
  string pair = "";
  pair += a;
  pair += b;
  
  return pair_scores[pair];
}

int gotoh_score(const string& seq1, const string& seq2, 
                map<string, int>& pair_scores, 
                int gap_open1, int gap_extend1, 
                int gap_open2, int gap_extend2,
                int gap_open3, int gap_extend3,
                int T_N_misalignment_penalty) {
  
  int m = seq1.length() + 1;
  int n = seq2.length() + 1;
  
  vector<int> N_index;
  for (int i = 0; i < seq2.length(); i++) {
    if (seq2[i] == 'N') {
      N_index.push_back(i); 
    }
  }
  
  vector<vector<int> > D(n, vector<int>(m, 0));
  vector<vector<int> > P(n, vector<int>(m, 0)); 
  vector<vector<int> > Q(n, vector<int>(m, 0));
  
  for (int i = 1; i < n; i++) {
    D[i][0] = gap_open1 + i * gap_extend1;
    P[i][0] = -1000000;
    Q[i][0] = -1000000;
  }
  
  for (int j = 1; j < m; j++) {
    D[0][j] = gap_open2 + j * gap_extend2;
    P[0][j] = -1000000;
    Q[0][j] = -1000000;
  }
  
  for (int i = 1; i < n; i++) {
    for (int j = 1; j < m; j++) {
      P[i][j] = max(P[i-1][j] + gap_extend1, D[i-1][j] + gap_open1 + gap_extend1);
      
      int match_point = D[i-1][j-1] + score_match(seq1[j-1], seq2[i-1], pair_scores, T_N_misalignment_penalty);
      
      bool within_segment = i-1 != n-2 && i > 0 && 
                          find(N_index.begin(), N_index.end(), i-1) == N_index.end() &&
                          find(N_index.begin(), N_index.end(), i) == N_index.end();
      
      if (!within_segment){
          Q[i][j] = max(Q[i][j-1] + gap_extend2, D[i][j-1] + gap_open2 + gap_extend2);
      }
      else{
          Q[i][j] = max(Q[i][j-1] + gap_extend3, D[i][j-1] + gap_open3 + gap_extend3);
      }

      D[i][j] = max(match_point, max(P[i][j], Q[i][j]));
    }
  }
  
  return D[n-1][m-1];
}

extern "C" {
int get_alignment_score(const char* seq1, const char* seq2) {
  string seq_1(seq1);
  string seq_2(seq2);

  map<string, int> pair_scores = {
    {"AA", 1}, {"AG", 1}, {"AX", -1}, {"AC", -1}, {"AU", -1}, {"AT", -1}, {"AN", -3},
    {"CA", -1}, {"CG", -1}, {"CX", -1}, {"CC", 1}, {"CU", 1}, {"CT", -1}, {"CN", -3}, 
    {"UA", -1}, {"UG", -1}, {"UX", -1}, {"UC", 1}, {"UU", 1}, {"UT", -1}, {"UN", -3},
    {"GA", 1}, {"GG", 1}, {"GC", -1}, {"GU", -1}, {"GX", -1}, {"GT", -1}, {"GN", -3},
    {"XA", -1}, {"XG", -1}, {"XC", -1}, {"XU", -1}, {"XX", -1}, {"XT", -1}, {"XN", -3}, 
    {"TA", -1}, {"TG", -1}, {"TC", -1}, {"TU", -1}, {"TX", -1}, {"TT", 2}, {"TN", 5},
    {"NA", -3}, {"NG", -3}, {"NC", -3}, {"NU", -3}, {"NX", -3}, {"NT", 5}, {"NN", 2},
    {"AP", -2}, {"GP", -2}, {"CP", -2}, {"UP", -2}, {"TP", -2},
    {"PA", -2}, {"PG", -2}, {"PC", -2}, {"PU", -2}, {"PT", -2},
  };

  int T_N_misalignment_penalty = -5;
  
  int gap_open1, gap_extend1, gap_open2, gap_extend2, gap_open3, gap_extend3;

  gap_open1 = -2;
  gap_extend1 = -2;
  gap_open2 = -2; 
  gap_extend2 = -2;
  gap_open3 = -5; 
  gap_extend3 = -4;

  int score = gotoh_score(seq_1, seq_2, pair_scores, gap_open1, gap_extend1, 
                          gap_open2, gap_extend2, gap_open3, gap_extend3, T_N_misalignment_penalty);
                          
  return score;
}
}
