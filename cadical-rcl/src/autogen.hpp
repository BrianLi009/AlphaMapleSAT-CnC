#ifndef AUTOGEN_HPP
#define AUTOGEN_HPP

#include "internal.hpp"
#include <set>
#include "../../nauty2_8_8/nauty.h"
#include "../../nauty2_8_8/naugroup.h"
#include <unordered_set>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#define l_False 0
#define l_True 1
#define l_Undef 2

#define MAX(X,Y) ((X) > (Y)) ? (X) : (Y)
#define MIN(X,Y) ((X) > (Y)) ? (Y) : (X)

const int MAXORDER = 64;  // This should be at least 8, preferably a power of 2

namespace std {
    template<>
    struct hash<vector<graph>> {
        size_t operator()(const vector<graph>& v) const {
            size_t seed = v.size();
            for(auto& i : v) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

class SymmetryBreaker : public CaDiCaL::ExternalPropagator {
    CaDiCaL::Solver * solver;
    std::vector<std::vector<int>> new_clauses;
    std::deque<std::vector<int>> current_trail;
    
    int * assign;
    bool * fixed;
    int * colsuntouched;
    int n = 0;
    int unembeddable_check = 0;
    long sol_count = 0;
    int num_edge_vars = 0;
    std::set<unsigned long> canonical_hashes[MAXORDER];
    std::set<unsigned long> solution_hashes;
    int learned_clauses_count;
    std::unordered_set<std::string> generated_clauses;

    // Profiling members
    long long larger_time = 0;
    long long smaller_time = 0;
    long larger_calls = 0;
    long smaller_calls = 0;
    long larger_clauses_generated = 0;
    long smaller_clauses_generated = 0;
    long total_larger_perms = 0;
    long total_smaller_perms = 0;

    std::unordered_set<int> unit_clause_cache;

    // New members for nauty
    DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    setword workspace[100];
    int lab[MAXORDER], ptn[MAXORDER], orbits[MAXORDER];
    graph g[MAXORDER*MAXORDER];

    private:
    // New members for self-adjusting perm_cutoff
    std::vector<long long> total_perms_checked;
    std::vector<long> total_checks;
    std::vector<bool> cutoff_calibrated;
    static const int CALIBRATION_SAMPLES = 1000;

    std::vector<graph> current_canonical_form;  // Make sure this is std::vector<graph>, not std::vector<char>

    std::unordered_set<std::vector<graph>> unique_canonical_forms;

    // Add these new members
    int CANONICITY_CHECK_FREQUENCY[MAXORDER];
    const int MIN_ORDER_FOR_FREQUENCY_CHECK = 5;
    long partial_assignment_counts[MAXORDER];

    std::vector<std::vector<long long>> perm_counts_during_calibration;

    std::mutex canonicity_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_threads;
    std::atomic<int> active_threads;
    int num_threads;
    // Add this declaration
    void remove_possibilities(int k, int pn[], const std::vector<int>& orbits);

public:
    SymmetryBreaker(CaDiCaL::Solver * s, int order, int uc);
    ~SymmetryBreaker ();
    void notify_assignment(int lit, bool is_fixed);
    void notify_new_decision_level ();
    void notify_backtrack (size_t new_level);
    bool cb_check_found_model (const std::vector<int> & model);
    bool cb_has_external_clause ();
    int cb_add_external_clause_lit ();
    int cb_decide ();
    int cb_propagate ();
    int cb_add_reason_clause_lit (int plit);
    bool is_canonical(int k, int p[], int& x, int& y, int& i);
    bool has_mus_subgraph(int k, int* P, int* p, int g);
    
    void generate_stronger_blocking_clause(int k, const std::vector<int>& partial_assignment, std::vector<int>& blocking_clause);
    
    void generate_blocking_clause_larger(int k, int p[], int x, int y);
    void generate_blocking_clause_smaller(int k, int p[], int x, int y);
    
    void print_stats();

    std::pair<std::vector<int>, std::vector<graph>> compute_orbits_and_canonical_form(int k);
    std::vector<char> convert_assignment_to_graph6(int k);
    std::vector<char> compute_canonical_graph6(int k);  // New method

private:
    // You might want to add a method to set custom frequencies if needed in the future
    // void set_canonicity_check_frequency(int order, int frequency);
};
#endif // AUTOGEN_HPP