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
#include <chrono>
#include <unordered_map>
#include <utility>

#define l_False 0
#define l_True 1
#define l_Undef 2

#define MAX(X,Y) ((X) > (Y)) ? (X) : (Y)
#define MIN(X,Y) ((X) > (Y)) ? (Y) : (X)

const int MAXORDER = 40;
constexpr int MAX_GRAPH_SIZE = 320;  // Maximum graph size, matching MAXN in canonize.c
constexpr int BUFFER_SIZE = 128 * 1024;  // 128 KB buffer for reading canonize output

// New definitions for nauty integration
#define WORDSIZE 64
#define MAXNV 64

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

#define DEFAULT_UNEMBEDDABLE_CHECK 17

// Add these typedefs near the top of the file
typedef int truth_value_t;
typedef std::vector<std::vector<truth_value_t>> adjacency_matrix_t;
typedef std::vector<std::pair<truth_value_t, std::pair<int, int>>> forbidden_graph_t;

// Add these constants
#define PRESENT_LABEL "1"
#define ABSENT_LABEL  "0"
#define UNKNOWN_LABEL  "u"

class SymmetryBreaker : public CaDiCaL::ExternalPropagator {
    CaDiCaL::Solver * solver;
    std::vector<std::vector<int>> new_clauses;
    std::deque<std::vector<int>> current_trail;
    
    int * assign;
    bool * fixed;
    int * colsuntouched;
    int n = 0;
    long sol_count = 0;
    int num_edge_vars = 0;
    std::set<unsigned long> canonical_hashes[MAXORDER];
    std::set<unsigned long> solution_hashes;
    int learned_clauses_count;
    std::unordered_set<std::string> generated_clauses;

    // Add members for unembeddable graph checking
    int unembeddable_check;
    long muscount;
    long muscounts[17];
    double mustime;

    std::unordered_set<int> unit_clause_cache;

    // New members for nauty
    statsblk stats;
    setword workspace[100];
    int lab[MAXNV], ptn[MAXNV], orbits[MAXNV];
    graph g[MAXNV*MAXNV];

    // New members for RCL-binary-based approach
    long canonize_calls;
    long long total_canonize_time;

    // Add the options structure
    DEFAULTOPTIONS_GRAPH(options);

    // Add to class SymmetryBreaker's private members:
    bool print_statement;

    int partition_size;  // Add this new member variable

    // Add this to the SymmetryBreaker class private members:
    std::vector<adjacency_matrix_t> forbiddenSubgraphs;
    std::vector<adjacency_matrix_t> forbiddenInducedSubgraphs;

    // Add this to store the master graph
    adjacency_matrix_t masterGraph;
    std::vector<int> masterGraphLabels;

    // Add these counters
    int subgraph_count = 0;       // Number of partial assignments that are subgraphs
    int non_subgraph_count = 0;   // Number of partial assignments that are not subgraphs

    // Add this flag
    bool use_master_graph = false; // Whether to use the master graph for subgraph checking

    // Add this flag to control solution printing
    bool no_print = false;

    // Add these counters
    long orthogonality_violations = 0;

    // Add these to the private members section
    std::unordered_set<size_t> seen_orthogonality_assignments;
    long orthogonality_checks = 0;
    long orthogonality_skipped = 0;

    // These were previously constants, now they're class members
    int SMALL_GRAPH_ORDER;
    int EDGE_CUTOFF;
    int NUM_PREDEFINED_VECTORS;
    
    // Dynamic array for predefined vectors
    std::vector<std::vector<int>> predefined_vectors;

private:
    // Add this new function declaration
    bool is_subgraph_of_master(const adjacency_matrix_t& graph);

    std::unordered_set<size_t> seen_partial_assignments;

    // Add these new function declarations
    adjacency_matrix_t string_to_adjacency_matrix(const std::string& input, int k);
    std::string adjacency_matrix_to_string(const adjacency_matrix_t& matrix);

    // Add these function declarations to the private section
    std::vector<int> cross_product(const std::vector<int>& a, const std::vector<int>& b);

public:
    SymmetryBreaker(CaDiCaL::Solver * s, int order, int uc = 0, int ps = 0);
    ~SymmetryBreaker();

    // Add this new function declaration
    void load_master_graph(const std::string& filepath);

    void notify_assignment(int lit, bool is_fixed);
    void notify_new_decision_level();
    void notify_backtrack(size_t new_level);
    bool cb_check_found_model(const std::vector<int> & model);
    bool cb_has_external_clause();
    int cb_add_external_clause_lit();
    int cb_decide();
    int cb_propagate();
    int cb_add_reason_clause_lit(int plit);
    
    void print_stats();

    // Add to public members:
    void set_print_statement(bool value) { print_statement = value; }

    // Add this function declaration
    void print_subgraph_statistics();

    // Add this function
    void set_use_master_graph(bool use_master) { use_master_graph = use_master; }

    // Add these function declarations
    int dot_product(const std::vector<int>& v1, const std::vector<int>& v2);
    std::vector<int> check_orthogonality_constraints(const std::string& assignment, int k);

    // Add this function declaration to the public section
    void set_use_orthogonality(bool value);

    // Add this function to control solution printing
    void set_no_print(bool value);

    // Add efficient solution printing without string allocation
    void print_solution_direct(long solution_number);

private:
    std::string convert_assignment_to_string(int k);
    void stringToGraph(const std::string& input, graph* g, int n, int m);
    std::vector<int> call_RCL_binary(const std::string& input, int k);
    std::vector<int> block_extension(int k);
    std::vector<int> generate_complex_blocking_clause(const std::string& assignment, int k);
    std::vector<int> generate_naive_blocking_clause(const std::string& assignment, bool only_true_edges = false);
    std::vector<int> check_non_decreasing_degrees(const std::string& assignment, int k);
    std::string extract_submatrix(const std::string& input, int k);

    void Getcan_Rec(graph g[MAXNV], int n, int can[], int orbits[]);
    bool isCanonical(const std::string& input);
    void initNautyOptions();
    void printGraph(const graph* g, int n, int m);

    // New function declaration
    std::pair<std::string, int> find_smallest_non_canonical_submatrix(const std::string& input);
    bool has_mus_subgraph(int k, int* P, int* p, int g);
};

#endif // AUTOGEN_HPP