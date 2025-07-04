#include "RCL.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unordered_set>
#include <cmath>
#include "unembeddable_graphs.h"

#define DEBUG_PRINT(x) if (print_statement) { std::cout << "DEBUG: " << x << std::endl; }

#define WORDSIZE 64
#define MAXNV 64
#define DEFAULT_PARTITION_SIZE 0

// Default number of unembeddable graphs to check
#define DEFAULT_UNEMBEDDABLE_CHECK 13

// Add these two variables for optimization
//const int SMALL_GRAPH_ORDER = 13;
//const int EDGE_CUTOFF = 78;  // 13 choose 2 = 78

// Add these constants for the predefined vectors at the top of the file with other constants
//const int NUM_PREDEFINED_VECTORS = 13;

// Remove these constants as they'll be set dynamically
// const int SMALL_GRAPH_ORDER = 25;
// const int EDGE_CUTOFF = 300;
// const int NUM_PREDEFINED_VECTORS = 25;

const int VECTOR_DIMENSION = 3;

long canon = 0;
long noncanon = 0;
double canontime = 0;
double noncanontime = 0;
long canonarr[MAXNV] = {};
long noncanonarr[MAXNV] = {};
double canontimearr[MAXNV] = {};
double noncanontimearr[MAXNV] = {};
long muscount = 0;
long muscounts[17] = {0};  // Initialize all elements to zero explicitly
double mustime = 0;
// Add at the top with other file pointers
// FILE* solution_file = fopen("solutions.txt", "w");

// Add at the top with other time variables
double subgraph_check_time = 0;
double orthogonality_check_time = 0;
long orthogonality_violations = 0;

// Add this to the top with other counters
std::unordered_set<size_t> seen_orthogonality_assignments;
long orthogonality_checks = 0;
long orthogonality_skipped = 0;

// Add this to the top with other boolean flags
bool use_orthogonality = false;

SymmetryBreaker::SymmetryBreaker(CaDiCaL::Solver* s, int order, int uc, int ps) : solver(s) {
    // Set print_statement to true at the beginning of the constructor
    print_statement = true;
    
    DEBUG_PRINT("Entered RCL SymmetryBreaker constructor");
    if (order == 0) {
        std::cout << "c Need to provide order to use programmatic code" << std::endl;
        return;
    }
    
    // Set default unembeddable check if not specified
    if (uc < 0) {
        // Not specified on command line, use default
        uc = DEFAULT_UNEMBEDDABLE_CHECK;
        std::cout << "c Using default unembeddable subgraphs check (" << uc << " graphs)" << std::endl;
    } else if (uc == 0) {
        // If explicitly set to 0, disable the check
        std::cout << "c Disabling unembeddable subgraphs check" << std::endl;
    } else {
        std::cout << "c Checking for " << uc << " unembeddable subgraphs" << std::endl;
    }
    
    // Store the unembeddable check value
    unembeddable_check = uc;
    
    // Set default partition size if not specified
    if (ps == 0) {
        ps = DEFAULT_PARTITION_SIZE;
        std::cout << "c Using default partition size (" << ps << ")" << std::endl;
    } else {
        std::cout << "c Using partition size " << ps << std::endl;
    }
    
    n = order;
    num_edge_vars = n*(n-1)/2;
    partition_size = ps;
    
    // Set SMALL_GRAPH_ORDER and NUM_PREDEFINED_VECTORS to partition_size
    SMALL_GRAPH_ORDER = partition_size;
    NUM_PREDEFINED_VECTORS = partition_size;
    
    // Set EDGE_CUTOFF to partition_size choose 2
    EDGE_CUTOFF = partition_size * (partition_size - 1) / 2;
    
    // Add a clear print statement showing both order and partition size
    std::cout << "c Running with order = " << n << " and partition size = " << partition_size << std::endl;
    std::cout << "c SMALL_GRAPH_ORDER = " << SMALL_GRAPH_ORDER << ", NUM_PREDEFINED_VECTORS = " << NUM_PREDEFINED_VECTORS << std::endl;
    std::cout << "c EDGE_CUTOFF = " << EDGE_CUTOFF << " (partition_size choose 2)" << std::endl;
    
    DEBUG_PRINT("Allocating memory");
    assign = new int[num_edge_vars];
    fixed = new bool[num_edge_vars];
    colsuntouched = new int[n];
    
    DEBUG_PRINT("Connecting external propagator");
    solver->connect_external_propagator(this);
    
    DEBUG_PRINT("Initializing arrays");
    for (int i = 0; i < num_edge_vars; i++) {
        assign[i] = l_Undef;
        fixed[i] = false;
        solver->add_observed_var(i+1);
    }
    std::fill_n(colsuntouched, n, 0);
    
    current_trail.push_back(std::vector<int>());
    
    learned_clauses_count = 0;
    canonize_calls = 0;
    total_canonize_time = 0;
    
    seen_partial_assignments.clear();
    
    initNautyOptions();

    // Load the master graph
    //load_master_graph("cadical-rcl/data/SI-C-c1-labeled-37.lad");
    //load_master_graph("cadical-rcl/data/SI-C-c2-labeled-853-13.lad");
    load_master_graph("cadical-rcl/data/SI-C-c2-labeled-853-25.lad");

    // Reset counters
    muscount = 0;
    for (int i = 0; i < 17; i++) {
        muscounts[i] = 0;
    }

    // Initialize use_orthogonality to false by default
    use_orthogonality = false;

    // Initialize predefined vectors
    // OLD 13-vertex vectors - COMMENTED OUT
    /*
    predefined_vectors = {
        {-1, 1, 1},   // v1
        {1, -1, 1},   // v2
        {1, 1, -1},   // v3
        {1, 1, 1},    // v4
        {1, 0, 0},    // v5
        {0, 1, 0},    // v6
        {0, 0, 1},    // v7
        {1, -1, 0},   // v8
        {1, 0, -1},   // v9
        {0, 1, -1},   // v10
        {0, 1, 1},    // v11
        {1, 0, 1},    // v12
        {1, 1, 0}     // v13
    };
    */
    
    // NEW 25-vertex vectors based on canonical ordering
    predefined_vectors = {
        {-2, 1, 1},   // v1  -> Position 1  -> Vertex 16
        {1, -2, 1},   // v2  -> Position 2  -> Vertex 23
        {-1, 2, 1},   // v3  -> Position 3  -> Vertex 20
        {2, -1, 1},   // v4  -> Position 4  -> Vertex 15
        {-1, 1, 2},   // v5  -> Position 5  -> Vertex 25
        {2, 1, -1},   // v6  -> Position 6  -> Vertex 17
        {1, 2, 1},    // v7  -> Position 7  -> Vertex 24
        {1, -1, 2},   // v8  -> Position 8  -> Vertex 22
        {1, 2, -1},   // v9  -> Position 9  -> Vertex 18
        {2, 1, 1},    // v10 -> Position 10 -> Vertex 14
        {1, 1, -2},   // v11 -> Position 11 -> Vertex 21
        {1, 1, 2},    // v12 -> Position 12 -> Vertex 19
        {1, 0, 0},    // v13 -> Position 13 -> Vertex 1
        {0, 1, 0},    // v14 -> Position 14 -> Vertex 2
        {1, -1, 0},   // v15 -> Position 15 -> Vertex 9
        {-1, 1, 1},   // v16 -> Position 16 -> Vertex 10
        {1, -1, 1},   // v17 -> Position 17 -> Vertex 11
        {0, 0, 1},    // v18 -> Position 18 -> Vertex 3
        {1, 1, -1},   // v19 -> Position 19 -> Vertex 12
        {1, 0, -1},   // v20 -> Position 20 -> Vertex 7
        {0, 1, -1},   // v21 -> Position 21 -> Vertex 4
        {0, 1, 1},    // v22 -> Position 22 -> Vertex 5
        {1, 1, 1},    // v23 -> Position 23 -> Vertex 13
        {1, 0, 1},    // v24 -> Position 24 -> Vertex 6
        {1, 1, 0}     // v25 -> Position 25 -> Vertex 8
    };
    
    // Ensure we have enough predefined vectors
    while (predefined_vectors.size() < NUM_PREDEFINED_VECTORS) {
        // Add more vectors if needed
        predefined_vectors.push_back({1, 1, 1});
    }

    DEBUG_PRINT("SymmetryBreaker constructor completed");
}

SymmetryBreaker::~SymmetryBreaker() {
    // First disconnect the propagator
    solver->disconnect_external_propagator();
    
    // Free memory
    delete[] assign;
    delete[] fixed;
    delete[] colsuntouched;
    
    // Remove the solution file closing code
    // if (solution_file) {
    //     fclose(solution_file);
    //     solution_file = nullptr;
    // }
    
    // Force flush stdout before printing statistics
    fflush(stdout);
    
    // Print all statistics with explicit fflush after each section
    printf("Number of solutions   : %ld\n", sol_count);
    fflush(stdout);
    
    printf("Canonical subgraphs   : %-12ld   (%.0f /sec)\n", canon, canon > 0 ? canon/canontime : 0);
    fflush(stdout);
    
    for(int i=2; i<n; i++) {
        printf("          order %2d    : %-12ld   (%.0f /sec)\n", 
               i+1, 
               canonarr[i], 
               canonarr[i] > 0 ? canonarr[i]/canontimearr[i] : 0);
        fflush(stdout);
    }
    
    printf("Noncanonical subgraphs: %-12ld   (%.0f /sec)\n", noncanon, noncanon > 0 ? noncanon/noncanontime : 0);
    fflush(stdout);
    
    for(int i=2; i<n; i++) {
        printf("          order %2d    : %-12ld   (%.0f /sec)\n", 
               i+1, 
               noncanonarr[i], 
               noncanonarr[i] > 0 ? noncanonarr[i]/noncanontimearr[i] : 0);
        fflush(stdout);
    }
    
    printf("Canonicity checking   : %g s\n", canontime);
    printf("Noncanonicity checking: %g s\n", noncanontime);
    printf("Total canonicity time : %g s\n", canontime + noncanontime);
    fflush(stdout);
    
    if (unembeddable_check > 0) {
        printf("Unembeddable checking : %g s\n", mustime);
        fflush(stdout);
        
        for(int g=0; g<unembeddable_check; g++) {
            printf("        graph #%2d     : %-12ld\n", g, muscounts[g]);
            fflush(stdout);
        }
        
        printf("Total unembed. graphs : %ld\n", muscount);
        fflush(stdout);
    }

    // Print subgraph statistics
    print_subgraph_statistics();
    fflush(stdout);
    
    // Final flush to ensure everything is printed
    fflush(stdout);
}

std::string SymmetryBreaker::convert_assignment_to_string(int k) {
    const int size = k * (k - 1) / 2;
    std::string result;
    result.reserve(size);  // Pre-allocate exact size needed
    
    // Use more efficient character access
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            result.push_back((assign[j*(j-1)/2 + i] == l_True) ? '1' : '0');
        }
    }
    return result;
}

void SymmetryBreaker::stringToGraph(const std::string& input, graph* g, int n, int m) {
    int index = 0;
    DEBUG_PRINT("Converting string to graph. Input: " << input);
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            if (input[index++] == '1') {
                ADDONEEDGE(g, i, j, m);
                ADDONEEDGE(g, j, i, m);  // Ensure symmetry
                DEBUG_PRINT("Added edge between " << i << " and " << j);
            }
        }
    }
}

void SymmetryBreaker::Getcan_Rec(graph g[MAXNV], int n, int can[], int orbits[]) {
    int lab1[MAXNV], lab2[MAXNV], inv_lab1[MAXNV], ptn[MAXNV];
    int i, j, k;
    setword st;
    graph g2[MAXNV];
    int m = SETWORDSNEEDED(n);

    if (n == 1) {
        can[n-1] = n-1;
    } else {
        // Set up nauty options
        options.writeautoms = FALSE;
        options.writemarkers = FALSE;
        options.getcanon = TRUE;
        
        // Update to use the partition_size member variable
        if (n > partition_size) {
            options.defaultptn = FALSE;
            for (i = 0; i < n-1; i++) {
                if (i == partition_size-1) {
                    ptn[i] = 0;  // Mark end of first partition
                } else {
                    ptn[i] = 1;  // Same partition continues
                }
            }
            ptn[n-1] = 0;  // Last vertex always ends its partition
        } else {
            options.defaultptn = TRUE;  // Use default partition for small graphs
        }

        // Initialize lab array with identity permutation
        for (i = 0; i < n; i++) {
            lab1[i] = i;
        }

        nauty(g, lab1, ptn, NULL, orbits, &options, &stats, workspace, 50, m, n, g2);

        for (i = 0; i < n; i++)
            inv_lab1[lab1[i]] = i;
        for (i = 0; i <= n-2; i++) {
            j = lab1[i];
            st = g[j];
            g2[i] = 0;
            while (st) {
                k = FIRSTBIT(st);
                st ^= bit[k];
                k = inv_lab1[k];
                if (k != n-1)
                    g2[i] |= bit[k];
            }
        }
        Getcan_Rec(g2, n-1, lab2, orbits);
        for (i = 0; i <= n-2; i++)
            can[i] = lab1[lab2[i]];
        can[n-1] = lab1[n-1];
    }
}

bool SymmetryBreaker::isCanonical(const std::string& input) {
    int n = static_cast<int>(std::sqrt(2 * input.length() + 0.25) + 0.5);
    int m = SETWORDSNEEDED(n);
    
    DEBUG_PRINT("Checking canonicity for input: " << input);
    DEBUG_PRINT("Calculated n = " << n << ", m = " << m);

    graph g[MAXNV];
    for (int i = 0; i < n; i++) {
        EMPTYSET(GRAPHROW(g, i, m), m);
    }
    stringToGraph(input, g, n, m);

    DEBUG_PRINT("Original graph:");
    printGraph(g, n, m);

    int can[MAXNV];
    graph cang[MAXNV];
    
    Getcan_Rec(g, n, can, orbits);

    if (print_statement) {
        std::cout << "DEBUG: Canonical labeling:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "  " << can[i] << " -> " << i << std::endl;
        }
    }

    // Construct the canonical graph
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cang[i*m + j] = 0;
        }
        for (int j = 0; j < n; j++) {
            if (ISELEMENT(GRAPHROW(g, can[i], m), can[j])) {
                ADDONEEDGE(cang, i, j, m);
            }
        }
    }

    DEBUG_PRINT("Canonical graph:");
    printGraph(cang, n, m);

    // Compare the input graph with the canonical graph
    bool is_canonical = true;
    for (int i = 0; i < n && is_canonical; i++) {
        for (int j = 0; j < m && is_canonical; j++) {
            if (g[i*m + j] != cang[i*m + j]) {
                is_canonical = false;
            }
        }
    }

    DEBUG_PRINT("isCanonical result: " << (is_canonical ? "true" : "false"));
    return is_canonical;
}

bool SymmetryBreaker::has_mus_subgraph(int k, int* P, int* p, int g) {
    int pl[12]; // pl[k] contains the current list of possibilities for kth vertex
    int pn[13]; // pn[k] contains the initial list of possibilities for kth vertex
    pl[0] = (1 << k) - 1;
    pn[0] = (1 << k) - 1;
    int i = 0;

    while(1) {
        // If no possibilities for ith vertex then backtrack
        if(pl[i]==0) {
            while((pl[i] & (pl[i] - 1)) == 0) {
                i--;
                if(i==-1) {
                    return false;  // No permutations produce a matrix containing the gth submatrix
                }
            }
            pl[i] = pl[i] & ~(1 << p[i]);
        }

        p[i] = log2(pl[i] & -pl[i]); // Get index of rightmost high bit
        pn[i+1] = pn[i] & ~(1 << p[i]); // List of possibilities for (i+1)th vertex

        // Check if permuted matrix contains the gth submatrix
        bool result_known = false;
        for(int j=0; j<i; j++) {
            if(!mus[g][i*(i-1)/2+j]) continue;
            const int px = MAX(p[i], p[j]);
            const int py = MIN(p[i], p[j]);
            const int pj = px*(px-1)/2 + py;
            if(assign[pj] == l_False) {
                result_known = true;
                break;
            }
        }

        if(!result_known && ((i == 9 && g < 2) || (i == 10 && g < 7) || i == 11)) {
            // Found complete gth submatrix in p(M)
            for(int j=0; j<=i; j++) {
                P[p[j]] = j;
            }
            return true;
        }
        if(!result_known) {
            i++;
            pl[i] = pn[i];
        } else {
            pl[i] = pl[i] & ~(1 << p[i]);
        }
    }
}

std::vector<int> SymmetryBreaker::call_RCL_binary(const std::string& input, int k) {
    if (!isCanonical(input)) {
        if (print_statement) {
            std::cout << "DEBUG: Input is not canonical" << std::endl;
        }
        return generate_naive_blocking_clause(input);
    }
    
    // Check for unembeddable subgraphs
    if (unembeddable_check > 0) {
        int P[n];
        int p[12];
        for(int j=0; j<n; j++) P[j] = -1;
        
        const double before = CaDiCaL::absolute_process_time();
        for(int g=0; g<unembeddable_check; g++) {
            if (has_mus_subgraph(k, P, p, g)) {
                muscount++;
                muscounts[g]++;
                
                // Generate blocking clause for unembeddable subgraph
                std::vector<int> blocking_clause;
                int c = 0;
                for(int jj=0; jj<k; jj++) {
                    for(int ii=0; ii<jj; ii++) {
                        if(input[c] == '1' && P[jj] != -1 && P[ii] != -1) {
                            if((P[ii] < P[jj] && mus[g][P[ii] + P[jj]*(P[jj]-1)/2]) || 
                               (P[jj] < P[ii] && mus[g][P[jj] + P[ii]*(P[ii]-1)/2])) {
                                blocking_clause.push_back(-(c+1));
                            }
                        }
                        c++;
                    }
                }
                const double after = CaDiCaL::absolute_process_time();
                mustime += (after-before);
                
                if (print_statement) {
                    std::cout << "DEBUG: Found unembeddable subgraph #" << g << std::endl;
                }
                return blocking_clause;
            }
        }
    }
    
    if (print_statement) {
        std::cout << "DEBUG: Input is canonical" << std::endl;
    }
    return std::vector<int>();
}

// Helper function to extract a submatrix from the input string
std::string SymmetryBreaker::extract_submatrix(const std::string& input, int k) {
    std::string submatrix;
    int n = static_cast<int>(std::sqrt(2 * input.length() + 0.25) + 0.5);
    for (int j = 1; j < k; ++j) {
        for (int i = 0; i < j; ++i) {
            int index = j * (j - 1) / 2 + i;
            submatrix += input[index];
        }
    }
    return submatrix;
}

// Add this new function to check for non-decreasing degrees
std::vector<int> SymmetryBreaker::check_non_decreasing_degrees(const std::string& assignment, int k) {
    std::vector<int> blocking_clause;
    
    // Only proceed if we have at least 2 columns
    if (k < 2) return blocking_clause;
    
    // Count ones in the last two columns only
    int second_last_ones = 0;
    int last_ones = 0;
    
    // Count ones in second-last column (k-2)
    for (int i = 0; i < k-2; i++) {
        int var_index = (k-2)*(k-3)/2 + i;
        if (assignment[var_index] == '1') {
            second_last_ones++;
        }
    }
    
    // Count ones in last column (k-1)
    for (int i = 0; i < k-1; i++) {
        int var_index = (k-1)*(k-2)/2 + i;
        if (assignment[var_index] == '1') {
            last_ones++;
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Last two column degrees: " 
                  << second_last_ones << " " << last_ones << std::endl;
    }

    // Check if second-last column has more ones than last column
    if (second_last_ones > last_ones) {
        if (print_statement) {
            std::cout << "DEBUG: Columns " << k-2 << " and " << k-1 
                     << " violate non-decreasing order (" 
                     << second_last_ones << " > " << last_ones << ")" << std::endl;
        }
        
        // Add literals for second-last column
        for (int i = 0; i < k-2; i++) {
            int var_index = (k-2)*(k-3)/2 + i;
            int literal = assignment[var_index] == '1' ? -(var_index + 1) : (var_index + 1);
            blocking_clause.push_back(literal);
        }
        
        // Add literals for last column
        for (int i = 0; i < k-1; i++) {
            int var_index = (k-1)*(k-2)/2 + i;
            int literal = assignment[var_index] == '1' ? -(var_index + 1) : (var_index + 1);
            blocking_clause.push_back(literal);
        }

        if (print_statement) {
            std::cout << "DEBUG: Generated degree blocking clause: ";
            for (int lit : blocking_clause) {
                std::cout << lit << " ";
            }
            std::cout << std::endl;
        }
    }

    return blocking_clause;
}

// Add this function to compute the dot product of two vectors
int SymmetryBreaker::dot_product(const std::vector<int>& v1, const std::vector<int>& v2) {
    int result = 0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Add this function to compute the cross product of two 3D vectors
std::vector<int> SymmetryBreaker::cross_product(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(3);
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

// Modify the check_orthogonality_constraints function to correctly calculate edge indices
std::vector<int> SymmetryBreaker::check_orthogonality_constraints(const std::string& assignment, int k) {
    const double start_time = CaDiCaL::absolute_process_time();
    std::vector<int> blocking_clause;
    
    // Only proceed if we have at least 14 vertices (13 predefined + at least 1 to check)
    if (k <= NUM_PREDEFINED_VECTORS) {
        orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
        return blocking_clause;
    }
    
    // Check if we've already processed this assignment
    size_t hash_value = std::hash<std::string>{}(assignment);
    if (seen_orthogonality_assignments.find(hash_value) != seen_orthogonality_assignments.end()) {
        if (print_statement) {
            DEBUG_PRINT("Skipping already processed orthogonality assignment of size " << k);
        }
        orthogonality_skipped++;
        orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
        return blocking_clause;
    }
    
    // Add this assignment to the seen set
    seen_orthogonality_assignments.insert(hash_value);
    orthogonality_checks++;
    
    if (print_statement) {
        DEBUG_PRINT("Checking orthogonality constraints for " << k << " vertices");
    }
    
    // Convert assignment string to adjacency matrix
    adjacency_matrix_t matrix = string_to_adjacency_matrix(assignment, k);
    
    // Create vectors for all vertices
    std::vector<std::vector<int>> vectors(k, std::vector<int>(VECTOR_DIMENSION, 0));
    std::vector<bool> vector_assigned(k, false);
    
    // Track dependencies for each vertex's vector assignment
    std::vector<std::vector<std::pair<int, int>>> vector_dependencies(k);
    
    // Initialize the first 13 vectors with predefined values
    for (int i = 0; i < NUM_PREDEFINED_VECTORS; i++) {
        for (int j = 0; j < VECTOR_DIMENSION; j++) {
            vectors[i][j] = predefined_vectors[i][j];
        }
        vector_assigned[i] = true;
        // Predefined vectors have no dependencies
    }
    
    // For each vertex beyond the predefined ones, try to assign vectors
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        
        for (int v = NUM_PREDEFINED_VECTORS; v < k; v++) {
            // Skip if already assigned
            if (vector_assigned[v]) continue;
            
            // Find connected vertices that already have vectors assigned
            std::vector<int> connected_vertices;
            for (int u = 0; u < v; u++) {
                if (matrix[u][v] == l_True && vector_assigned[u]) {
                    connected_vertices.push_back(u);
                    if (connected_vertices.size() >= 2) break;
                }
            }
            
            // If we found at least two connected vertices, we can compute the cross product
            if (connected_vertices.size() >= 2) {
                int u1 = connected_vertices[0];
                int u2 = connected_vertices[1];
                
                if (print_statement) {
                    DEBUG_PRINT("Vertex " << v << " is connected to vertices " << u1 << " and " << u2);
                    DEBUG_PRINT("Vector for vertex " << u1 << ": [" << vectors[u1][0] << "," 
                              << vectors[u1][1] << "," << vectors[u1][2] << "]");
                    DEBUG_PRINT("Vector for vertex " << u2 << ": [" << vectors[u2][0] << "," 
                              << vectors[u2][1] << "," << vectors[u2][2] << "]");
                    
                    // Explain where these vectors came from
                    if (u1 < NUM_PREDEFINED_VECTORS) {
                        DEBUG_PRINT("Vector for vertex " << u1 << " is predefined");
                    } else {
                        DEBUG_PRINT("Vector for vertex " << u1 << " was derived from its connections:");
                        for (const auto& dep : vector_dependencies[u1]) {
                            DEBUG_PRINT("  - Connection to vertex " << dep.second << " (edge " << dep.first << ")");
                        }
                    }
                    
                    if (u2 < NUM_PREDEFINED_VECTORS) {
                        DEBUG_PRINT("Vector for vertex " << u2 << " is predefined");
                    } else {
                        DEBUG_PRINT("Vector for vertex " << u2 << " was derived from its connections:");
                        for (const auto& dep : vector_dependencies[u2]) {
                            DEBUG_PRINT("  - Connection to vertex " << dep.second << " (edge " << dep.first << ")");
                        }
                    }
                    
                    DEBUG_PRINT("Computing cross product to find vector for vertex " << v 
                              << " that is orthogonal to both connected vertices");
                }
                
                // Compute cross product of the two vectors
                std::vector<int> new_vector = cross_product(vectors[u1], vectors[u2]);
                
                if (print_statement) {
                    DEBUG_PRINT("Cross product result: [" << new_vector[0] << "," 
                              << new_vector[1] << "," << new_vector[2] << "]");
                }
                
                // Check if the cross product is zero (vectors are parallel)
                if (new_vector[0] == 0 && new_vector[1] == 0 && new_vector[2] == 0) {
                    if (print_statement) {
                        DEBUG_PRINT("Cross product is zero for vertex " << v);
                        DEBUG_PRINT("Vectors for vertices " << u1 << " and " << u2 << " are parallel or one is zero");
                        DEBUG_PRINT("This means there is no vector that can be orthogonal to both simultaneously");
                        DEBUG_PRINT("Therefore, vertex " << v << " cannot be connected to both " << u1 << " and " << u2);
                    }
                    
                    // Collect all dependencies for this violation
                    std::set<std::pair<int, int>> all_dependencies;
                    
                    // Add direct dependencies
                    int edge_index1 = v*(v-1)/2 + u1;
                    int edge_index2 = v*(v-1)/2 + u2;
                    all_dependencies.insert({edge_index1, u1});
                    all_dependencies.insert({edge_index2, u2});
                    
                    // Add transitive dependencies for u1 and u2
                    for (const auto& dep : vector_dependencies[u1]) {
                        all_dependencies.insert(dep);
                    }
                    for (const auto& dep : vector_dependencies[u2]) {
                        all_dependencies.insert(dep);
                    }
                    
                    if (print_statement) {
                        DEBUG_PRINT("Generated smart blocking clause for parallel vectors with all dependencies:");
                        DEBUG_PRINT("Dependency chain explanation:");
                        
                        for (const auto& dep : all_dependencies) {
                            int edge_var = dep.first;
                            int connected_vertex = dep.second;
                            int vertex1 = 0, vertex2 = 0;
                            
                            // Convert edge variable back to vertex pair
                            for (int j = 1; j < k; j++) {
                                for (int i = 0; i < j; i++) {
                                    int var_idx = j*(j-1)/2 + i;
                                    if (var_idx == edge_var) {
                                        vertex1 = i;
                                        vertex2 = j;
                                        break;
                                    }
                                }
                                if (vertex1 != 0 || vertex2 != 0) break;
                            }
                            
                            DEBUG_PRINT("  Literal " << -(edge_var + 1) << " blocks edge between vertices " 
                                      << vertex1 << " and " << vertex2);
                            
                            if (vertex1 < NUM_PREDEFINED_VECTORS && vertex2 < NUM_PREDEFINED_VECTORS) {
                                DEBUG_PRINT("    Both vertices have predefined vectors");
                            } else if (vertex1 < NUM_PREDEFINED_VECTORS) {
                                DEBUG_PRINT("    Vertex " << vertex1 << " has a predefined vector");
                            } else if (vertex2 < NUM_PREDEFINED_VECTORS) {
                                DEBUG_PRINT("    Vertex " << vertex2 << " has a predefined vector");
                            }
                        }
                        
                        DEBUG_PRINT("This blocking clause prevents configurations where parallel vectors would be required to be orthogonal");
                    }
                    
                    // Create blocking clause from all dependencies
                    for (const auto& dep : all_dependencies) {
                        blocking_clause.push_back(-(dep.first + 1));
                    }
                    
                    orthogonality_violations++;
                    orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
                    return blocking_clause;
                }
                
                // Assign the new vector
                vectors[v] = new_vector;
                vector_assigned[v] = true;
                made_progress = true;
                
                // Record dependencies - FIXED INDEX CALCULATION
                // The edge variable for vertices i and j (where i < j) is calculated as:
                // j*(j-1)/2 + i
                int edge_index1 = v*(v-1)/2 + u1;
                int edge_index2 = v*(v-1)/2 + u2;
                
                if (print_statement) {
                    DEBUG_PRINT("Edge between vertices " << u1 << " and " << v << " has variable index " << edge_index1);
                    DEBUG_PRINT("Edge between vertices " << u2 << " and " << v << " has variable index " << edge_index2);
                }
                
                vector_dependencies[v].push_back({edge_index1, u1});
                vector_dependencies[v].push_back({edge_index2, u2});
                
                // Add transitive dependencies
                for (const auto& dep : vector_dependencies[u1]) {
                    vector_dependencies[v].push_back(dep);
                }
                for (const auto& dep : vector_dependencies[u2]) {
                    vector_dependencies[v].push_back(dep);
                }
                
                if (print_statement) {
                    DEBUG_PRINT("Assigned vector [" << new_vector[0] << "," << new_vector[1] << "," 
                              << new_vector[2] << "] to vertex " << v);
                    DEBUG_PRINT("This vector is orthogonal to both vertex " << u1 << " and vertex " << u2);
                    DEBUG_PRINT("Dependencies for this vector assignment:");
                    DEBUG_PRINT("  - Direct connection to vertex " << u1 << " (edge " << edge_index1 << ")");
                    DEBUG_PRINT("  - Direct connection to vertex " << u2 << " (edge " << edge_index2 << ")");
                    
                    if (!vector_dependencies[u1].empty() || !vector_dependencies[u2].empty()) {
                        DEBUG_PRINT("  - Transitive dependencies:");
                        for (const auto& dep : vector_dependencies[u1]) {
                            DEBUG_PRINT("    - From vertex " << u1 << ": connection to vertex " 
                                      << dep.second << " (edge " << dep.first << ")");
                        }
                        for (const auto& dep : vector_dependencies[u2]) {
                            DEBUG_PRINT("    - From vertex " << u2 << ": connection to vertex " 
                                      << dep.second << " (edge " << dep.first << ")");
                        }
                    }
                    
                    DEBUG_PRINT("Now checking if this vector is orthogonal to all other connected vertices");
                }
                
                // Verify that this vector is orthogonal to all other connected vertices
                bool orthogonality_violation = false;
                int violating_vertex = -1;
                
                for (int u = 0; u < v; u++) {
                    if (matrix[u][v] == l_True && u != u1 && u != u2 && vector_assigned[u]) {
                        int dot = dot_product(vectors[v], vectors[u]);
                        if (print_statement) {
                            DEBUG_PRINT("Checking orthogonality with vertex " << u);
                            DEBUG_PRINT("Vector for vertex " << u << ": [" << vectors[u][0] << "," 
                                      << vectors[u][1] << "," << vectors[u][2] << "]");
                            DEBUG_PRINT("Dot product: " << dot << " (should be 0 for orthogonal vectors)");
                        }
                        
                        if (dot != 0) {
                            orthogonality_violation = true;
                            violating_vertex = u;
                            break;
                        }
                    }
                }
                
                if (orthogonality_violation) {
                    if (print_statement) {
                        DEBUG_PRINT("VIOLATION: Vector for vertex " << v << " is not orthogonal to vertex " << violating_vertex);
                        DEBUG_PRINT("This means vertex " << v << " cannot be connected to vertices " 
                                  << u1 << ", " << u2 << ", and " << violating_vertex << " simultaneously");
                        DEBUG_PRINT("The vector for vertex " << v << " was determined by its connections to vertices " 
                                  << u1 << " and " << u2);
                        DEBUG_PRINT("But this vector is not orthogonal to vertex " << violating_vertex);
                    }
                    
                    // Collect all dependencies for this violation
                    std::set<std::pair<int, int>> all_dependencies;
                    
                    // Add direct dependencies
                    int var_index1 = v*(v-1)/2 + u1;
                    int var_index2 = v*(v-1)/2 + u2;
                    int var_index3 = v*(v-1)/2 + violating_vertex;
                    all_dependencies.insert({var_index1, u1});
                    all_dependencies.insert({var_index2, u2});
                    all_dependencies.insert({var_index3, violating_vertex});
                    
                    // Add transitive dependencies for u1, u2, and violating_vertex
                    for (const auto& dep : vector_dependencies[u1]) {
                        all_dependencies.insert(dep);
                    }
                    for (const auto& dep : vector_dependencies[u2]) {
                        all_dependencies.insert(dep);
                    }
                    for (const auto& dep : vector_dependencies[violating_vertex]) {
                        all_dependencies.insert(dep);
                    }
                    
                    if (print_statement) {
                        DEBUG_PRINT("Generated smart blocking clause for non-orthogonal vectors with all dependencies:");
                        DEBUG_PRINT("Dependency chain explanation:");
                        
                        for (const auto& dep : all_dependencies) {
                            int edge_var = dep.first;
                            int connected_vertex = dep.second;
                            int vertex1 = 0, vertex2 = 0;
                            
                            // Convert edge variable back to vertex pair
                            for (int j = 1; j < k; j++) {
                                for (int i = 0; i < j; i++) {
                                    int var_idx = j*(j-1)/2 + i;
                                    if (var_idx == edge_var) {
                                        vertex1 = i;
                                        vertex2 = j;
                                        break;
                                    }
                                }
                                if (vertex1 != 0 || vertex2 != 0) break;
                            }
                            
                            DEBUG_PRINT("  Literal " << -(edge_var + 1) << " blocks edge between vertices " 
                                      << vertex1 << " and " << vertex2);
                            
                            if (edge_var == var_index1) {
                                DEBUG_PRINT("    This edge directly determines the vector for vertex " << v);
                            } else if (edge_var == var_index2) {
                                DEBUG_PRINT("    This edge directly determines the vector for vertex " << v);
                            } else if (edge_var == var_index3) {
                                DEBUG_PRINT("    This edge creates the orthogonality violation with vertex " << violating_vertex);
                            } else {
                                DEBUG_PRINT("    This edge is part of the dependency chain for determining vectors");
                            }
                        }
                        
                        DEBUG_PRINT("This blocking clause prevents configurations where non-orthogonal vectors would be connected");
                    }
                    
                    // Create blocking clause from all dependencies
                    for (const auto& dep : all_dependencies) {
                        blocking_clause.push_back(-(dep.first + 1));
                    }
                    
                    orthogonality_violations++;
                    orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
                    return blocking_clause;
                }
            }
        }
    }
    
    // Now check if all edges satisfy orthogonality
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < j; i++) {
            // Skip checking edges between predefined vertices
            if (i < NUM_PREDEFINED_VECTORS && j < NUM_PREDEFINED_VECTORS) {
                continue;
            }
            
            // Skip if either vector is not assigned or if the edge is undefined
            if (!vector_assigned[i] || !vector_assigned[j] || matrix[i][j] == l_Undef) {
                continue;
            }
            
            int dot = dot_product(vectors[i], vectors[j]);
            
            if (print_statement) {
                DEBUG_PRINT("Checking direct orthogonality between vertices " << i << " and " << j);
                DEBUG_PRINT("Vector for vertex " << i << ": [" << vectors[i][0] << "," 
                          << vectors[i][1] << "," << vectors[i][2] << "]");
                DEBUG_PRINT("Vector for vertex " << j << ": [" << vectors[j][0] << "," 
                          << vectors[j][1] << "," << vectors[j][2] << "]");
                DEBUG_PRINT("Dot product: " << dot << " (should be 0 for orthogonal vectors)");
                DEBUG_PRINT("Edge exists between these vertices: " << (matrix[i][j] == l_True ? "Yes" : "No"));
            }
            
            // Only check for the case where vectors are not orthogonal but they are connected
            if (dot != 0 && matrix[i][j] == l_True) {
                if (print_statement) {
                    DEBUG_PRINT("VIOLATION: Vertices " << i << " and " << j << " are connected but their vectors are not orthogonal");
                    DEBUG_PRINT("Connected vertices must have orthogonal vectors (dot product = 0)");
                    DEBUG_PRINT("But the dot product is " << dot << " ≠ 0");
                }
                
                // Collect all dependencies for this violation
                std::set<std::pair<int, int>> all_dependencies;
                
                // Add direct dependency
                int var_index = j*(j-1)/2 + i;
                all_dependencies.insert({var_index, i});
                
                // Add transitive dependencies for i and j
                for (const auto& dep : vector_dependencies[i]) {
                    all_dependencies.insert(dep);
                }
                for (const auto& dep : vector_dependencies[j]) {
                    all_dependencies.insert(dep);
                }
                
                if (print_statement) {
                    DEBUG_PRINT("Generated smart blocking clause for direct orthogonality violation with all dependencies:");
                    DEBUG_PRINT("Dependency chain explanation:");
                    
                    for (const auto& dep : all_dependencies) {
                        int edge_var = dep.first;
                        int connected_vertex = dep.second;
                        int vertex1 = 0, vertex2 = 0;
                        
                        // Convert edge variable back to vertex pair
                        for (int j2 = 1; j2 < k; j2++) {
                            for (int i2 = 0; i2 < j2; i2++) {
                                int var_idx = j2*(j2-1)/2 + i2;
                                if (var_idx == edge_var) {
                                    vertex1 = i2;
                                    vertex2 = j2;
                                    break;
                                }
                            }
                            if (vertex1 != 0 || vertex2 != 0) break;
                        }
                        
                        DEBUG_PRINT("  Literal " << -(edge_var + 1) << " blocks edge between vertices " 
                                  << vertex1 << " and " << vertex2);
                        
                        if (edge_var == var_index) {
                            DEBUG_PRINT("    This is the direct edge that violates orthogonality");
                        } else {
                            DEBUG_PRINT("    This edge is part of the dependency chain for determining vectors");
                        }
                    }
                    
                    DEBUG_PRINT("This blocking clause prevents configurations where non-orthogonal vectors would be connected");
                }
                
                // Create blocking clause from all dependencies
                for (const auto& dep : all_dependencies) {
                    blocking_clause.push_back(-(dep.first + 1));
                }
                
                orthogonality_violations++;
                orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
                return blocking_clause;
            }
        }
    }
    
    orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
    return blocking_clause;
}

// Modify the block_extension function to conditionally check orthogonality
std::vector<int> SymmetryBreaker::block_extension(int k) {
    std::string input = convert_assignment_to_string(k);
    if (print_statement) {
        std::cout << "Checking for partial assignment: " << input << std::endl;
        std::cout << "DEBUG: use_master_graph = " << use_master_graph << std::endl;
        std::cout << "DEBUG: use_orthogonality = " << use_orthogonality << std::endl;
    }
    
    // Convert to adjacency matrix
    adjacency_matrix_t matrix = string_to_adjacency_matrix(input, k);
    
    // Check if the partial assignment is a subgraph of the master graph
    if (use_master_graph) {
        bool is_subgraph = is_subgraph_of_master(matrix);
        if (is_subgraph) {
            subgraph_count++;
        } else {
            non_subgraph_count++;
            if (print_statement) {
                std::cout << "DEBUG: Partial assignment is not a subgraph of the master graph" << std::endl;
            }
            return generate_naive_blocking_clause(input, true);
        }
    }
    
    // Check orthogonality if enabled
    if (use_orthogonality && k > NUM_PREDEFINED_VECTORS) {
        std::vector<int> orthogonality_clause = check_orthogonality_constraints(input, k);
        if (!orthogonality_clause.empty()) {
            if (print_statement) {
                std::cout << "DEBUG: Orthogonality constraint violation detected" << std::endl;
            }
            return orthogonality_clause;
        }
    }
    
    const double before = CaDiCaL::absolute_process_time();
    std::vector<int> blocking_clause = call_RCL_binary(input, k);
    const double after = CaDiCaL::absolute_process_time();
    
    if (blocking_clause.empty()) {
        canon++;
        canontime += (after-before);
        canonarr[k-1]++;
        canontimearr[k-1] += (after-before);
    } else {
        noncanon++;
        noncanontime += (after-before);
        noncanonarr[k-1]++;
        noncanontimearr[k-1] += (after-before);
    }

    if (!blocking_clause.empty()) {
        learned_clauses_count++;
    }

    return blocking_clause;
}

void SymmetryBreaker::notify_assignment(int lit, bool is_fixed) {
    int var = std::abs(lit) - 1;
    if (var < num_edge_vars) {
        assign[var] = (lit > 0) ? l_True : l_False;
        fixed[var] = is_fixed;
    }
}

void SymmetryBreaker::notify_new_decision_level() {
    current_trail.push_back(std::vector<int>());
}

void SymmetryBreaker::notify_backtrack(size_t new_level) {
    while (current_trail.size() > new_level) {
        for (int lit : current_trail.back()) {
            int var = std::abs(lit) - 1;
            if (var < num_edge_vars) {
                assign[var] = l_Undef;
                fixed[var] = false;
            }
        }
        current_trail.pop_back();
    }
}

bool SymmetryBreaker::cb_check_found_model(const std::vector<int>& model) {
    std::vector<int> blocking_clause = block_extension(n);

    if (blocking_clause.empty()) {  // Canonical
        sol_count++;
        
        // Optimize: avoid string generation when possible
        if (print_statement && !no_print) {
            // Both debug and solution output needed - generate string once
            std::string full_assignment = convert_assignment_to_string(n);
            std::cout << "Found canonical solution #" << sol_count << ": " << full_assignment << std::endl;
            printf("Solution %ld: %s\n", sol_count, full_assignment.c_str());
            fflush(stdout);
        } else if (print_statement) {
            // Only debug output needed
            std::string full_assignment = convert_assignment_to_string(n);
            std::cout << "Found canonical solution #" << sol_count << ": " << full_assignment << std::endl;
        } else if (!no_print) {
            // Only solution output needed - use efficient direct printing
            print_solution_direct(sol_count);
        }
        
        // Generate a blocking clause for this solution
        for (int i = 0; i < num_edge_vars; i++) {
            blocking_clause.push_back(assign[i] == l_True ? -(i + 1) : (i + 1));
        }
    } else if (print_statement) {
        // Only generate string for debug output when needed
        std::string full_assignment = convert_assignment_to_string(n);
        std::cout << "Found non-canonical full assignment: " << full_assignment << std::endl;
    }

    new_clauses.push_back(blocking_clause);
    return false;
}

bool SymmetryBreaker::cb_has_external_clause() {
    if (!new_clauses.empty()) {
        if (print_statement) {
            DEBUG_PRINT("Found existing clauses in queue, returning true");
        }
        return true;
    }

    static int subgraph_check_counter = 0;
    static std::unordered_set<std::string> seen_subgraph_assignments;
    
    // Check if we only have variables <= 78 assigned
    bool only_small_vars = true;
    int highest_assigned_var = 0;
    
    for (int var = 0; var < num_edge_vars; var++) {
        if (assign[var] != l_Undef) {
            highest_assigned_var = std::max(highest_assigned_var, var + 1);
            if (var + 1 > EDGE_CUTOFF) {
                only_small_vars = false;
                break;
            }
        }
    }
    
    // Skip all external propagator checks if we only have small variables
    if (only_small_vars) {
        if (print_statement) {
            DEBUG_PRINT("Skipping external propagator checks - only variables <= 78 are assigned (highest: " 
                      << highest_assigned_var << ")");
        }
        return false;
    }
    
    // Print current partial assignment
    std::string current_assignment_str = convert_assignment_to_string(n);
    if (print_statement) {
        DEBUG_PRINT("Processing partial assignment: " << current_assignment_str);
    }
    
    // First check orthogonality for any partial assignment with at least 14 vertices
    // This is separate from the canonicity check which requires complete assignments
    if (use_orthogonality && n > NUM_PREDEFINED_VECTORS) {
        // Check if we have any edges involving vertices > 13
        bool has_large_vertex_edges = false;
        for (int j = NUM_PREDEFINED_VECTORS; j < n; j++) {
            for (int i = 0; i < j; i++) {
                int var_index = j*(j-1)/2 + i;
                if (assign[var_index] == l_True) {
                    has_large_vertex_edges = true;
                    break;
                }
            }
            if (has_large_vertex_edges) break;
        }
        
        if (has_large_vertex_edges) {
            // We have some edges involving vertices > 13, check orthogonality
            std::vector<int> orthogonality_clause = check_orthogonality_constraints(current_assignment_str, n);
            if (!orthogonality_clause.empty()) {
                if (print_statement) {
                    DEBUG_PRINT("Orthogonality constraint violation detected for partial assignment");
                    DEBUG_PRINT("Generated orthogonality blocking clause: ");
                    for (int lit : orthogonality_clause) {
                        std::cout << "  " << lit;
                    }
                    std::cout << std::endl;
                }
                
                new_clauses.push_back(orthogonality_clause);
                return true;
            }
        }
    }
    
    // Then check if current partial assignment is a subgraph of master graph
    if (use_master_graph && subgraph_check_counter % 5 == 0) {
        if (print_statement) {
            DEBUG_PRINT("Starting master graph subgraph check");
        }
        
        // Create a string representation of the current assignment (only true edges)
        std::string current_assignment;
        current_assignment.reserve(num_edge_vars * 4); // Pre-allocate memory
        
        // Quick check if we have any true assignments
        bool has_true_assignments = false;
        for (int j = 1; j < n && !has_true_assignments; j++) {
            for (int i = 0; i < j; i++) {
                int var_index = j*(j-1)/2 + i;
                if (assign[var_index] == l_True) {
                    has_true_assignments = true;
                    break;
                }
            }
        }
        
        // If no true assignments, it's trivially a subgraph
        bool need_to_check = has_true_assignments;
        
        if (need_to_check) {
            for (int j = 1; j < n; j++) {
                for (int i = 0; i < j; i++) {
                    int var_index = j*(j-1)/2 + i;
                    if (assign[var_index] == l_True) {
                        current_assignment += std::to_string(var_index) + ",";
                    }
                }
            }
            
            // Check if we've already processed this assignment
            if (seen_subgraph_assignments.find(current_assignment) != seen_subgraph_assignments.end()) {
                need_to_check = false;
            } else {
                seen_subgraph_assignments.insert(current_assignment);
            }
        }
        
        if (need_to_check) {
            if (print_statement) {
                DEBUG_PRINT("Converting partial assignment to adjacency matrix");
            }
            adjacency_matrix_t partial_matrix = string_to_adjacency_matrix(current_assignment_str, n);
            
            if (print_statement) {
                DEBUG_PRINT("Checking if partial assignment is subgraph of master graph");
            }
            bool is_subgraph = is_subgraph_of_master(partial_matrix);
            
            if (!is_subgraph) {
                if (print_statement) {
                    DEBUG_PRINT("Partial assignment is NOT a subgraph of master graph");
                }
                std::vector<int> blocking_clause = generate_naive_blocking_clause(current_assignment_str, true);
                
                if (!blocking_clause.empty() && print_statement) {
                    DEBUG_PRINT("Generated subgraph blocking clause: ");
                    for (int lit : blocking_clause) {
                        std::cout << "  " << lit;
                    }
                    std::cout << std::endl;
                }
                
                new_clauses.push_back(blocking_clause);
                non_subgraph_count++;
                return true;
            } else if (print_statement) {
                DEBUG_PRINT("Partial assignment IS a subgraph of master graph");
            }
        }
    }
    
    subgraph_check_counter++;

    // Then proceed with the regular canonicity checks (requiring complete assignments)
    for (int k = SMALL_GRAPH_ORDER + 1; k <= n; k++) {
        bool is_complete = true;
        for (int j = 0; j < k*(k-1)/2; j++) {
            if (assign[j] == l_Undef) {
                is_complete = false;
                break;
            }
        }
        
        if (is_complete) {
            std::string partial_assignment = convert_assignment_to_string(k);
            
            // Skip orthogonality check here since we already did it above for any partial assignment
            
            // Only proceed with canonicity check
            size_t hash_value = std::hash<std::string>{}(partial_assignment);

            if (seen_partial_assignments.find(hash_value) != seen_partial_assignments.end()) {
                if (print_statement) {
                    DEBUG_PRINT("Skipping already processed partial assignment of size " << k);
                }
                continue;
            }

            seen_partial_assignments.insert(hash_value);
            if (print_statement) {
                DEBUG_PRINT("Starting canonicity check for partial assignment of size " << k);
            }

            std::vector<int> blocking_clause = block_extension(k);
            
            if (!blocking_clause.empty()) {
                if (print_statement) {
                    DEBUG_PRINT("Partial assignment of size " << k << " is non-canonical");
                    DEBUG_PRINT("Generated blocking clause: ");
                    for (int lit : blocking_clause) {
                        std::cout << "  " << lit;
                    }
                    std::cout << std::endl;
                }
                
                new_clauses.push_back(blocking_clause);
                return true;
            } else if (print_statement) {
                DEBUG_PRINT("Partial assignment of size " << k << " is canonical");
            }
        } else if (print_statement) {
            DEBUG_PRINT("Partial assignment of size " << k << " is incomplete, stopping canonicity checks");
            break;
        }
    }

    if (print_statement) {
        DEBUG_PRINT("No clauses generated for current partial assignment");
    }
    return false;
}

int SymmetryBreaker::cb_add_external_clause_lit () {
    if (new_clauses.empty()) return 0;
    else {
        assert(!new_clauses.empty());
        size_t clause_idx = new_clauses.size() - 1;
        if (new_clauses[clause_idx].empty()) {
            new_clauses.pop_back();
            return 0;
        }

        int lit = new_clauses[clause_idx].back();
        
        // Print the entire clause when we start adding it
        if (new_clauses[clause_idx].size() > 1 && print_statement) {
            std::cout << "DEBUG: Adding blocking clause: ";
            for (int i = new_clauses[clause_idx].size() - 1; i >= 0; i--) {
                std::cout << new_clauses[clause_idx][i] << " ";
            }
            std::cout << std::endl;
        }
        
        new_clauses[clause_idx].pop_back();
        return lit;
    }
}

int SymmetryBreaker::cb_decide() {
    return 0;
}

int SymmetryBreaker::cb_propagate() {
    return 0;
}

int SymmetryBreaker::cb_add_reason_clause_lit(int plit) {
    (void)plit; // Suppress unused parameter warning
    return 0;
}

void SymmetryBreaker::printGraph(const graph* g, int n, int m) {
    if (print_statement) {
        std::cout << "Graph representation:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "  Node " << i << " connected to: ";
            for (int j = 0; j < n; j++) {
                if (ISELEMENT(GRAPHROW(g, i, m), j)) {
                    std::cout << j << " ";
                }
            }
            std::cout << std::endl;
        }
    }
}

void SymmetryBreaker::initNautyOptions() {
    options.writeautoms = FALSE;
    options.writemarkers = FALSE;
    options.getcanon = TRUE;
    options.defaultptn = TRUE;
}

// Modify the generate_complex_blocking_clause function to remove the degree check
std::vector<int> SymmetryBreaker::generate_complex_blocking_clause(const std::string& assignment, int k) {
    std::vector<int> blocking_clause;
    
    // Find the minimal noncanonical matrix size
    int minimal_k = k;
    std::string current_assignment = assignment;
    
    while (minimal_k > 1) {
        std::string submatrix = extract_submatrix(current_assignment, minimal_k - 1);
        if (isCanonical(submatrix)) {
            break;
        }
        current_assignment = submatrix;
        minimal_k--;
    }
    
    if (print_statement) {
        std::cout << "DEBUG: Minimal noncanonical matrix size: " << minimal_k << std::endl;
        std::cout << "DEBUG: Minimal noncanonical assignment: " << current_assignment << std::endl;
    }
    
    if (minimal_k > 1) {
        for (size_t i = 0; i < current_assignment.length(); ++i) {
            int literal = current_assignment[i] == '1' ? -(i + 1) : (i + 1);
            blocking_clause.push_back(literal);
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Generated blocking clause: ";
        for (int lit : blocking_clause) {
            std::cout << lit << " ";
        }
        std::cout << std::endl;
    }

    return blocking_clause;
}

// Optimize the generate_naive_blocking_clause function to avoid unnecessary string operations
std::vector<int> SymmetryBreaker::generate_naive_blocking_clause(const std::string& assignment, bool only_true_edges) {
    std::vector<int> blocking_clause;
    blocking_clause.reserve(assignment.length()); // Pre-allocate memory
    
    // Skip the first 78 variables (indices 0-77)
    const int skip_vars = EDGE_CUTOFF;
    
    for (size_t i = 0; i < assignment.length(); ++i) {
        // Only include variables with index > 78
        if (i + 1 > skip_vars) {
            if (only_true_edges) {
                // For subgraph blocking, only include TRUE edges
                if (assignment[i] == '1') {
                    blocking_clause.push_back(-(i + 1));
                }
            } else {
                // For canonicity blocking, include both TRUE and FALSE edges
                blocking_clause.push_back(assignment[i] == '1' ? -(i + 1) : (i + 1));
            }
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Generated " << (only_true_edges ? "subgraph" : "canonicity") 
                  << " blocking clause (skipping first " << skip_vars << " variables): ";
        for (int lit : blocking_clause) {
            std::cout << lit << " ";
        }
        std::cout << std::endl;
    }

    return blocking_clause;
}

// Optimize the string_to_adjacency_matrix function
adjacency_matrix_t SymmetryBreaker::string_to_adjacency_matrix(const std::string& input, int k) {
    adjacency_matrix_t matrix(k, std::vector<truth_value_t>(k, l_False));
    int index = 0;
    
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            if (index < input.length() && input[index++] == '1') {
                matrix[i][j] = l_True;
                matrix[j][i] = l_True;
            }
        }
    }
    return matrix;
}

std::string SymmetryBreaker::adjacency_matrix_to_string(const adjacency_matrix_t& matrix) {
    std::string result;
    int k = matrix.size();
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            result += (matrix[i][j] == l_True) ? '1' : '0';  // Changed to l_True
        }
    }
    return result;
}

void SymmetryBreaker::load_master_graph(const std::string& filepath) {
    // Set print_statement to true at the beginning of this function
    bool original_print_setting = print_statement;
    print_statement = true;
    
    // Check if we've already loaded the master graph
    if (!masterGraph.empty()) {
        std::cout << "DEBUG: Master graph already loaded, skipping reload" << std::endl;
        print_statement = original_print_setting;  // Restore original setting
        return;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open master graph file " << filepath << std::endl;
        std::cerr << "Please check that the file exists and is readable" << std::endl;
        print_statement = original_print_setting;  // Restore original setting
        return;
    }

    // Basic implementation for .lad format
    std::string line;
    std::getline(file, line); // Read the first line (number of vertices)
    int n = std::stoi(line);
    masterGraph = adjacency_matrix_t(n, std::vector<truth_value_t>(n, l_False));
    
    // Store vertex labels
    std::vector<int> vertexLabels(n);

    for (int i = 0; i < n; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        
        int label;
        iss >> label;  // Read the first number as the vertex label
        vertexLabels[i] = label;
        
        int degree;
        iss >> degree;  // Read the number of successors
        
        for (int j = 0; j < degree; j++) {
            int neighbor;
            iss >> neighbor;
            masterGraph[i][neighbor] = l_True;
            masterGraph[neighbor][i] = l_True;  // For undirected graph
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Loaded master graph with " << n << " vertices" << std::endl;
        std::cout << "DEBUG: Vertex labels: ";
        for (int i = 0; i < n; i++) {
            std::cout << vertexLabels[i] << " ";
        }
        std::cout << std::endl;
        
        // Print out connections for each vertex
        // std::cout << "DEBUG: Master graph connections:" << std::endl;
        // for (int i = 0; i < n; i++) {
        //     std::cout << "Vertex " << i << " (label " << vertexLabels[i] << ") connects to: ";
        //     bool first = true;
        //     for (int j = 0; j < n; j++) {
        //         if (masterGraph[i][j] == l_True) {
        //             if (!first) {
        //                 std::cout << ", ";
        //             }
        //             std::cout << j;
        //             first = false;
        //         }
        //     }
        //     std::cout << std::endl;
        // }
    }
    
    // Store vertex labels as a member variable for later use
    masterGraphLabels = vertexLabels;
    use_master_graph = true;
    
    // Restore original print_statement setting at the end
    print_statement = original_print_setting;
}

bool SymmetryBreaker::is_subgraph_of_master(const adjacency_matrix_t& graph) {
    // Add timer start
    const double before = CaDiCaL::absolute_process_time();

    // Implement a simple subgraph check without Glasgow
    if (masterGraph.empty()) {
        std::cerr << "Error: Master graph not loaded" << std::endl;
        const double after = CaDiCaL::absolute_process_time();
        subgraph_check_time += (after - before);
        return false;
    }

    // Simple subgraph check: for each edge in the input graph, check if it exists in the master graph
    int n = graph.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i][j] == l_True) {
                // If this edge exists in the input graph, it must exist in the master graph
                if (i >= masterGraph.size() || j >= masterGraph.size() || masterGraph[i][j] != l_True) {
                    const double after = CaDiCaL::absolute_process_time();
                    subgraph_check_time += (after - before);
                    return false;
                }
            }
        }
    }

    const double after = CaDiCaL::absolute_process_time();
    subgraph_check_time += (after - before);
    return true;
}

void SymmetryBreaker::print_subgraph_statistics() {
    if (use_master_graph) {
        printf("Subgraph checks       : %-12d\n", subgraph_count + non_subgraph_count);
        printf("  Subgraphs           : %-12d\n", subgraph_count);
        printf("  Non-subgraphs       : %-12d\n", non_subgraph_count);
        printf("Subgraph check time   : %g s\n", subgraph_check_time);
    }
    
    if (use_orthogonality) {
        printf("Orthogonality checks  : %-12ld\n", orthogonality_checks);
        printf("  Skipped checks      : %-12ld\n", orthogonality_skipped);
        printf("  Violations found    : %-12ld\n", orthogonality_violations);
        printf("Orthogonality time    : %g s\n", orthogonality_check_time);
    }
}

// Add this setter method
void SymmetryBreaker::set_use_orthogonality(bool value) {
    use_orthogonality = value;
    if (print_statement) {
        DEBUG_PRINT("Orthogonality checking " << (value ? "enabled" : "disabled"));
    }
}

// Add this setter method for controlling solution printing
void SymmetryBreaker::set_no_print(bool value) {
    no_print = value;
    if (print_statement) {
        DEBUG_PRINT("Solution printing " << (value ? "disabled" : "enabled"));
    }
}

// Efficient solution printing without string allocation
void SymmetryBreaker::print_solution_direct(long solution_number) {
    printf("Solution %ld: ", solution_number);
    
    // Print directly without creating intermediate string
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            putchar((assign[j*(j-1)/2 + i] == l_True) ? '1' : '0');
        }
    }
    putchar('\n');
    fflush(stdout);
}