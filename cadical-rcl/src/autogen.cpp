#include "autogen.hpp"
#include <iostream>
#include <algorithm>
#include "hash_values.h"
#include "../../nauty2_8_8/nauty.h"
#include <unordered_set>
#include <string>
#include <chrono>
#include <cstring>
#include <bitset>
#include <limits>
#include <sstream>

FILE * exhaustfile = NULL;

// Replace the CANONICITY_CHECK_FREQUENCY constant with an array
int CANONICITY_CHECK_FREQUENCY[MAXORDER];

const int MIN_ORDER_FOR_FREQUENCY_CHECK = 1;

// Array to store the dynamic cutoff values
long perm_cutoff[MAXORDER];

long canon = 0;
long noncanon = 0;
double canontime = 0;
double noncanontime = 0;
long canonarr[MAXORDER] = {};
long noncanonarr[MAXORDER] = {};
double canontimearr[MAXORDER] = {};
double noncanontimearr[MAXORDER] = {};
#ifdef PERM_STATS
long canon_np[MAXORDER] = {};
long noncanon_np[MAXORDER] = {};
#endif

// At the class level, add a new array to keep track of when we last updated each order
long last_update_check[MAXORDER] = {0};

// Add these new variables at the class level
std::vector<long long> total_perms_checked;
std::vector<long> total_checks;
std::vector<bool> cutoff_calibrated;
const int CALIBRATION_SAMPLES = 100;

SymmetryBreaker::SymmetryBreaker(CaDiCaL::Solver * s, int order, int uc) : solver(s) {
    (void)uc; // Parameter is intentionally unused
    if (order == 0) {
        std::cout << "c Need to provide order to use programmatic code" << std::endl;
        return;
    }
    n = order;
    num_edge_vars = n*(n-1)/2;
    
    std::cout << "c Initializing SymmetryBreaker with order " << n << std::endl;
    
    // Create unique filename based on parameters
    std::stringstream exhaust_filename;
    
    // Base name with order
    exhaust_filename << "exhaust_order" << n;
    
    // Add file extension
    exhaust_filename << ".txt";
    
    // Open the file
    exhaustfile = fopen(exhaust_filename.str().c_str(), "w");
    
    if (!exhaustfile) {
        std::cerr << "Error: Could not open output file" << std::endl;
        return;
    }
    
    assign = new int[num_edge_vars];
    fixed = new bool[num_edge_vars];
    colsuntouched = new int[n];
    
    std::cout << "c Arrays allocated" << std::endl;
    
    solver->connect_external_propagator(this);
    for (int i = 0; i < num_edge_vars; i++) {
        assign[i] = l_Undef;
        fixed[i] = false;
    }
    
    std::cout << "c Arrays initialized" << std::endl;
    
    std::cout << "c Running orderly generation on order " << n << " (" << num_edge_vars << " edge variables)" << std::endl;
    
    // The root-level of the trail is always there
    current_trail.push_back(std::vector<int>());
    // Observe the edge variables for orderly generation
    for (int i = 0; i < num_edge_vars; i++) {
        solver->add_observed_var(i+1);
    }
    learned_clauses_count = 0;

    std::cout << "c Initializing new vector members" << std::endl;
    
    total_perms_checked.resize(MAXORDER, 0);
    total_checks.resize(MAXORDER, 0);
    cutoff_calibrated.resize(MAXORDER, false);
    
    std::cout << "c New vector members initialized" << std::endl;

    // Initialize nauty options
    options.getcanon = FALSE;
    options.digraph = FALSE;
    options.writeautoms = FALSE;
    options.writemarkers = FALSE;
    options.defaultptn = FALSE;
    options.cartesian = FALSE;
    options.linelength = 0;
    options.outfile = NULL;
    options.userrefproc = NULL;
    options.userautomproc = NULL;
    options.userlevelproc = NULL;
    options.usernodeproc = NULL;
    options.usercanonproc = NULL;

    // Add these lines (around line 70)
    for (int i = 0; i < MAXORDER; ++i) {
        total_perms_checked[i] = 0;
        total_checks[i] = 0;
        last_update_check[i] = 0;
    }

    std::cout << "c Initializing CANONICITY_CHECK_FREQUENCY" << std::endl;
    
    // Initialize CANONICITY_CHECK_FREQUENCY array
    for (int i = 0; i < MAXORDER; ++i) {
        CANONICITY_CHECK_FREQUENCY[i] = std::max(10, i * 2);
    }

    // Initialize partial assignment counts for each order
    for (int i = 0; i < MAXORDER; ++i) {
        partial_assignment_counts[i] = 0;
    }

    total_perms_checked.resize(MAXORDER, 0);
    total_checks.resize(MAXORDER, 0);
    cutoff_calibrated.resize(MAXORDER, false);

    perm_counts_during_calibration.resize(MAXORDER);
    cutoff_calibrated.resize(MAXORDER, false);
}

SymmetryBreaker::~SymmetryBreaker() {
    if (n != 0) {
        // Close the file
        if (exhaustfile) fclose(exhaustfile);
        
        solver->disconnect_external_propagator ();
        delete [] assign;
        delete [] colsuntouched;
        delete [] fixed;
        printf("Number of solutions   : %ld\n", sol_count);
        printf("Canonical subgraphs   : %-12" PRIu64 "   (%.0f /sec)\n", (uint64_t)canon, canon/canontime);
        for(int i=2; i<n; i++) {
#ifdef PERM_STATS
            printf("          order %2d    : %-12" PRIu64 "   (%.0f /sec) %.0f avg. perms\n", i+1, (uint64_t)canonarr[i], canonarr[i]/canontimearr[i], canon_np[i]/(float)(canonarr[i] > 0 ? canonarr[i] : 1));
#else
            printf("          order %2d    : %-12" PRIu64 "   (%.0f /sec)\n", i+1, (uint64_t)canonarr[i], canonarr[i]/canontimearr[i]);
#endif
        }
        printf("Noncanonical subgraphs: %-12" PRIu64 "   (%.0f /sec)\n", (uint64_t)noncanon, noncanon/noncanontime);
        for(int i=2; i<n; i++) {
#ifdef PERM_STATS
            printf("          order %2d    : %-12" PRIu64 "   (%.0f /sec) %.0f avg. perms\n", i+1, (uint64_t)noncanonarr[i], noncanonarr[i]/noncanontimearr[i], noncanon_np[i]/(float)(noncanonarr[i] > 0 ? noncanonarr[i] : 1));
#else
            printf("          order %2d    : %-12" PRIu64 "   (%.0f /sec)\n", i+1, (uint64_t)noncanonarr[i], noncanonarr[i]/noncanontimearr[i]);
#endif
        }
        printf("Canonicity checking   : %g s\n", canontime);
        printf("Noncanonicity checking: %g s\n", noncanontime);
        printf("Total canonicity time : %g s\n", canontime+noncanontime);

        print_stats();

        printf("Final perm_cutoff values:\n");
        for (int i = 0; i < n; i++) {
            if (cutoff_calibrated[i]) {
                printf("Order %2d: %ld\n", i+1, perm_cutoff[i]);
            } else {
                printf("Order %2d: Not calibrated\n", i+1);
            }
        }

        printf("Final statistics:\n");
        for (int i = 0; i < n; i++) {
            printf("Order %2d: total_checks = %ld, total_perms_checked = %ld\n", 
                   i+1, total_checks[i], total_perms_checked[i]);
        }
    }
}

void SymmetryBreaker::notify_assignment(int lit, bool is_fixed) {
    assign[abs(lit)-1] = (lit > 0 ? l_True : l_False);
    if (is_fixed) {
        fixed[abs(lit)-1] = true;
    } else {
        current_trail.back().push_back(lit);
    }
}

void SymmetryBreaker::notify_new_decision_level () {
    current_trail.push_back(std::vector<int>());
}

void SymmetryBreaker::notify_backtrack (size_t new_level) {
    while (current_trail.size() > new_level + 1) {
        for (const auto& lit: current_trail.back()) {
            const int x = abs(lit) - 1;
            // Don't remove literals that have been fixed
            if(fixed[x])
                continue;
            assign[x] = l_Undef;
            const int col = 1+(-1+sqrt(1+8*x))/2;
            for(int i=col; i<n; i++)
                colsuntouched[i] = false;
        }
        current_trail.pop_back();
    }
}

bool SymmetryBreaker::cb_check_found_model(const std::vector<int> & model) {
    assert(model.size() == num_edge_vars);

    // Compute the canonical form for the current model
    auto [orbits, current_canonical_form] = compute_orbits_and_canonical_form(n);

    // Check if this canonical form has been seen before
    bool is_unique = unique_canonical_forms.insert(current_canonical_form).second;

    // Only write to file and increment counter if it's a unique solution
    if (is_unique) {
        sol_count += 1;
        
        // Write the unique solution to exhaust file
        if (exhaustfile != NULL) {
            for (const auto& lit : model) {
                fprintf(exhaustfile, "%d ", lit);
            }
            fflush(exhaustfile);
        }
    }

    // Always add the blocking clause
    std::vector<int> clause;
    for (const auto& lit: model) {
        clause.push_back(-lit);
    }
    new_clauses.push_back(clause);

    return false;
}

bool SymmetryBreaker::cb_has_external_clause () {
    if(!new_clauses.empty())
        return true;

    long hash = 0;

    // Initialize i to be the first column that has been touched since the last analysis
    int i = 2;
    for(; i < n; i++) {
        if(!colsuntouched[i])
            break;
    }
    // Ensure variables are defined and update current graph hash
    for(int j = 0; j < i*(i-1)/2; j++) {
        if(assign[j] == l_Undef)
            return false;
        else if(assign[j] == l_True)
            hash += hash_values[j];
    }
    for(; i < n; i++) {
        // Ensure variables are defined and update current graph hash
        for(int j = i*(i-1)/2; j < i*(i+1)/2; j++) {
            if(assign[j]==l_Undef) {
                return false;
            }
            if(assign[j]==l_True) {
                hash += hash_values[j];
            }
        }
        colsuntouched[i] = true;

        // Increment the partial assignment count for this order
        partial_assignment_counts[i]++;

        // Check if it's a full assignment or if it's time to perform a canonicity check
        if ((partial_assignment_counts[i] % CANONICITY_CHECK_FREQUENCY[i] == 0 || i + 1 < MIN_ORDER_FOR_FREQUENCY_CHECK)) {
            // Check if current graph hash has been seen
            if(canonical_hashes[i].find(hash)==canonical_hashes[i].end())
            {
                // Found a new subgraph of order i+1 to test for canonicity
                // Uses a pseudo-check except when i+1 = n
                const double before = CaDiCaL::absolute_process_time();
                // Run canonicity check
                int p[i+1]; // Permutation on i+1 vertices
                int x, y;   // These will be the indices of first adjacency matrix entry that demonstrates noncanonicity (when such indices exist)
                int mi;     // This will be the index of the maximum defined entry of p
                bool ret = is_canonical(i+1, p, x, y, mi);
                const double after = CaDiCaL::absolute_process_time();

                // If subgraph is canonical
                if (ret) {
                    canon++;
                    canontime += (after-before);
                    canonarr[i]++;
                    canontimearr[i] += (after-before);
                    canonical_hashes[i].insert(hash);  // Add hash to set of canonical hashes
                }
                // If subgraph is not canonical then block it
                else {
                    noncanon++;
                    noncanontime += (after-before);
                    noncanonarr[i]++;
                    noncanontimearr[i] += (after-before);
                    
                    new_clauses.push_back(std::vector<int>());

                    // Start timing for generate_blocking_clause_smaller
                    auto start = std::chrono::high_resolution_clock::now();
                    smaller_calls++;

                    // Generate a blocking clause smaller than the naive blocking clause
                    new_clauses.back().push_back(-(x*(x-1)/2+y+1));
                    const int px = MAX(p[x], p[y]);
                    const int py = MIN(p[x], p[y]);
                    new_clauses.back().push_back(px*(px-1)/2+py+1);
                    for(int ii=0; ii < x+1; ii++) {
                        for(int jj=0; jj < ii; jj++) {
                            if(ii==x && jj==y) {
                                break;
                            }
                            const int pii = MAX(p[ii], p[jj]);
                            const int pjj = MIN(p[ii], p[jj]);
                            if(ii==pii && jj==pjj) {
                                continue;
                            } else if(assign[ii*(ii-1)/2+jj] == l_True) {
                                new_clauses.back().push_back(-(ii*(ii-1)/2+jj+1));
                            } else if (assign[pii*(pii-1)/2+pjj] == l_False) {
                                new_clauses.back().push_back(pii*(pii-1)/2+pjj+1);
                            }
                        }
                    }

                    // End timing for generate_blocking_clause_smaller
                    auto end = std::chrono::high_resolution_clock::now();
                    smaller_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                    learned_clauses_count++;

                    return true;
                }
            }
        }
    }

    // No programmatic clause generated
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
        new_clauses[clause_idx].pop_back();
        return lit;
    }
}

int SymmetryBreaker::cb_decide () { return 0; }
int SymmetryBreaker::cb_propagate () { return 0; }
int SymmetryBreaker::cb_add_reason_clause_lit (int plit) {
    (void)plit;
    return 0;
};

// Modify the is_canonical function
bool SymmetryBreaker::is_canonical(int k, int p[], int& x, int& y, int& i) {
    int pl[k]; // pl[k] contains the current list of possibilities for kth vertex (encoded bitwise)
    int pn[k+1]; // pn[k] contains the initial list of possibilities for kth vertex (encoded bitwise)
    
    // Initialize all possibilities
    for (int j = 0; j <= k; j++) {
        pn[j] = (1 << k) - 1;
    }

    auto [orbits, canonical_form] = compute_orbits_and_canonical_form(k);
    current_canonical_form = canonical_form;  // Store the canonical form
    remove_possibilities(k, pn, orbits);

    for (int j = 0; j <= k; j++) {
        pl[j] = pn[j];
    }
    i = 0;

    int np = 1;
    long long limit = std::numeric_limits<long long>::max(); // No limit during calibration

    while(np < limit) {
        // If no possibilities for ith vertex then backtrack
        while (pl[i] == 0) {
            i--;
            if (i == -1) {
#ifdef PERM_STATS
                canon_np[k-1] += np;
#endif
                return true;
            }
            // Reset possibilities for the next vertex
            pl[i+1] = pn[i+1];
        }

        p[i] = __builtin_ctz(pl[i]); // Get index of rightmost high bit
        pl[i] &= pl[i] - 1;  // Remove this possibility
        
        if (i < k - 1) {
            pl[i+1] = pn[i+1] & ~(1 << p[i]); // Update possibilities for next vertex
        }

        // Always apply the pseudo-test optimization for subgraphs
        if(i == 0 && p[i] == 1 && k < n) {
            const int PSEUDO_TEST_LIMIT = std::min(1000, 100 * k);
            limit = np + PSEUDO_TEST_LIMIT;
        }

        // Determine if the permuted matrix p(M) is lex-smaller than M
        bool lex_result_unknown = false;
        x = 1;  // Always start from the first row
        y = 0;
        int j;
        for(j = 0; j < k*(k-1)/2; j++) {  // Start from the beginning of the matrix
            if(x > i) {
                // Unknown if permutation produces a larger or smaller matrix
                lex_result_unknown = true;
                break;
            }
            const int px = MAX(p[x], p[y]);
            const int py = MIN(p[x], p[y]);
            const int pj = px*(px-1)/2 + py;
            if(assign[j] != assign[pj]) {
                if(assign[j] == l_False && assign[pj] == l_True) {
                    // P(M) > M
                    total_larger_perms++;  // Increment the counter
                    break;
                }
                if(assign[j] == l_True && assign[pj] == l_False) {
                    // P(M) < M: Generate clause to block M
                    total_smaller_perms++;
                    return false;
                }
            }

            y++;
            if(x==y) {
                x++;
                y = 0;
            }
        }

        if(lex_result_unknown) {
            i++;
            if (i < k) {
                pl[i] = pn[i];
                for (int j = 0; j < i; j++) {
                    pl[i] &= ~(1 << p[j]);  // Remove already used vertices
                }
            }
        }
        else {
            np++;
        }
    }

    // After the main loop:
    if (!cutoff_calibrated[k-1]) {
        perm_counts_during_calibration[k-1].push_back(np);

        // Check if we've gathered enough samples for calibration
        if (perm_counts_during_calibration[k-1].size() >= CALIBRATION_SAMPLES) {
            std::vector<long long>& counts = perm_counts_during_calibration[k-1];
            std::nth_element(counts.begin(), counts.begin() + counts.size()/2, counts.end());
            const double PERM_CUTOFF_MULTIPLIER = 5.0;
            perm_cutoff[k-1] = static_cast<long>(counts[counts.size()/2] * PERM_CUTOFF_MULTIPLIER);
            cutoff_calibrated[k-1] = true;
            std::cout << "Calibrated perm_cutoff for order " << k << ": " << perm_cutoff[k-1] << std::endl;
            
            // Clear the vector to free up memory
            std::vector<long long>().swap(counts);
        }
    }

    // Use the calibrated cutoff if available
    if (cutoff_calibrated[k-1]) {
        return np < perm_cutoff[k-1];
    }

    // If not calibrated yet, assume canonical
    return true;
}

void SymmetryBreaker::generate_blocking_clause_larger(int k, int p[], int x, int y) {
    (void)y;  // Indicate that y is intentionally unused
    // Check if the permutation includes all vertices
    if (x != k - 1) {
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    larger_calls++;

    std::vector<int> clause;
    clause.reserve(k);  // Pre-allocate space for efficiency

    // Generate the clause
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < i; j++) {
            if (assign[i*(i-1)/2 + j] != assign[p[i]*(p[i]-1)/2 + p[j]]) {
                if (assign[i*(i-1)/2 + j] == l_False) {
                    clause.push_back(i*(i-1)/2 + j + 1);
                } else {
                    clause.push_back(-(i*(i-1)/2 + j + 1));
                }
                break;
            }
        }
        if (clause.size() >= 4) break;  // Stop after finding 3 literals
    }

    // Only proceed if the clause has 1 to 3 literals
    if (clause.size() > 4 || clause.empty()) {
        auto end = std::chrono::high_resolution_clock::now();
        larger_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return;
    }

    // If it's a unit clause, check if it's already in the cache
    if (clause.size() == 1 && unit_clause_cache.find(clause[0]) != unit_clause_cache.end()) {
        auto end = std::chrono::high_resolution_clock::now();
        larger_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return;
    }

    // Create a string representation of the clause
    std::string clause_str;
    for (int lit : clause) {
        clause_str += std::to_string(lit) + " ";
    }

    // Check if this clause has already been generated
    if (generated_clauses.insert(clause_str).second) {
        // If it's a new clause, add it to the solver
        learned_clauses_count++;
        larger_clauses_generated++;  // Increment the counter for actually generated clauses
    }

    auto end = std::chrono::high_resolution_clock::now();
    larger_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void SymmetryBreaker::print_stats() {
    std::cout << "Total permutations P satisfying P(M) > M: " << total_larger_perms << std::endl;
    std::cout << "Total permutations P satisfying P(M) < M: " << total_smaller_perms << std::endl;
    std::cout << "generate_blocking_clause_larger: " << larger_calls << " calls, " 
              << larger_time << " microseconds total, " 
              << (larger_calls > 0 ? larger_time / larger_calls : 0) << " microseconds average per call\n";
    std::cout << "generate_blocking_clause_smaller: " << smaller_calls << " calls, " 
              << smaller_time << " microseconds total, " 
              << (smaller_calls > 0 ? smaller_time / smaller_calls : 0) << " microseconds average per call\n";
}

std::pair<std::vector<int>, std::vector<graph>> SymmetryBreaker::compute_orbits_and_canonical_form(int k) {
    std::vector<char> g6 = convert_assignment_to_graph6(k);
    if (g6.empty()) return std::make_pair(std::vector<int>(), std::vector<graph>());
    int n = k;
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE, m, n, NAUTYVERSIONID);

    std::vector<graph> g(m * n);
    std::vector<graph> cg(m * n);  // For canonical graph
    EMPTYGRAPH(g.data(), m, n);

    int pos = 0;
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            if (g6[pos] == '1') {
                ADDONEEDGE(g.data(), i, j, m);
            }
            ++pos;
        }
    }

    std::vector<int> orbits(n);
    std::vector<int> lab(n);
    std::vector<int> ptn(n);
    for (int i = 0; i < n; ++i) {
        lab[i] = i;
        ptn[i] = 1;
    }
    ptn[n-1] = 0;

    DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;  // We want the canonical graph

    statsblk stats;
    
    densenauty(g.data(), lab.data(), ptn.data(), orbits.data(), &options, &stats, m, n, cg.data());

    return std::make_pair(orbits, cg);
}

std::vector<char> SymmetryBreaker::convert_assignment_to_graph6(int k) {
    std::vector<char> g6;
    g6.reserve(k * (k - 1) / 2);

    for (int j = 1; j < k; ++j) {
        for (int i = 0; i < j; ++i) {
            bool edge_exists = (assign[j*(j-1)/2 + i] == l_True);
            g6.push_back(edge_exists ? '1' : '0');
        }
    }

    return g6;
}

void SymmetryBreaker::remove_possibilities(int k, int pn[], const std::vector<int>& orbits) {
    if (k > 0 && k <= MAXORDER && orbits.size() == static_cast<size_t>(k)) {
        // Find the largest orbit
        int largest_orbit_size = 0;
        int largest_orbit_id = -1;
        int first_vertex_largest_orbit = -1;

        for (int i = 0; i < k; i++) {
            int orbit_size = 0;
            for (int j = 0; j < k; j++) {
                if (orbits[j] == orbits[i]) orbit_size++;
            }
            if (orbit_size > largest_orbit_size) {
                largest_orbit_size = orbit_size;
                largest_orbit_id = orbits[i];
                first_vertex_largest_orbit = i;
            }
        }

        if (largest_orbit_id != -1) {
            // Process the largest orbit
            for (int i = 0; i < k; i++) {
                if (orbits[i] == largest_orbit_id) {
                    for (int j = 0; j < k; j++) {
                        if (i != j && orbits[j] == largest_orbit_id) {
                            pn[i] &= ~(1 << j);
                        }
                    }
                }
            }

            // std::cout << "Processing other orbits..." << std::endl;
            // Process other orbits
            for (int i = 0; i < k; i++) {
                if (orbits[i] != largest_orbit_id) {
                    // std::cout << "  Processing vertex " << i << " (orbit " << orbits[i] << ")" << std::endl;
                    bool is_representative = true;
                    for (int j = 0; j < i; j++) {
                        if (orbits[j] == orbits[i]) {
                            is_representative = false;
                            // std::cout << "    Not a representative (same orbit as " << j << ")" << std::endl;
                            break;
                        }
                    }
                    if (!is_representative) {
                        pn[first_vertex_largest_orbit] &= ~(1 << i);
                    }
                }
            }
        }
    } else {
        // std::cout << "Conditions not met. Skipping orbit processing." << std::endl;
    }
    
    // Log final pn values
    // std::cout << "Final pn values:" << std::endl;
    // for (int i = 0; i < k; i++) {
    //     std::cout << "  pn[" << i << "] = " << std::bitset<32>(pn[i]) << std::endl;
    // }
    
    // std::cout << "Exiting remove_possibilities" << std::endl;
}