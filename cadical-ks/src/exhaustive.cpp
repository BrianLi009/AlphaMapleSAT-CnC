#include "exhaustive.hpp"
#include <iostream>

ExhaustiveSearch::ExhaustiveSearch(CaDiCaL::Solver * s, int order) : solver(s) {
    if (order <= 0) {
        std::cout << "c Error: Order must be positive for exhaustive search" << std::endl;
        return;
    }
    
    num_edge_vars = order * (order - 1) / 2;
    solver->connect_external_propagator(this);
    
    std::cout << "c Running exhaustive search on order " << order << " (" << num_edge_vars << " edge variables)" << std::endl;
    
    // Observe all variables
    int max_var = solver->vars();
    for (int i = 1; i <= max_var; i++) {
        solver->add_observed_var(i);
    }
}

ExhaustiveSearch::~ExhaustiveSearch() {
    if (num_edge_vars != 0) {
        solver->disconnect_external_propagator();
        printf("Number of solutions: %ld\n", sol_count);
    }
}

void ExhaustiveSearch::notify_assignment(int lit, bool is_fixed) {
    // Not needed for simple exhaustive search
    (void)lit;
    (void)is_fixed;
}

void ExhaustiveSearch::notify_new_decision_level() {
    // Not needed for simple exhaustive search
}

void ExhaustiveSearch::notify_backtrack(size_t new_level) {
    // Not needed for simple exhaustive search
    (void)new_level;
}

bool ExhaustiveSearch::cb_check_found_model(const std::vector<int> & model) {
    sol_count += 1;

#ifdef VERBOSE
    std::cout << "c New solution was found: ";
#endif
    std::vector<int> clause;
    // Only include the first num_edge_vars variables in the blocking clause
    for (int i = 0; i < num_edge_vars && i < model.size(); i++) {
#ifdef VERBOSE
        if (model[i] > 0) {
            std::cout << model[i] << " ";
        }
#endif
        clause.push_back(-model[i]);
    }
#ifdef VERBOSE
    std::cout << std::endl;
#endif
    new_clauses.push_back(clause);
    solver->add_trusted_clause(clause);

    return false;
}

bool ExhaustiveSearch::cb_has_external_clause() {
    return !new_clauses.empty();
}

int ExhaustiveSearch::cb_add_external_clause_lit() {
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

int ExhaustiveSearch::cb_decide() { return 0; }
int ExhaustiveSearch::cb_propagate() { return 0; }
int ExhaustiveSearch::cb_add_reason_clause_lit(int plit) {
    (void)plit;
    return 0;
}
