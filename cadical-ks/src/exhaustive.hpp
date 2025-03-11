#ifndef _exhaustive_hpp_INCLUDED
#define _exhaustive_hpp_INCLUDED

#include "internal.hpp"

class ExhaustiveSearch : CaDiCaL::ExternalPropagator {
    CaDiCaL::Solver * solver;
    std::vector<std::vector<int>> new_clauses;
    int num_edge_vars = 0;
    long sol_count = 0;

public:
    ExhaustiveSearch(CaDiCaL::Solver * s, int order);
    ~ExhaustiveSearch();
    
    // Required ExternalPropagator interface methods
    void notify_assignment(int lit, bool is_fixed);
    void notify_new_decision_level();
    void notify_backtrack(size_t new_level);
    bool cb_check_found_model(const std::vector<int> & model);
    bool cb_has_external_clause();
    int cb_add_external_clause_lit();
    int cb_decide();
    int cb_propagate();
    int cb_add_reason_clause_lit(int plit);
};

#endif
