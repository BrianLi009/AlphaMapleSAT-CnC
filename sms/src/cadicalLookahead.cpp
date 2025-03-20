#include "useful.h"
#include "cadicalSMS.hpp"
#include "cadical.hpp"

// #define NUM_RELEVANT_VARIABLES 3157 // TODO delete later!!!!!

void CadicalSolver::setDefaultLookaheadArguments()
{
    // return;
    if (!solver->set("chronoalways", 1))
        EXIT_UNWANTED_STATE

    if (!solver->set("restart", 0))
        EXIT_UNWANTED_STATE

    if (config.lookaheadAll)
    {
        int numRelevantVariables = config.nextFreeVariable - 1;
        // numRelevantVariables = NUM_RELEVANT_VARIABLES; // TODO delete
        // resize current assignment
        auto currentAssignmentCopy = currentAssignment;
        currentAssignment = vector<truth_value_t>(numRelevantVariables + 1, truth_value_unknown);
        for (size_t i = 0; i < currentAssignmentCopy.size(); i++)
            currentAssignment[i] = currentAssignmentCopy[i];

        auto isFixedCopy = isFixed;
        isFixed = vector<bool>(numRelevantVariables + 1, false);
        for (size_t i = 0; i < isFixedCopy.size(); i++)
            isFixed[i] = isFixedCopy[i];

        // add all non edge variables as observed
        int numEdges = vertices * (vertices - 1) / 2;
        for (int i = numEdges + 1; i <= numRelevantVariables; i++)
        {
            solver->add_observed_var(i);
        }

        config.frequency = 1; // high frequency
    }
}

int CadicalSolver::lookaheadDecide()
{
    if (inPrerunState)
        return 0;

    // fflush(stdout);
    // printf("Lookahead decide\n");

    size_t decisionLvl = current_trail.size() - 1; // the actual current decision level before picking the next branching literal
    // printf("Current decision level: %ld\n", decisionLvl);

    bool startNewLookaheadLevel = false;

    if (lookaheadPause)
    {
        if (allEdgeVariablesAssigned())
            return 0; // no lookahead just continue

        lookaheadPause = false;
        startNewLookaheadLevel = true;
    }
    else
    {
        // printf("Decision level vs currentLookaheadDecisionLevel: %ld vs %ld\n", decisionLvl, currentLookaheadDecisionLevel);
        if (decisionLvl > currentLookaheadDecisionLevel + 1)
        {
            printf("At most one level higher than current decision level\n");
            EXIT_UNWANTED_STATE
        }

        if (decisionLvl <= currentLookaheadDecisionLevel)
            startNewLookaheadLevel = true;

        if (decisionLvl == currentLookaheadDecisionLevel + 1) // one level higher, i.e., literal was tested
        {
            // update the number of propagated literals
            int absLit = abs(previousLookaheadLiteral);
            int numPropagated = (int)current_trail.back().size();
            // if (true)
            // {
            //     numPropagated = 0;
            //     int numEdges = vertices * (vertices - 1) / 2;
            //     for (auto lit: current_trail.back())
            //     {
            //         if (abs(lit) <= numEdges)
            //             numPropagated++;
            //     }
            // }

            if (previousLookaheadLiteral > 0)
                this->numPropagated[absLit].first = numPropagated;
            else
                this->numPropagated[absLit].second = numPropagated;

            // printf("Num propagated literals for %d: %d\n", previousLookaheadLiteral, numPropagated);

            // printf("Before: %ld\n", current_trail.size());

            solver->force_backtrack(currentLookaheadDecisionLevel); // backtrack to test another literal or make a final decision
            // printf("Force backtrack to level %ld\n", currentLookaheadDecisionLevel);

            // printf("After: %ld\n", current_trail.size());
        }
    }

    // !!!! trail must be at correct level here, i.e., backtracked and everything

    decisionLvl = current_trail.size() - 1; // update decision level
    if (allEdgeVariablesAssigned())
    {
        // printf("All edge variables are assigned\n");
        lookaheadPause = true;
        return 0; // lookahead doesn't make sense
    }

    if (startNewLookaheadLevel)
        startLookaheadOnNewLevel(decisionLvl);

    return lookaheadPickVariable();
}

#define SMALL_DOUBLE 0.00000001

inline double scoringFunction1(int x, int y)
{
    return std::min(x, y) + SMALL_DOUBLE * (x + y);
}

inline double scoringFunction2(int x, int y)
{
    return x + y + SMALL_DOUBLE * std::min(x, y);
}

inline double scoringFunction3(int x, int y)
{
    return x + y + 5 * std::min(x, y);
}

inline double scoringFunction4(int x, int y)
{
    return x * y + (x + y);
}

// int numRealDecisions = 0;

// pick the next variable for branching based on the lookahead heuristic
int CadicalSolver::nextBranchingLiteral()
{
    int numRelevantVariables = vertices * (vertices - 1) / 2;
    if (config.lookaheadAll)
    {
        numRelevantVariables = nextFreeVariable - 1;
        // numRelevantVariables = NUM_RELEVANT_VARIABLES;
    }

    // printf("Next lookahead literal\n");
    int bestLit = 0;
    double bestScore = -1;
    for (int i = 1; i <= numRelevantVariables; i++)
    {
        if (currentAssignment[i] != truth_value_unknown) // already fixed
            continue;
        double score = 0;
        switch (config.lookaheadHeuristic)
        {
        case 1:
            score = scoringFunction1(numPropagated[i].first, numPropagated[i].second);
            break;
        case 2:
            score = scoringFunction2(numPropagated[i].first, numPropagated[i].second);
            break;
        case 3:
            score = scoringFunction3(numPropagated[i].first, numPropagated[i].second);
            break;
        case 4:
            score = scoringFunction4(numPropagated[i].first, numPropagated[i].second);
            break;
        default:
            EXIT_UNWANTED_STATE;
        } 
        
        
        if (score > bestScore)
        {
            bestScore = score;
            bestLit = i;
        }
    }
    if (numPropagated[bestLit].first < numPropagated[bestLit].second)
        bestLit = -bestLit; // always take the branch first which is more likely to fail, i.e., has more propagated literals

    inLookaheadState = false;

    // numRealDecisions++;
    // printf("Number of real decisions: %d\n", numRealDecisions);

    return bestLit;
}

// get all unassigned literals for the specified level and add them to the potential lookahead literals
void CadicalSolver::initBranchingLiterals()
{

    int numRelevantVariables = vertices * (vertices - 1) / 2;
    if (config.lookaheadAll)
    {
        numRelevantVariables = nextFreeVariable - 1;
        // numRelevantVariables = NUM_RELEVANT_VARIABLES;
    }

    lookaheadLiterals.clear();
    numPropagated = vector<pair<int, int>>(numRelevantVariables + 1, std::make_pair(-1, -1));

    // printf("Init branching literals\n");
    for (int i = numRelevantVariables; i >= 1; i--) // more likely that the variables are more important (read from the end later)
    {
        if (currentAssignment[i] != truth_value_unknown) // already fixed
            continue;
        lookaheadLiterals.push_back(i);
        lookaheadLiterals.push_back(-i);
        // printf("ASDF\n");
    }
    // printf("Number of branching literals: %ld from potential %d\n", lookaheadLiterals.size(), numEdges * 2);
}

// return the next literal to test for the lookahead
int CadicalSolver::nextLitToTest()
{
    inLookaheadState = true;
    while (!lookaheadLiterals.empty())
    {
        int nextLit = lookaheadLiterals.back();
        lookaheadLiterals.pop_back();
        if (currentAssignment[abs(nextLit)] != truth_value_unknown)
            continue;
        // printf("Next literal to test: %d\n", nextLit);
        previousLookaheadLiteral = nextLit;
        return nextLit;
    }
    return 0;
}

// start the lookahead for the current level
void CadicalSolver::startLookaheadOnNewLevel(size_t level)
{
    currentLookaheadDecisionLevel = level;
    initBranchingLiterals();
}

// pick brunching variable and finish lookahead, i.e., increasing currentLookaheadDecisionLevel
int CadicalSolver::finishLookahead()
{
    int lit = nextBranchingLiteral();
    currentLookaheadDecisionLevel++;
    // printf("Next branching literal: %d\n", lit);
    if (lit == 0)
    {
        assert(allEdgeVariablesAssigned());
        printf("Strange case that no branching literal is left\n");
        lookaheadPause = true;
        return 0;
    }
    // return 0;

    return lit;
}

// asserts that currentLookaheadDecisionLevel is set correctly
int CadicalSolver::lookaheadPickVariable()
{
    int lit = nextLitToTest();
    if (lit)
        return lit;
    return finishLookahead();
}
