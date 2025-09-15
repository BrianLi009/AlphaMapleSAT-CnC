#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define SIMP_NCONF 10000

enum {CUBE = 1, SOLVE = 2, DONE = 3};
enum {UNKNOWN = 0, UNSAT = 20, SAT = 10};

class Cube {
    private:
        std::string cnf;
        std::string cube;
        std::string nextcube;
        std::string instance;
        std::string simp_inst;
        std::string id;
        int order;
        int index;
        int cutoffv;
        int d;
        int status;
    public:
        Cube(std::string cnf, std::string cube, std::string id,
            int order, int index, int cutoffv, int d) :
                cnf(cnf), cube(cube), id(id),
                order(order), index(index), 
                cutoffv(cutoffv), d(d) {};
        void apply();
        int simplify(bool ext, char cutoff);
        std::string gen_cube(std::string cubing_mode, int numMCTS);
        int solve();
        std::string get_id();
        std::string get_instance() { return instance; };
        std::string get_nextcube() { return nextcube; };
        int get_status() { return status; };
};

// add the cube variable to CNF
void Cube::apply() {
    if (cube == "") {
        instance = cnf;
        return;
    }
    std::stringstream cmd;
    cmd << "./gen_cubes/apply.sh ";
    cmd << cnf << " " << cube << " " << index << " > ";
    cmd << cnf << "." << id << index << ".cnf";

    printf("%s\n", cmd.str().c_str());
    system(cmd.str().c_str());

    std::stringstream ss;
    ss << cnf << "." << id << index << ".cnf";
    instance = ss.str();
}

//
int Cube::simplify(bool ext, char cutoff) {
    // simplification
    std::stringstream scmd;
    scmd << "./simplification/simplify-by-conflicts.sh ";
    scmd << instance << " " << order << " " << SIMP_NCONF << " -cas";
    status = system(scmd.str().c_str());
    simp_inst = instance + ".simp";
    if (WEXITSTATUS(status) == UNSAT) { return UNSAT; }

    // get vars removed
    int var_removed;
    std::stringstream vcmd;
    vcmd << "sed -E 's/.* 0 [-]*([0-9]*) 0$/\\1/' < ";
    vcmd << instance << ".ext ";
    vcmd << "| awk '$0<=" << order*(order-1)/2 << "' ";
    vcmd << "| sort | uniq | wc -l";
    FILE* fp = popen(vcmd.str().c_str(), "r");
    fscanf(fp, "%d", &var_removed);

    if (ext) {
        if (cutoff == 'v') {
            cutoffv = var_removed + 20;
        } else {
            cutoffv = cutoffv + 5;
        }
    }

    if (cutoff == 'd') {
        if (d >= cutoffv) {
            status = SOLVE;
            return UNKNOWN;
        }
    } else {
        if (var_removed >= cutoffv) {
            status = SOLVE;
            return UNKNOWN;
        }
    }
    status = CUBE;
    return UNKNOWN;
}

std::string Cube::gen_cube(std::string cubing_mode, int numMCTS) {
    std::stringstream cmd, mvcmd;
    if (cubing_mode == "march") {
        cmd << "./march/march_cu " << simp_inst << " ";
        cmd << "-d 1 -m " << order*(order-1)/2 << " ";
        cmd << "-o " << simp_inst << ".temp";
    } else {
        cmd << "python3 -u alpha-zero-general/main.py " << simp_inst << " ";
        cmd << "-d 1 -m " << order*(order-1)/2 << " ";
        cmd << "-o " << simp_inst << ".temp ";
        cmd << "-prod -numMCTSSims " << numMCTS;
    }
    system(cmd.str().c_str());
    d += 1;

    std::stringstream nc;
    nc << cnf << "." << id << index << ".cube";
    nextcube = nc.str();

    if (cube == "") {
        mvcmd << "mv " << simp_inst << ".temp " << nextcube;
    } else {
        mvcmd << "sed -E \"s/^a (.*)/$(head -n " << index << " " << cube;
        mvcmd << " | tail -n 1 | sed -E \'s/(.*) 0/\\1/\') \\1/\" ";
        mvcmd << simp_inst << ".temp > ";
        mvcmd << nextcube;
    }
    system(mvcmd.str().c_str());
    return nextcube;
}

int Cube::solve() {
    return 0;
}

std::string Cube::get_id() {
    std::stringstream i;
    i << id << index;
    return i.str();
}

int main(int argc, char** argv) {
    Cube c("instances/ks_17.cnf", "", "", 17, 0, 20, 0);
    c.apply();
    if (c.simplify(true, 'v') == UNSAT) { return 0; };
    c.gen_cube("ams", 10);
    Cube c1("instances/ks_17.cnf", c.get_nextcube(), c.get_id(), 17, 1, 20, 0);
    Cube c2("instances/ks_17.cnf", c.get_nextcube(), c.get_id(), 17, 2, 20, 0);
    c1.apply();
    if (c1.simplify(true, 'v') != UNSAT) {
        c1.gen_cube("ams", 10);
    }
    c2.apply();
    if (c2.simplify(true, 'v') != UNSAT) {
        c2.gen_cube("ams", 10);
    }
    return 0;

    int order = atoi(argv[2]);
    std::vector<Cube> workstack;
    Cube top(argv[1], "", "", order, 0, atoi(argv[3]), 0);
    
    
}