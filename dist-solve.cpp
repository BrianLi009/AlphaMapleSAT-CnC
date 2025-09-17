#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <queue>

#include <time.h>

#include <mpi.h>

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
        std::string index;
        int order;
        int cutoffv;
        int d;
        int status;
    public:
        Cube(std::string cnf, int order, std::string id, std::string index, int cutoffv, int d, int status);
        Cube(std::string& cubestr, int order);
        void apply();
        std::string execute(std::string cubing_mode, int numMCTS, bool ext, char cutoff, int timeout);
        int simplify(bool ext, char cutoff);
        void gen_cube(std::string cubing_mode, int numMCTS);
        int solve(int timeout);
        std::string get_id();
        std::string get_instance() { return instance; };
        int get_status() { return status; };
        void set_status(int status) { this->status = status; };
        std::string str();
};

// initialize Cube
Cube::Cube(std::string cnf, int order, std::string id, std::string index, int cutoffv, int d, int status) {
    this->cnf = cnf;
    this->order = order;
    if (id == "") {
        this->cube = "";
    } else {
        this->cube = cnf + "." + id + ".cube";
    } 
    this->id = id;
    this->index = index;
    this->cutoffv = cutoffv;
    this->d = d;
    this->status = status;
}

// consume Cube from cube string
Cube::Cube(std::string& cubestr, int order) {
    std::stringstream iss(cubestr);
    std::getline(iss, cnf, ' ');
    std::getline(iss, id, ' ');
    std::getline(iss, index, ' ');
    std::string temp;
    std::getline(iss, temp, ' ');
    cutoffv = atoi(temp.c_str());
    std::getline(iss, temp, ' ');
    d = atoi(temp.c_str());
    std::getline(iss, temp, ' ');
    status = atoi(temp.c_str());
    if (id == "") {
        this->cube = "";
    } else {
        this->cube = cnf + "." + id + ".cube";
    }
    this->order = order;
    //printf("%s %s %s %s %d %d %d\n", cnf.c_str(), cube.c_str(), id.c_str(), index.c_str(), cutoffv, d, status);
}

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

    //printf("%s\n", cmd.str().c_str());
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

void Cube::gen_cube(std::string cubing_mode, int numMCTS) {
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
}

int Cube::solve(int timeout) {
    std::stringstream cmd;
    cmd << "./solve.sh " << order << " -cadical " << timeout << " -cas " << simp_inst;
    int res = system(cmd.str().c_str());
    return WEXITSTATUS(res);
}

std::string Cube::get_id() {
    std::stringstream i;
    i << id << index;
    return i.str();
}

std::string Cube::str() {
    std::stringstream ss;
    ss << cnf << " ";
    ss << id << " " << index << " ";
    ss << cutoffv << " " << d << " " << status;
    return ss.str();
}

std::string Cube::execute(std::string cubing_mode, int numMCTS, bool ext, char cutoff, int timeout) {
    // force cubing
    if (status == CUBE) {
        apply();
        if (simplify(ext, cutoff) == UNSAT) {
            return "";
        }
        gen_cube(cubing_mode, numMCTS);
        Cube c1(cnf, order, get_id(), "1", cutoffv, d, UNKNOWN);
        Cube c2(cnf, order, get_id(), "2", cutoffv, d, UNKNOWN);
        std::string sendstr = c1.str() + " ," + c2.str();
        return sendstr;
    }

    // force solving
    else if (status == SOLVE) {
        apply();
        int res = solve(timeout);
        return "";
    }

    // otherwise do usual simplification + solve/cube
    else {
        apply();
        if (simplify(ext, cutoff) == UNSAT) {
            return "";
        }
        if (status == CUBE) {
            gen_cube(cubing_mode, numMCTS);
            Cube c1(cnf, order, get_id(), "1", cutoffv, d, UNKNOWN);
            Cube c2(cnf, order, get_id(), "2", cutoffv, d, UNKNOWN);
            std::string sendstr = c1.str() + " ," + c2.str();
            return sendstr;
        }
        else {
            int res = solve(timeout);
            if (res) {
                return "";
            } else {
                gen_cube(cubing_mode, numMCTS);
                Cube c1(cnf, order, get_id(), "1", cutoffv, d, UNKNOWN);
                Cube c2(cnf, order, get_id(), "2", cutoffv, d, UNKNOWN);
                std::string sendstr = c1.str() + " ," + c2.str();
                return sendstr;
            }
        }
    }
}

void send_cube(int dst, std::string cube) {
    int sendlen = strlen(cube.c_str()) + 1;
    MPI_Send(&sendlen, 1, MPI_INT, dst, 0, MPI_COMM_WORLD);
    MPI_Send(cube.c_str(), sendlen, MPI_CHAR, dst, 1, MPI_COMM_WORLD);
}

std::string recv_cube(int src) {
    int recvlen;
    MPI_Recv(&recvlen, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char* recvdata = new char[recvlen];
    MPI_Recv(recvdata, recvlen, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return std::string(recvdata);
}

void isend_cube(int dst, std::string cube) {
    int sendlen = strlen(cube.c_str()) + 1;
    MPI_Request req;
    MPI_Isend(&sendlen, 1, MPI_INT, dst, 0, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
    MPI_Isend(cube.c_str(), sendlen, MPI_CHAR, dst, 1, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
}


int main(int argc, char** argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int order = atoi(argv[2]);

    if (rank == 0) {
        std::queue<Cube> cubestack;
        std::vector<bool> workers;
        std::vector<MPI_Request> requests;
        std::vector<int> recvlen;
        workers.resize(size);
        requests.resize(size);
        recvlen.resize(size);

        auto start = MPI_Wtime();

        int working = 0;

        // top level problem
        cubestack.push(Cube(argv[1], order, "", "0", atoi(argv[3]), 0, UNKNOWN));
        // initiate requests
        for (int i = 0; i < size; i++) {
            requests[i] = MPI_REQUEST_NULL;
        }

        bool startup = true;
        int ncubes = 1;

        for (;;) {
            // get first available worker
            for (int i = 1; i < size; i++) {
                if (cubestack.size() == 0) {
                    break;
                }
                if (!workers[i]) {
                    working++;
                    std::string cubestr = cubestack.front().str();
                    cubestack.pop();
                    Cube cube(cubestr, order);
                    if (startup && ncubes < size-1) {
                        //printf("forcing cubing of cube '%s'\n", cube.str().c_str());
                        cube.set_status(CUBE);
                    }
                    if (startup && ncubes >= size-1) {
                        startup = false;
                    }
                    ncubes--;
                    isend_cube(i, cube.str());
                    workers[i] = true;
                    MPI_Irecv(&(recvlen.data()[i]), 1, MPI_INT, i, 0, MPI_COMM_WORLD, &(requests.data()[i]));
                }
            }

            auto now = MPI_Wtime();
            printf("(WORKERTIMESTAMP) %.2f %d\n", now-start, working);

            // check if recieved completion from workers
            int idx;
            //printf("rank %d waiting...\n", rank);
            MPI_Waitany(size, requests.data(), &idx, MPI_STATUS_IGNORE);
            if (idx == MPI_UNDEFINED) {
                break;
            }
            workers[idx] = false;
            char *recvdata = new char[recvlen[idx]];
            MPI_Recv(recvdata, recvlen[idx], MPI_CHAR, idx, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::string newcubes = std::string(recvdata);
            std::stringstream ss = std::stringstream(newcubes);
            //printf("rank %d recieved \"%s\"\n", rank, newcubes.c_str());
            delete[] recvdata;
            working--;

            // add new cubes to queue
            if (newcubes != "") {
                while (!ss.eof()) {
                    std::string cube;
                    std::getline(ss, cube, ',');
                    cubestack.push(Cube(cube, order));
                    ncubes++;
                }
            }
        }
        // done
        printf("all cubes solved\n");
        for (int i = 1; i < size; i++) {
            isend_cube(i, "");
        }
    }

    else {
        for (;;) {
            std::string cubestr = recv_cube(0);
            //printf("rank %d recieved \"%s\"\n", rank, cubestr.c_str());
            if (cubestr.size() == 0) {
                break;
            }
            Cube cube(cubestr, order);
            std::string newcubes = cube.execute("ams", 10, false, 'v', 3600);
            //printf("rank %d sending \"%s\"\n", rank, newcubes.c_str());
            send_cube(0, newcubes);
        }
    }
    
    MPI_Finalize();
    return 0;
}