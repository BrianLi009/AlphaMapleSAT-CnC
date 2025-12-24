#include <iostream>
#include <cassert>
#include <cmath>
#include <queue>
#include <vector>
#include <ctime>

#include <mpi.h>

// CaDiCaL
#include "internal.hpp"
#include "signal.hpp"
#include "symbreak.hpp"

// dist-solve
#include "cube.hpp"
#include "def.hpp"
#include "progresstracker.hpp"

struct WInfo {
    int status;
    WInfo() { status = IDLE; }
};

class Manager {
    public:
        int nproc, nworkers, order;
        int num_solving, num_terminated;
        std::string top_name;

        ProgressTracker pt;
        std::queue<Cube> cube_queue;
        WInfo* winfo;

        void recv_cubes(int rank, int mode);
        void start();

        Manager(int nproc, int order, std::string top_name) {
            this->nproc = nproc;
            this->nworkers = nproc-1;
            this->order = order;
            this->top_name = top_name;
            winfo = new WInfo[nproc];
            num_solving = 0;
            num_terminated = 0;
        }
};

void Manager::recv_cubes(int rank, int mode) {
    std::string id1, id2;
    int count;
    MPI_Recv(&count, 1, MPI_INT, rank, CUBENUM, 
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (count) {
        id1 = recv_cube(rank, CUBEID);
        id2 = recv_cube(rank, CUBEID);
        Cube c1 = Cube(order, mode, id1, top_name);
        Cube c2 = Cube(order, mode, id2, top_name);
        cube_queue.push(c1);
        cube_queue.push(c2);
        fflush(stdout);
    }
}

void Manager::start() {
    Cube top_cube = Cube(order, CUBING, "", top_name);
    cube_queue.push(top_cube);
    int rank, count, flag;
    double progress;
    MPI_Status stat;

#if 1
    // cubing stage
    while (!cube_queue.empty() && (int) cube_queue.size() < nworkers) {
        int ncubes = std::min(cube_queue.size(), nworkers-cube_queue.size());
        for (int i = 0; i < ncubes; i++) {
            Cube work = cube_queue.front();
            isend_cube(i+1, work.str());
            cube_queue.pop();
        }
        for (int i = 0; i < ncubes; i++) {
            fflush(stdout);
            recv_cubes(i+1, CUBING);
            fflush(stdout);
        }
    }

#endif
    for (int i = 0; i < (int) cube_queue.size(); i++) {
        // set cubes to SOLVE mode
        Cube cube = cube_queue.front();
        cube_queue.pop();
        cube.status = SOLVING;
        cube_queue.push(cube);
    }

    // solving stage
    while (!cube_queue.empty() || num_solving) {
        // send cubes to solve
        for (int i = 1; i < nproc; i++) {
            if (winfo[i].status != IDLE) { continue; }
            if (!cube_queue.size()) { break; }
            isend_cube(i, cube_queue.front().str());
            cube_queue.pop();
            winfo[i].status = SOLVING;
            num_solving++;
            printf("c ACTIVE SOLVERS: %d/%d\n", num_solving, pt.size());
            fflush(stdout);
        }
        // if empty workers, try to interrupt
         if (num_solving + num_terminated < nworkers && (rank = pt.pop())) {
            MPI_Send(NULL, 0, MPI_INT, rank, 
                INTERRUPT, MPI_COMM_WORLD);
            winfo[rank].status = TERMINATED;
            pt.update(rank, 0);
            num_terminated++;
        }
        // check for completed solves
        for (;;) {
            flag = true;
            MPI_Iprobe(MPI_ANY_SOURCE, CUBENUM, 
                    MPI_COMM_WORLD, &flag, &stat);
            if (!flag) { break; }
            rank = stat.MPI_SOURCE;
            recv_cubes(rank, SOLVING);
            if (winfo[rank].status == SOLVING) {
                MPI_Send(NULL, 0, MPI_INT, rank, 
                    INTERRUPT, MPI_COMM_WORLD);
            } else if (winfo[rank].status == TERMINATED) {
                num_terminated--;
            } else {
                assert (false);
            }
            winfo[rank].status = IDLE;
            pt.update(rank, 0);
            num_solving--;
            printf("c ACTIVE SOLVERS: %d/%d\n", num_solving, pt.size());
            fflush(stdout);
        }
        // check for progress updates
        for (;;) {
            flag = true;
            MPI_Iprobe(MPI_ANY_SOURCE, PROGRESS, 
                    MPI_COMM_WORLD, &flag, &stat);
            if (!flag) { break; }
            rank = stat.MPI_SOURCE;
            MPI_Recv(&progress, 1, MPI_DOUBLE, rank, PROGRESS,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("c PROGRESS UPDATE: %d: %lf\n", rank, progress);
            fflush(stdout);
            pt.update(rank, progress);
        }
        fflush(stdout);
    }

    // done
    for (int i = 0; i < nproc-1; i++) {
        isend_cube(i+1, "");
    }
}

class Worker : public CaDiCaL::Terminator, public CaDiCaL::Handler {
    public:
        /*--------- Solver ----------*/
        CaDiCaL::Solver *solver; 
        Cube *cube;
        int res; // result
        volatile bool timesup = false;
        int time_limit; // time limit in seconds
        int max_var;

        int get(const char *o) { return solver->get (o); };
        bool set(const char *o, int v) { return solver->set (o, v); };
        bool set(const char *arg) { return solver->set_long_option (arg); };

        int split();
        int simplify();
        int solve();
        void write_file();
        void read_file(std::string name);

        /*--------- Terminator ----------*/
        int counter;
        double prev_progress;
        bool p_flag;
        bool terminate ();
        void catch_signal (int sig) { return; };
        void catch_alarm () { timesup = true; };

        /*--------- Message Handler ----------*/
        MPI_Request interrupt_req, progress_req; 

        /*--------- Worker ----------*/
        int state, rank;
        double start_ts;
        Worker() {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            solver = 0;
            counter = 0; 
            state = 0;
            prev_progress = 1.1;
        };
        ~Worker() { return; };
        void start();
        void send_cubeids(int count, int root);
};


void Worker::read_file(std::string name) {
    bool incremental;
    std::vector<int> cube_literals;
    solver->read_dimacs (name.c_str(), max_var, true, 
                            incremental, cube_literals);
}

void Worker::write_file() {
    std::string output_path = cube->name + ".simp";
    solver->write_dimacs (output_path.c_str(), max_var);
}

int Worker::simplify() {
    // setup solver
    assert (!solver);
    solver = new CaDiCaL::Solver ();
    //set("report", 1);
    CaDiCaL::Signal::set (this);
    CaDiCaL::Signal::alarm (TIMELIMIT);
    solver->limit ("proofsize", PROOFSIZE);
    solver->limit ("conflicts", SIMPLIMIT);

    // simplify
    std::string solving_file = cube->name;
    printf("c ----- SIMPLIFY -----\n");
    read_file(solving_file.c_str());
    SymmetryBreaker* se = new SymmetryBreaker(solver, cube->order, 0);
    max_var = solver->active ();
    res = solver->solve ();
    if (res == 0) { write_file(); }
    return res;
}

int Worker::solve() {
    // setup solver
    assert (!solver);
    solver = new CaDiCaL::Solver ();
    //set("report", 1);
    CaDiCaL::Signal::alarm (TIMELIMIT);
    CaDiCaL::Signal::set (this);
    solver->limit ("proofsize", PROOFSIZE);
    // terminiator for solving
    start_ts = clock();
    p_flag = false;
    solver->connect_terminator (this);

    // solve
    printf("c ----- SOLVE -----\n");
    read_file(cube->name.c_str());
    SymmetryBreaker* se = new SymmetryBreaker(solver, cube->order, 0);
    max_var = solver->active ();
    res = solver->solve ();
    if (res == 0) { write_file(); }
    printf("c ----- %d RESULT: ", rank);
    if (res == 0) {
        printf("UNKNOWN -----\n");
        fflush(stdout);
    } else if (res == 10) {
        printf("SATISFIABLE -----\n");
        fflush(stdout);
    } else {
        printf("UNSATISFIABLE -----\n"); 
        fflush(stdout);   
    }
    delete se;
    delete solver;
    solver = 0;
    return res;
}

bool Worker::terminate() {
    if (timesup) 
        return true;
    // check every NUMSKIP termination calls to reduce overhead
    if (counter % NUMSKIP == 0) {
        counter = 1;
        // send progress
        // warmup solving period, should not be preempted
        if ((clock() - start_ts) / CLOCKS_PER_SEC > WARMUP) {
            if (!p_flag || solver->progress() < prev_progress) {
                p_flag = true;
                prev_progress = solver->progress();
                MPI_Isend(&prev_progress, 1, MPI_DOUBLE, ROOT, PROGRESS, MPI_COMM_WORLD, &progress_req);
            }
        }
        // probe interrupt request
        int flag = false;
        MPI_Test(&interrupt_req, &flag, MPI_STATUS_IGNORE);
        return flag;
    } else {
        counter++;
        return false;
    }
}

int Worker::split() {
    // in order to generate cubes, we must call a python
    // program via a system call as this is the easiest
    // and most reliable method
    
    std::stringstream apply_cmd1, apply_cmd2;
    std::stringstream cube_cmd, cube_file;
    cube_file << cube->top_name;
    cube_file << "." << cube->id << ".cube";

    cube_cmd << "python3 -u alpha-zero-general/main.py ";
    cube_cmd << cube->name + ".simp ";
    cube_cmd << "-d 1 -m" << cube->order*(cube->order-1)/2 << " ";
    cube_cmd << "-o " << cube_file.str() << " ";
    cube_cmd << "-prod -numMCTSSims " << NUMMCTS;

    apply_cmd1 << "./gen_cubes/apply.sh ";
    apply_cmd1 << cube->name << ".simp " << cube_file.str() << " 1 > ";
    apply_cmd1 << cube->top_name << "." << cube->id << "1.cnf";

    apply_cmd2 << "./gen_cubes/apply.sh ";
    apply_cmd2 << cube->name << ".simp " << cube_file.str() << " 2 > ";
    apply_cmd2 << cube->top_name << "." << cube->id << "2.cnf";

    system(cube_cmd.str().c_str());
    system(apply_cmd1.str().c_str());
    system(apply_cmd2.str().c_str());
    return 2;
}

void Worker::start() {
    std::string cubestr;
    int count;
    // work loop
    for (;;) {
        state = IDLE;
        count = 0;

        // recieve cube
        cubestr = recv_cube(ROOT, CUBESTR);
        if (!cubestr.size()) { break; } // end
        cube = new Cube(cubestr);
        state = cube->status;

        // if forced cubing
        if (state == CUBING) {
            count = simplify() ? 0 : 2;
            if (count) split();
            send_cubeids(count, ROOT);
        }

        // if solve
        else if (state == SOLVING) {
            // setup interrupt request
            MPI_Irecv(NULL, 0, MPI_INT, ROOT, INTERRUPT, 
                MPI_COMM_WORLD, &interrupt_req);
            int res = solve();
            if (!res) { count = split(); }
            send_cubeids(count, ROOT);
            // clean up interrupt request
            MPI_Wait(&interrupt_req, MPI_STATUS_IGNORE); 
            //MPI_Request_free(&interrupt_req);
        }
    }
}

void Worker::send_cubeids(int count, int root) {
    MPI_Send(&count, 1, MPI_INT, ROOT, CUBENUM, MPI_COMM_WORLD);
    if (count) {
        // send cube ids
        std::string id1 = cube->id + "1";
        std::string id2 = cube->id + "2";
        MPI_Send(id1.c_str(), id1.size()+1, MPI_CHAR,
            ROOT, CUBEID, MPI_COMM_WORLD);
        MPI_Send(id2.c_str(), id2.size()+1, MPI_CHAR,
            ROOT, CUBEID, MPI_COMM_WORLD);
    }   
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        Manager manager(size, atoi(argv[2]), argv[1]);
        manager.start();
    } else {
        Worker worker;
        worker.start();
    }
    MPI_Finalize();
}
