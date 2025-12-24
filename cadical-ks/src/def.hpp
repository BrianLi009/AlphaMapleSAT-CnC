#ifndef DEFHPP
#define DEFHPP

#define ROOT 0

#define NUMSKIP 100
#define WARMUP 30

#define SIMPLIMIT 10000
#define PROOFSIZE 7168
#define TIMELIMIT 7200
#define NUMMCTS 10

// worker state
enum { IDLE = 1, CUBING = 2, SOLVING = 3, TERMINATED = 4 };
// cube state
enum { UNKNOWN = 0, SAT = 10, UNSAT = 20 };
// message id
enum { PROGRESS = 100, INTERRUPT = 101, CUBESTR = 103, CUBENUM = 104, CUBEID = 105 };

#endif