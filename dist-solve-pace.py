import argparse
import os
import subprocess

import mpi4py
from mpi4py import MPI

def clean_files(files):
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}", flush=True)
        except OSError as e:
            print(f"Error: {e.strerror}. File: {file}", flush=True)

def simplify(file, order):
    command =f"./simplification/simplify-by-conflicts.sh {file} {order} 10000 -cas"
    subprocess.run(command, shell=True)

def gen_cube(instance, cube, index):
    command = f"./gen_cubes/apply.sh {instance} {cube} {index} > {cube}{index}.cnf"
    subprocess.run(command, shell=True)

def solve(file, order, timeout):
    command = f"./solve.sh {order} -cadical {timeout} -cas {file}"
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        stdout_str = stdout.decode()

        if "SATISFIABLE" in stdout_str:
            print("solved", flush=True)
            process.terminate()
            return True
        else:
            print("Continue cubing this subproblem...", flush=True)
            return False
    except Exception as e:
        print(f"Failed to run command due to: {str(e)}", flush=True)

def cube(instance, cube, index, m, order, numMCTS, timeout, cutoff="d", cutoffv=5, d=0, cubing_mode="march", extension="False"):
    if cube == "N":
        simplify(instance, order)
        file_to_cube = f"{instance}.simp"
        simplog_file = f"{instance}.simplog"
        file_to_check = f"{instance}.ext"
    else:
        gen_cube(instance, cube, index)
        simplify(f"{cube}{index}.cnf", order)
        file_to_cube = f"{cube}{index}.cnf.simp"
        simplog_file = f"{cube}{index}.cnf.simplog"
        file_to_check = f"{cube}{index}.cnf.ext"

    # Check if the output contains "c exit 20"
    with open(simplog_file, "r") as file:
        if "c exit 20" in file.read():
            print("the cube is UNSAT", flush=True)
            if cube != "N":
                files_to_remove = [f'{cube}{index}.cnf', file_to_cube, file_to_check]
                #remove_related_files(files_to_remove)
            return []
        
    command = f"sed -E 's/.* 0 [-]*([0-9]*) 0$/\\1/' < {file_to_check} | awk '$0<={m}' | sort | uniq | wc -l"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    var_removed = int(result.stdout.strip())
    if extension == "True":
        if cutoff == 'v':
            cutoffv = var_removed + 20
        else:
            cutoffv = cutoffv + 5

    print(f'{var_removed} variables removed from the cube', flush=True)

    if cutoff == 'd':
        if d >= cutoffv:
            files_to_remove = [f'{cube}{index}.cnf']
            clean_files(files_to_remove)
            solved = solve(file_to_cube, order, timeout)
            if solved:
                return []
            else:
                return [[file_to_cube, "N", 0, m, order, numMCTS, timeout, cutoff, cutoffv, 0, cubing_mode, "True"]]
    elif cutoff == 'v':
        if var_removed >= cutoffv:
            files_to_remove = [f'{cube}{index}.cnf']
            clean_files(files_to_remove)
            solved = solve(file_to_cube, order, timeout)
            if solved:
                return []
            else:
                return [[file_to_cube, "N", 0, m, order, numMCTS, timeout, cutoff, cutoffv, 0, cubing_mode, "True"]]

    

    # Select cubing method based on cubing_mode
    if cubing_mode == "march":
        subprocess.run(f"./march/march_cu {file_to_cube} -d 1 -m {m} -o {file_to_cube}.temp", shell=True)
    else:  # ams mode
        subprocess.run(f"python3 -u alpha-zero-general/main.py {file_to_cube} -d 1 -m {m} -o {file_to_cube}.temp -prod -numMCTSSims {numMCTS}", shell=True)

    # output {file_to_cube}.temp with the cubes
    d += 1
    if cube != "N":
        subprocess.run(f'''sed -E "s/^a (.*)/$(head -n {index} {cube} | tail -n 1 | sed -E 's/(.*) 0/\\1/') \\1/" {file_to_cube}.temp > {cube}{index}''', shell=True)
        next_cube = f'{cube}{index}'
    else:
        subprocess.run(f'mv {file_to_cube}.temp {instance}0', shell=True)
        next_cube = f'{instance}0'
    if cube != "N":
        files_to_remove = [
            f'{cube}{index}.cnf',
            f'{file_to_cube}.temp',
            file_to_cube,
            file_to_check
        ]
        clean_files(files_to_remove)
    else:
        files_to_remove = [file_to_cube, file_to_check]
        clean_files(files_to_remove)

    # return new jobs
    return [
        [instance, next_cube, 1, m, order, numMCTS, timeout, cutoff, cutoffv, d, cubing_mode],
        [instance, next_cube, 2, m, order, numMCTS, timeout, cutoff, cutoffv, d, cubing_mode]
    ]

def main(order, file_name_solve, m, solving_mode="other", cubing_mode="march", numMCTS=2, cutoff='d', cutoffv=5, solveaftercube='True', timeout=2147483647):
    """
    Parameters:
    - order: the order of the graph (required for satcas and exhaustive-no-cas modes)
    - file_name_solve: input file name
    - m: number of variables to consider for cubing (required)
    - solving_mode: 'satcas' (cadical simplification with cas, maplesat solving with cas) 
                   or 'exhaustive-no-cas' (cadical with exhaustive search)
                   or 'sms' (placeholder for SMS mode)
                   or 'other' (cadical simplification no cas, maplesat solving no cas)
    - cubing_mode: 'march' (use march_cu) or 'ams' (use alpha-zero-general)
    - numMCTS: number of MCTS simulations (only used with ams mode)
    - cutoff: 'd' for depth-based or 'v' for variable-based
    - cutoffv: cutoff value
    - solveaftercube: whether to solve after cubing
    - timeout: timeout in seconds (default: 1 hour)
    """
    # Validate input parameters
    if solving_mode not in ["satcas", "exhaustive-no-cas", "sms", "smsd2", "other"]:
        raise ValueError("solving_mode must be one of 'satcas', 'exhaustive-no-cas', 'sms', 'smsd2', or 'other'")
    if cubing_mode not in ["march", "ams"]:
        raise ValueError("cubing_mode must be either 'march' or 'ams'")
    if m is None:
        raise ValueError("m parameter must be specified")
    if (solving_mode == "satcas" or solving_mode == "exhaustive-no-cas" or solving_mode == "sms") and order is None:
        raise ValueError("order parameter must be specified when using satcas, exhaustive-no-cas, or sms mode")
    
    d = 0
    cutoffv = int(cutoffv)
    m = int(m)

    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank != 0:
        while (True):
            work = comm.recv(source=0, tag=0)
            if work:
                new_jobs = cube(*work)
                comm.send(new_jobs, dest=0, tag=0)
            else:
                print("Rank " + str(rank) +" done.", flush=True)               
                break
    else:
        requests = []
        status = MPI.Status()
        queue = [[file_name_solve, "N", 0, m, order, numMCTS, timeout, cutoff, cutoffv, d, cubing_mode]]
        workers = [0]*size

        while queue or requests:
            print(queue, flush=True)
            for i in range(1,size):
                if queue and workers[i] == 0:
                    job = queue.pop(0)
                    comm.isend(job, dest=i, tag=0)
                    requests.append(comm.irecv(source=i, tag=0))
                    workers[i] = 1
            if requests:
                jobs = MPI.Request.waitany(requests, status)
                requests.pop(jobs[0])
                workers[status.Get_source()] = 0
                queue.extend(jobs[1])
        for i in range(1, size):
            comm.send([], dest=i, tag=0)
    MPI.Finalize()

if __name__ == "__main__":
    mpi4py.rc.recv_mprobe = False
    parser = argparse.ArgumentParser(
        epilog='Example usage: python3 parallel-solve.py 17 instances/ks_17.cnf -m 136 --solving-mode satcas --cubing-mode ams --timeout 7200'
    )
    parser.add_argument('order', type=int, nargs='?', default=None, 
                        help='Order of the graph (required for satcas, exhaustive-no-cas, and sms modes)')
    parser.add_argument('file_name_solve', help='Input file name')
    parser.add_argument('-m', type=int, required=True,
                        help='Number of variables to consider for cubing')
    parser.add_argument('--solving-mode', choices=['satcas', 'exhaustive-no-cas', 'sms', 'smsd2', 'other'], default='other',
                        help='Solving mode: satcas (cadical+cas), exhaustive-no-cas (cadical+exhaustive), sms, smsd2, or other (default)')
    parser.add_argument('--cubing-mode', choices=['march', 'ams'], default='march',
                        help='Cubing mode: march (default) or ams (alpha-zero-general)')
    parser.add_argument('--numMCTS', type=int, default=2,
                        help='Number of MCTS simulations (only for ams mode)')
    parser.add_argument('--cutoff', choices=['d', 'v'], default='d',
                        help='Cutoff type: d (depth-based) or v (variable-based)')
    parser.add_argument('--cutoffv', type=int, default=5,
                        help='Cutoff value')
    parser.add_argument('--solveaftercube', choices=['True', 'False'], default='True',
                        help='Whether to solve after cubing')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout in seconds (default: 3600)')

    args = parser.parse_args()
    
    # Additional validation
    if (args.solving_mode == "satcas" or args.solving_mode == "exhaustive-no-cas" or args.solving_mode == "sms" or args.solving_mode == "smsd2") and args.order is None:
        parser.error("order parameter is required when using satcas, exhaustive-no-cas, sms, or smsd2 mode")

    main(args.order, args.file_name_solve, args.m, args.solving_mode, args.cubing_mode,
         args.numMCTS, args.cutoff, args.cutoffv, args.solveaftercube, args.timeout)

