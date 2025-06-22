import subprocess
import multiprocessing
import os
import queue
import argparse

remove_file = True

def run_command(command):
    process_id = os.getpid()
    print(f"Process {process_id}: Executing command: {command}", flush=True)

    file_to_cube = command.split()[-1]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        stdout_str = stdout.decode()

        if stderr:
            print(f"Error executing command: {stderr.decode()}", flush=True)

        # Check for SMS mode completion
        if (solving_mode_g == "sms" or solving_mode_g == "smsd2") and "Search finished" in stdout_str:
            print("solved", flush=True)
            process.terminate()
        # Check for SAT solver completion
        elif "UNSATISFIABLE" in stdout_str:
            print("solved", flush=True)
            process.terminate()
        elif "SATISFIABLE" in stdout_str:
            print("solved", flush=True)
            process.terminate()
        else:
            print("Continue cubing this subproblem...", flush=True)
            command = f"cube('{file_to_cube}', 'N', 0, {mg}, '{orderg}', {numMCTSg}, queue, '{cutoffg}', {cutoffvg}, {dg}, 'True')"
            queue.put(command)

    except Exception as e:
        print(f"Failed to run command due to: {str(e)}", flush=True)

def run_cube_command(command):
    print(command, flush=True)
    eval(command)

def remove_related_files(files_to_remove):
    global remove_file
    if not remove_file:
        return
        
    for file in files_to_remove:
        try:
            os.remove(file)
            print(f"Removed: {file}", flush=True)
        except OSError as e:
            print(f"Error: {e.strerror}. File: {file}", flush=True)

def rename_file(filename):
    # Remove .simp from file name
    
    if filename.endswith('.simp'):
        filename = filename[:-5]
    
    return filename
    
def worker(queue):
    while True:
        args = queue.get()
        if args is None:
            queue.task_done()
            break
        if args.startswith("./solve"):
            run_command(args)
        else:
            run_cube_command(args)
        queue.task_done()

def cube(original_file, cube, index, m, order, numMCTS, queue, cutoff='d', cutoffv=5, d=0, extension="False"):
    global solving_mode_g, cubing_mode_g
    
    if cube != "N":
        if solving_mode_g == "satcas":
            command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf && ./simplification/simplify-by-conflicts.sh {cube}{index}.cnf {order} 10000 -cas"
        elif solving_mode_g == "exhaustive-no-cas":
            command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf && ./simplification/simplify-by-conflicts.sh {cube}{index}.cnf {order} 10000 -exhaustive-no-cas"
        elif solving_mode_g == "sms":
            command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf && ./simplification/simplify-by-conflicts.sh {cube}{index}.cnf {order} 10000 -sms"
        elif solving_mode_g == "smsd2":
            command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf && ./simplification/simplify-by-conflicts.sh {cube}{index}.cnf {order} 10000 -smsd2"
        else:
            command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf && ./simplification/simplify-by-conflicts.sh {cube}{index}.cnf {order} 10000"
        file_to_cube = f"{cube}{index}.cnf.simp"
        simplog_file = f"{cube}{index}.cnf.simplog"
        file_to_check = f"{cube}{index}.cnf.ext"
    else:
        if solving_mode_g == "satcas":
            command = f"./simplification/simplify-by-conflicts.sh {original_file} {order} 10000 -cas"
        elif solving_mode_g == "exhaustive-no-cas":
            command = f"./simplification/simplify-by-conflicts.sh {original_file} {order} 10000 -exhaustive-no-cas"
        elif solving_mode_g == "sms":
            command = f"./simplification/simplify-by-conflicts.sh {original_file} {order} 10000 -sms"
        elif solving_mode_g == "smsd2":
            command = f"./simplification/simplify-by-conflicts.sh {original_file} {order} 10000 -smsd2"
        else:
            command = f"./simplification/simplify-by-conflicts.sh {original_file} {order} 10000"
        file_to_cube = f"{original_file}.simp"
        simplog_file = f"{original_file}.simplog"
        file_to_check = f"{original_file}.ext"
    subprocess.run(command, shell=True)
    # Remove the cube file after it's been used
    #remove_related_files([cube])

    # Check if the output contains "c exit 20"
    with open(simplog_file, "r") as file:
        if "c exit 20" in file.read():
            print("the cube is UNSAT", flush=True)
            if cube != "N":
                files_to_remove = [f'{cube}{index}.cnf', file_to_cube, file_to_check]
                #remove_related_files(files_to_remove)
            return
    
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
            if solveaftercubeg == 'True':
                files_to_remove = [f'{cube}{index}.cnf']
                remove_related_files(files_to_remove)
                if solving_mode_g == "satcas":
                    command = f"./solve.sh {order} -cadical {timeout_g} -cas {file_to_cube}"
                elif solving_mode_g == "exhaustive-no-cas":
                    command = f"./solve.sh {order} -cadical {timeout_g} -exhaustive-no-cas {file_to_cube}"
                elif solving_mode_g == "sms":
                    if cube != "N":
                        command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf"
                        subprocess.run(command, shell=True)
                        command = f"./solve.sh {order} -cadical {timeout_g} -sms {cube}{index}.cnf"
                    else:
                        command = f"./solve.sh {order} -cadical {timeout_g} -sms {original_file}"
                elif solving_mode_g == "smsd2":
                    if cube != "N":
                        command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf"
                        subprocess.run(command, shell=True)
                        command = f"./solve.sh {order} -cadical {timeout_g} -smsd2 {cube}{index}.cnf"
                    else:
                        command = f"./solve.sh {order} -cadical {timeout_g} -smsd2 {original_file}"
                else:
                    command = f"./solve.sh {order} -cadical {timeout_g} {file_to_cube}"
                queue.put(command)
            return
    if cutoff == 'v':
        if var_removed >= cutoffv:
            if solveaftercubeg == 'True':
                files_to_remove = [f'{cube}{index}.cnf']
                remove_related_files(files_to_remove)
                if solving_mode_g == "satcas":
                    command = f"./solve.sh {order} -cadical {timeout_g} -cas {file_to_cube}"
                elif solving_mode_g == "exhaustive-no-cas":
                    command = f"./solve.sh {order} -cadical {timeout_g} -exhaustive-no-cas {file_to_cube}"
                elif solving_mode_g == "sms":
                    if cube != "N":
                        command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf"
                        subprocess.run(command, shell=True)
                        command = f"./solve.sh {order} -cadical {timeout_g} -sms {cube}{index}.cnf"
                    else:
                        command = f"./solve.sh {order} -cadical {timeout_g} -sms {original_file}"
                elif solving_mode_g == "smsd2":
                    if cube != "N":
                        command = f"./gen_cubes/apply.sh {original_file} {cube} {index} > {cube}{index}.cnf"
                        subprocess.run(command, shell=True)
                        command = f"./solve.sh {order} -cadical {timeout_g} -smsd2 {cube}{index}.cnf"
                    else:
                        command = f"./solve.sh {order} -cadical {timeout_g} -smsd2 {original_file}"
                else:
                    command = f"./solve.sh {order} -cadical {timeout_g} {file_to_cube}"
                queue.put(command)
            return

    # Select cubing method based on cubing_mode
    if cubing_mode_g == "march":
        subprocess.run(f"./march/march_cu {file_to_cube} -d 1 -m {m} -o {file_to_cube}.temp", shell=True)
    else:  # ams mode
        subprocess.run(f"python3 -u AlphaMapleSAT/alphamaplesat/main.py {file_to_cube} -d 1 -m {m} -o {file_to_cube}.temp -prod -numMCTSSims {numMCTS}", shell=True)
        #subprocess.run(f"python3 -u AlphaMapleSAT/alphamaplesat/main.py {file_to_cube} -d 1 -m {m} -o {file_to_cube}.temp -order {order} -prod -numMCTSSims {numMCTS}", shell=True)

    #output {file_to_cube}.temp with the cubes
    d += 1
    if cube != "N":
        subprocess.run(f'''sed -E "s/^a (.*)/$(head -n {index} {cube} | tail -n 1 | sed -E 's/(.*) 0/\\1/') \\1/" {file_to_cube}.temp > {cube}{index}''', shell=True)
        next_cube = f'{cube}{index}'
    else:
        subprocess.run(f'mv {file_to_cube}.temp {original_file}0', shell=True)
        next_cube = f'{original_file}0'
    if cube != "N":
        files_to_remove = [
            f'{cube}{index}.cnf',
            f'{file_to_cube}.temp',
            file_to_cube,
            file_to_check
        ]
        remove_related_files(files_to_remove)
    else:
        files_to_remove = [file_to_cube, file_to_check]
        remove_related_files(files_to_remove)
    command1 = f"cube('{original_file}', '{next_cube}', 1, {m}, '{order}', {numMCTS}, queue, '{cutoff}', {cutoffv}, {d})"
    command2 = f"cube('{original_file}', '{next_cube}', 2, {m}, '{order}', {numMCTS}, queue, '{cutoff}', {cutoffv}, {d})"
    queue.put(command1)
    queue.put(command2)

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

    # Update global variables
    global queue, orderg, numMCTSg, cutoffg, cutoffvg, dg, mg, solveaftercubeg, file_name_solveg, solving_mode_g, cubing_mode_g, timeout_g
    orderg, numMCTSg, cutoffg, cutoffvg, dg, mg, solveaftercubeg, file_name_solveg = order, numMCTS, cutoff, cutoffv, d, m, solveaftercube, file_name_solve
    solving_mode_g = solving_mode
    cubing_mode_g = cubing_mode
    timeout_g = timeout

    queue = multiprocessing.JoinableQueue()
    num_worker_processes = multiprocessing.cpu_count()

    # Start worker processes
    processes = [multiprocessing.Process(target=worker, args=(queue,)) for _ in range(num_worker_processes)]
    for p in processes:
        p.start()

    #file_name_solve is a file where each line is a filename to solve
    with open(file_name_solve, 'r') as file:
        first_line = file.readline().strip()  # Read the first line and strip whitespace

        # Check if the first line starts with 'p cnf'
        if first_line.startswith('p cnf'):
            print("input file is a CNF file", flush=True)
            cube(file_name_solve, "N", 0, m, order, numMCTS, queue, cutoff, cutoffv, d)
        else:
            print("input file contains name of multiple CNF file, solving them first", flush=True)
            # Prepend the already read first line to the list of subsequent lines
            instance_lst = [first_line] + [line.strip() for line in file]
            for instance in instance_lst:
                if solving_mode_g == "satcas":
                    command = f"./solve.sh {order} -cadical {timeout_g} -cas {instance}"
                elif solving_mode_g == "exhaustive-no-cas":
                    command = f"./solve.sh {order} -cadical {timeout_g} -exhaustive-no-cas {instance}"
                elif solving_mode_g == "sms":
                    command = f"./solve.sh {order} -cadical {timeout_g} -sms {instance}"
                elif solving_mode_g == "smsd2":
                    command = f"./solve.sh {order} -cadical {timeout_g} -smsd2 {instance}"
                else:
                    command = f"./solve.sh {order} -cadical {timeout_g} {instance}"
                queue.put(command)

    # Wait for all tasks to be completed
    queue.join()

    # Stop workers
    for _ in processes:
        queue.put(None)
    for p in processes:
        p.join()

if __name__ == "__main__":
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
