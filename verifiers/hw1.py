from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import shutil
import concurrent
from analyzers.clang_remark_analyzer import OptimizationRemark
from verifiers.base_verifier import PerformanceVerifier
import os
import json
import logging
from tabulate import tabulate
from typing import Dict, Tuple, List
import subprocess
from utils import git_utils
from tqdm import tqdm


from analyzers.stack_analyzer import StackAnalyzer

logger = logging.getLogger()
class APEHW1(PerformanceVerifier):

    def get_remarks(self, branch="") -> List[str]:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            gen_remark_cmd = f"""
                pushd {self.codebase_dir} >/dev/null
                make clean && make
                popd >/dev/null
            """
            logger.debug(f"Running command (to generate remarks): {gen_remark_cmd}")
            result = subprocess.run(
                gen_remark_cmd,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.debug(f"gen_remark_output:\n{result.stdout}")
            if result.returncode != 0:
                logger.error(f"Build failed with error:\n{result.stderr}")

            yaml_contents = []
            for root, _, files in os.walk(self.codebase_dir):  # <-- search everything under codebase_dir
                for file in files:
                    if file.endswith(".opt.yaml"):
                        yaml_path = os.path.join(root, file)
                        try:
                            with open(yaml_path, 'r') as f:
                                yaml_contents.append(f.read())
                        except Exception as e:
                            logger.warning(f"Could not read {yaml_path}: {e}")

            logger.debug(f"Found {len(yaml_contents)} .opt.yaml files")

            os.system(f'sudo rm -rf {self.codebase_dir}/obj')

            if branch:
                git_utils.reset_hard(self.codebase_dir)

            return yaml_contents


    def __init__(self, codebase_dir: str):
        """
        Initializes the APEHW1 class with the directory of a codebase.
        """
        super().__init__(codebase_dir)

    def get_counter_value(self, file_path, event_name=None):
        try:
            event_map = {}
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line.strip())
                        if "event" in data and "counter-value" in data:
                            event_map[data["event"]] = float(data["counter-value"])
                    except json.JSONDecodeError:
                        continue
            
            if event_name:
                return event_map.get(event_name, "Event not found")
            return event_map
        except FileNotFoundError:
            return "File not found"
        except Exception as e:
            return f"An error occurred: {e}"

    def get_counters_from_directory(self, directory):
        result = {}
        for root, _, files in os.walk(directory):
            if "perfstat.json" in files:
                subdir_name = os.path.relpath(root, directory)
                file_path = os.path.join(root, "perfstat.json")
                result[subdir_name] = self.get_counter_value(file_path)
        return result
    
    def get_performance(self, branch="", n_iters=1, fn_list=[]) -> Tuple[Dict, bool]:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            # run and get output of performance generation script
            perf_cmd = f'(cd tmp/adiprerepa/cs598APE-hw1 && taskset -c 10 bash perfstatgen.sh -PG && cd ../../..)'
            
            agg_perf = None
            for i in range(n_iters):
                logger.debug(f"Running command: {perf_cmd}")
                # os.system(perf_cmd)
                result = subprocess.run(perf_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                perf_output = result.stdout.decode(errors='replace').strip()
                logger.debug(f"perf_output: {perf_output}")

                perf = self.get_counters_from_directory(f"{self.codebase_dir}/perf")

                tests_passed = self.tests_pass()
                if not tests_passed:
                    logger.error(f"Tests failed for branch {branch} after iteration {i+1}")
                    return perf, tests_passed

                if self.validate_performance(perf):
                    if agg_perf is None:
                        agg_perf = perf
                    else:
                        for k in agg_perf:
                            for k2 in agg_perf[k]:
                                if k2 in perf[k]:
                                    agg_perf[k][k2] = min(agg_perf[k][k2], perf[k][k2])                    
                    # aggregate target functions to "target_fns", which has a dict of function names to their exclusive time
                    for fn in fn_list:
                        for k in agg_perf:
                            if "target_fns" not in agg_perf[k]:
                                agg_perf[k]["target_fns"] = {}
                            fn_counter = self.get_counters_for_function_from_directory(f"{self.codebase_dir}/perf/{k}", fn)
                            if fn not in agg_perf[k]["target_fns"]:
                                agg_perf[k]["target_fns"][fn] = fn_counter
                            else:
                                agg_perf[k]["target_fns"][fn] = min(agg_perf[k]["target_fns"][fn], fn_counter)
                os.system(f'sudo rm -rf {self.codebase_dir}/perf')

            # If we switched to a different branch, stash any changes before returning
            if branch:
                os.system(f'git -C {self.codebase_dir} stash -q')

            return agg_perf, tests_passed



    def get_all_performance_parallel(self, branches: List[str], n_iters=1, fn="") -> Dict[str, Dict]:
        all_perf_data = {}
        dump_dirs = {}

        # Step 1: Prepare all dumps (normal copy since repos are small)
        for branch in branches:
            with git_utils.temp_checkout(branch, self.codebase_dir):
                dump_dir = f'/home/ayuram/pesquared/dumps/{branch}'
                if os.path.exists(dump_dir):
                    shutil.rmtree(dump_dir)
                shutil.copytree(self.codebase_dir, dump_dir)
                dump_dirs[branch] = dump_dir

        # Step 2: Setup CPU core partitioning
        total_cores = multiprocessing.cpu_count() * (2/3) # 2/3 of the cores
        max_parallel = total_cores

        logger.info(f"Total cores: {total_cores}, max parallel branches: {max_parallel}")

        # Step 3: Define the perf runner
        def run_perf(branch_id, branch, dump_dir):
            start_core = branch_id
            agg_perf = None
            successful_iters = 0
            one_test_passed = False
            for i in range(n_iters):
                perf_cmd = f'taskset -c {start_core} bash perfstatgen.sh -PG'
                full_cmd = f'(pushd {dump_dir} >/dev/null && {perf_cmd} ; popd >/dev/null)'
                logger.debug(f"Running command for {branch}: {full_cmd}")

                result = subprocess.run(['bash', '-c', full_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    logger.error(f"perfstatgen.sh failed for branch {branch}: {result.stderr.decode(errors="replace")}")
                    return branch, None, False

                perf_output = result.stdout.decode(errors="replace").strip()
                logger.debug(f"perf_output for {branch} (iter {i}): {perf_output}")

                perf = self.get_counters_from_directory(f"{dump_dir}/perf")
                if fn != "":
                    for k in perf:
                        perf[k]["target_fn_cycles"] = self.get_counters_for_function_from_directory(f"{dump_dir}/perf/{k}", fn)
                tests_passed = self.tests_pass()
                if tests_passed and self.validate_performance(perf):
                    one_test_passed = True
                    successful_iters += 1
                    if agg_perf is None:
                        agg_perf = perf
                    else:
                        for k in agg_perf:
                            for k2 in agg_perf[k]:
                                if k2 in perf[k]:
                                    agg_perf[k][k2] = min(agg_perf[k][k2], float(perf[k][k2]))
                                    # agg_perf[k][k2] += float(perf[k][k2])
            # if agg_perf is not None:
            #     for k in agg_perf:
            #         for k2 in agg_perf[k]:
            #             agg_perf[k][k2] /= successful_iters
                    # os.system(f'sudo rm -rf {dump_dir}/perf')

            return branch, agg_perf, one_test_passed

        futures = []
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(branches))) as executor:
            for idx, (branch, dump_dir) in enumerate(dump_dirs.items()):
                futures.append(executor.submit(run_perf, idx, branch, dump_dir))
                logger.info(f"Submitted perf job {branch} with core {idx}")

            for future in tqdm(as_completed(futures), total=len(futures), desc="Running perf jobs"):
                branch, perf, tests_passed = future.result()
                if perf is not None:
                    all_perf_data[branch] = perf
                    all_perf_data[branch]['tests_passed'] = tests_passed

        # Step 5: Clean up
        for branch, dump_dir in dump_dirs.items():
            if os.path.exists(dump_dir):
                shutil.rmtree(dump_dir)

        return all_perf_data

    

    def validate_performance(self, perf: dict) -> bool:
        """
        Validate that performance was measured correctly.
        
        Args:
            perf: Performance dictionary
            
        Returns:
            True if performance is valid, False otherwise
        """
        # if any value is 0/negative OR perf just has 'target_fns' and no other keys, then it is invalid
        return len(perf) >= 2 and len(perf["globe"]) > 1 and len(perf["pianoroom"]) > 1
        
        
    def get_compilation_error(self, branch="") -> str:
        """
        Get compilation error from the branch by running make in the container.
        
        Args:
            branch: Branch to check for compilation errors
            
        Returns:
            Compilation error message if error found, empty string if compilation succeeded
        """
        with git_utils.temp_checkout(branch, self.codebase_dir):
            # Run make command in the container to check for compilation errors
            compile_cmd = f'docker run --privileged -it -v ~/college/cs598ape/598APE-HW1:/host adiprerepa/598ape /bin/bash -c "cd /host && make -j 2>&1"'
            compile_output = os.popen(compile_cmd).read().strip()
            
            # Store the compilation result
            result = ""
            if "error" in compile_output.lower():
                # Extract approximately 20 lines around the error (10 before, 10 after)
                lines = compile_output.split('\n')
                error_indices = [i for i, line in enumerate(lines) if "error" in line.lower()]
                
                if error_indices:
                    # Use the first error occurrence
                    error_index = error_indices[0]
                    start_index = max(0, error_index - 10)
                    end_index = min(len(lines), error_index + 10)
                    
                    # Extract the relevant lines
                    result = '\n'.join(lines[start_index:end_index])
            
            return result


    def compare_performance(self, old: dict, new: dict) -> bool:
        """
        Returns True if the weighted average percent change of the given metric 
        across workloads (computed per workload first) is negative.
        """
        event = "cycles"  # Change this to any metric you want to compare

        # Define a custom weighting for each workload (values sum to 1)
        weight_map = {
            # "elephant": 0.33,
            "pianoroom": 0.5,
            "globe": 0.5
        }

        if len(old) != len(new):
            # did not compile possibly
            return False
        
        # if the length of any of the objects in new is 0, then it did not compile
        for workload in new:
            if len(new[workload]) == 0:
                return False

        weighted_percent_change = 0

        for workload, weight in weight_map.items():
            if workload in old and workload in new:
                old_value = float(old[workload][event])
                new_value = float(new[workload][event])

                if old_value == 0:
                    continue  # Avoid division by zero

                percent_change = ((new_value - old_value) / old_value) * 100
                weighted_percent_change += weight * percent_change

        return weighted_percent_change < 0  # Return True if net weighted percent change is negative



    def summarize_improvements(self, old: dict, new: dict, event="cycles") -> None:
        table_data = []
        
        for workload in old:
            if workload in new:
                try:
                    old_value = float(old[workload][event])
                    new_value = float(new[workload][event])
                    percent_change = ((new_value - old_value) / old_value) * 100
                    table_data.append([workload, old_value, new_value, f"{percent_change:.2f}%"])
                except (ValueError, KeyError) as e:
                    logger.error(f"Error processing workload {workload} in event '{event}': {e}")
            else:
                logger.warning(f"Workload: {workload} not found in new performance data")

        if table_data:
            headers = ["Workload", f"Old {event}", f"New {event}", "Percent Change"]
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            print(f"\nEvent: {event}\n{table}")  # Ensures event type is displayed


    def tests_pass(self, branch="") -> bool:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            output_dir = f"{self.codebase_dir}/output"
            baseline_dir = f"{self.codebase_dir}/baseline-outputs"
            
            # Check if both directories exist
            if not os.path.exists(output_dir) or not os.path.exists(baseline_dir):
                logger.error(f"Missing directories: outputs or baseline-outputs")
                return False
            
            # Get all .ppm files in both directories
            output_files = [f for f in os.listdir(output_dir) if f.endswith('.ppm')]
            baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith('.ppm')]
            
            # Compare each output file with its corresponding baseline file
            all_passed = True
            for output_file in output_files:
                # Check if the file exists in baseline-outputs
                if output_file not in baseline_files:
                    logger.error(f"File {output_file} not found in baseline-outputs")
                    all_passed = False
                    continue
                
                # Compare file contents
                with open(os.path.join(output_dir, output_file), 'rb') as f1, open(os.path.join(baseline_dir, output_file), 'rb') as f2:
                    if f1.read() != f2.read():
                        # logger.error(f"File {output_file} content doesn't match baseline")
                        all_passed = False
            
            return all_passed