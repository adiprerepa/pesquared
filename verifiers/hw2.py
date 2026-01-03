from verifiers.base_verifier import PerformanceVerifier
import os
import json
import logging
from tabulate import tabulate
from typing import Dict, Tuple, List
from utils import git_utils
# from analyzers.clang_remark_analyzer import OptimizationRemark, parse_optimization_summary
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

logger = logging.getLogger()
class APEHW2(PerformanceVerifier):

    def __init__(self, codebase_dir: str):
        """
        Initializes the APEHW2 class with the directory of a codebase.
        For symbolic machine learning evaluation.
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

    
    def get_performance(self, branch="", n_iters=1, fn_list=[]) -> Tuple[Dict, bool]:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            perf_cmd = f"""
                pushd {self.codebase_dir} >/dev/null
                taskset -c 10 bash perfstatgen.sh -D
                popd >/dev/null
            """

            agg_perf = None
            for i in range(n_iters):
                logger.debug(f"Running command:\n{perf_cmd}")
                result = subprocess.run(
                    perf_cmd,
                    shell=True,
                    executable="/bin/bash",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                perf_output = result.stdout.strip()
                logger.debug(f"perf_output:\n{perf_output}")
                if result.returncode != 0:
                    logger.error(f"perfstatgen.sh failed:\n{result.stderr}")
                    return {}, False

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

                    # Process target function counters
                    for fn in fn_list:
                        for k in agg_perf:
                            if "target_fns" not in agg_perf[k]:
                                agg_perf[k]["target_fns"] = {}
                            fn_counter = self.get_counters_for_function_from_directory(
                                f"{self.codebase_dir}/perf/{k}", fn
                            )
                            if fn not in agg_perf[k]["target_fns"]:
                                agg_perf[k]["target_fns"][fn] = fn_counter
                            else:
                                agg_perf[k]["target_fns"][fn] = min(agg_perf[k]["target_fns"][fn], fn_counter)

                os.system(f'sudo rm -rf {self.codebase_dir}/perf')

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
                perf_cmd = f'taskset -c {start_core} bash perfstatgen.sh -D'
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
        # For now, only checking diabetes dataset
        return "diabetes" in perf and len(perf["diabetes"]) > 0
        
    def get_compilation_error(self, branch="") -> str:
        """
        Get compilation error from the branch by running make in the container.
        
        Args:
            branch: Branch to check for compilation errors
            
        Returns:
            Compilation error message if error found, empty string if compilation succeeded
        """
        with git_utils.temp_checkout(branch, self.codebase_dir):
            # Run make command to check for compilation errors
            compile_cmd = f'cd {self.codebase_dir} && make -j 2>&1'
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
        Returns True if the percent change of the given metric for diabetes dataset is negative.
        """
        event = "cycles"  # Change this to any metric you want to compare

        if len(old) != len(new):
            # did not compile possibly
            return False
        
        # if diabetes is not in new or has 0 length, it did not compile
        if "diabetes" not in new or len(new.get("diabetes", {})) == 0:
            return False

        # Only compare diabetes workload for now
        if "diabetes" in old and "diabetes" in new:
            old_value = float(old["diabetes"][event])
            new_value = float(new["diabetes"][event])

            if old_value == 0:
                return False  # Avoid division by zero

            percent_change = ((new_value - old_value) / old_value) * 100
            return percent_change < 0  # Return True if percent change is negative
        
        return False


    def summarize_improvements(self, old: dict, new: dict, event="cycles") -> None:
        table_data = []
        
        # The full implementation would look at all datasets (diabetes, cancer, housing)
        # But for now, we're only evaluating the diabetes dataset
        if "diabetes" in old and "diabetes" in new:
            try:
                old_value = float(old["diabetes"][event])
                new_value = float(new["diabetes"][event])
                percent_change = ((new_value - old_value) / old_value) * 100
                table_data.append(["diabetes", old_value, new_value, f"{percent_change:.2f}%"])
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing diabetes dataset in event '{event}': {e}")
        else:
            logger.warning(f"Diabetes dataset not found in performance data")

        if table_data:
            headers = ["Dataset", f"Old {event}", f"New {event}", "Percent Change"]
            table = tabulate(table_data, headers=headers, tablefmt="grid")
            logger.info(f"\nEvent: {event}\n{table}")  # Ensures event type is displayed


    def tests_pass(self, branch="") -> bool:
        """
        Check if tests pass by comparing output files with golden files.
        For each file in the output directory, check if the corresponding
        golden file content is a substring of the output file content.
        
        Returns:
            bool: True if all tests pass, False otherwise
        """
        output_dir = os.path.join(self.codebase_dir, "output")
        golden_dir = os.path.join(self.codebase_dir, "golden")
        
        # Check if directories exist
        if not os.path.exists(output_dir) or not os.path.exists(golden_dir):
            logger.error(f"Output or golden directory not found")
            return False
            
        # Get all files in output directory
        for filename in os.listdir(output_dir):
            output_file_path = os.path.join(output_dir, filename)
            golden_file_path = os.path.join(golden_dir, filename)
            
            # Check if corresponding golden file exists
            if not os.path.exists(golden_file_path):
                logger.warning(f"No golden file found for {filename}")
                return False
                
            # Read file contents
            try:
                with open(output_file_path, 'r') as f:
                    a = f.read()
                with open(golden_file_path, 'r') as f:
                    b = f.read()
                    
                # Check if golden content is substring of output content
                if b not in a:
                    logger.error(f"Test failed for {filename}: golden content not in output")
                    return False
            except Exception as e:
                logger.error(f"Error comparing {filename}: {str(e)}")
                return False
                
        return True