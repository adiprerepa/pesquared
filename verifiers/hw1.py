from analyzers.clang_remark_analyzer import OptimizationRemark
from verifiers.base_verifier import PerformanceVerifier
import os
import json
import logging
from tabulate import tabulate
from typing import Dict, Tuple, List
from utils import git_utils

logger = logging.getLogger()
class APEHW1(PerformanceVerifier):

    def get_remarks(self, branch=""):
        logger.error("get_remarks not supported for APEHW1")
        pass

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
                            event_map[data["event"]] = data["counter-value"]
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
    
    def get_performance(self, branch="") -> Tuple[Dict, bool]:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            # run and get output of performance generation script
            perf_cmd = f'docker run --privileged -it -v ~/college/cs598ape/598APE-HW1:/host adiprerepa/598ape /bin/bash -c "cd /host && bash perfstatgen.sh -PEG"'
            
            logger.debug(f"Running command: {perf_cmd}")

            perf_output = os.popen(perf_cmd).read().strip()
            logger.debug(f"perf_output: {perf_output}")

            perf = self.get_counters_from_directory(f"{self.codebase_dir}/perf")

            tests_passed = self.tests_pass()

            os.system(f'sudo rm -rf {self.codebase_dir}/perf')

            # If we switched to a different branch, stash any changes before returning
            if branch:
                os.system(f'git -C {self.codebase_dir} stash -q')

            return perf, tests_passed
    

    def validate_performance(self, perf: dict) -> bool:
        """
        Validate that performance was measured correctly.
        
        Args:
            perf: Performance dictionary
            
        Returns:
            True if performance is valid, False otherwise
        """
        return len(perf) == 3 and all(len(v) > 0 for v in perf.values())
        
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
            "elephant": 0.33,
            "pianoroom": 0.33,
            "globe": 0.33
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
            baseline_dir = f"{self.codebase_dir}/pesquared-baseline-output"
            
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