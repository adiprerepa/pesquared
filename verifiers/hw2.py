from verifiers.base_verifier import PerformanceVerifier
import os
import json
import logging
from tabulate import tabulate
from typing import Dict, Tuple, List
from utils import git_utils
from analyzers.clang_remark_analyzer import OptimizationRemark, parse_optimization_summary

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

    def get_remarks(self, branch="") -> List[OptimizationRemark]:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            gen_remark_cmd = f'docker run --privileged -it -v ~/college/cs598ape/598APE-HW2:/host adiprerepa/598ape /bin/bash -c "cd /host && make clean && make"'
            logger.debug(f"Running command (to generate remarks): {gen_remark_cmd}")
            gen_remark_output = os.popen(gen_remark_cmd).read().strip()
            logger.debug(f"gen_remark_output: {gen_remark_output}")

            remarks_cmd = f"python3 scripts/parse_clang_remarks.py -o {self.codebase_dir}/missed_remarks_tmp.yaml {self.codebase_dir}/obj"
            logger.debug(f"Running command (to parse remarks): {remarks_cmd}")
            remarks_output = os.popen(remarks_cmd).read().strip()
            logger.debug(f"remarks_output: {remarks_output}")

            remarks = parse_optimization_summary(f"{self.codebase_dir}/missed_remarks_tmp.yaml")
            logger.debug(f"got {len(remarks)} remarks")

            os.system(f'sudo rm -rf {self.codebase_dir}/obj')
            os.system(f'sudo rm {self.codebase_dir}/missed_remarks_tmp.yaml')

            # Reset hard before returning to original branch
            if branch:
                git_utils.reset_hard(self.codebase_dir)
            return remarks
    
    def get_performance(self, branch="") -> Tuple[Dict, bool]:
        with git_utils.temp_checkout(branch, self.codebase_dir):
            # Run perfreport.sh with -D flag (for diabetes dataset only)
            perf_cmd = f'docker run --privileged -it -v ~/college/cs598ape/598APE-HW2:/host adiprerepa/598ape /bin/bash -c "cd /host && bash perfstatgen.sh -D"'
            
            logger.debug(f"Running command: {perf_cmd}")

            perf_output = os.popen(perf_cmd).read().strip()
            logger.debug(f"perf_output: {perf_output}")

            perf = self.get_counters_from_directory(f"{self.codebase_dir}/perf")

            tests_passed = self.tests_pass()

            remarks_cmd = f"python3 scripts/parse_clang_remarks.py -o {self.codebase_dir}/missed_remarks_tmp.yaml {self.codebase_dir}/obj/"

            os.system(f'sudo rm -rf {self.codebase_dir}/perf')
            os.system(f'sudo rm -rf {self.codebase_dir}/obj')
            os.system(f'sudo rm {self.codebase_dir}/missed_remarks_tmp.yaml')
            
            # Reset hard before returning to original branch
            if branch:
                git_utils.reset_hard(self.codebase_dir)
            
            return perf, tests_passed
    

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