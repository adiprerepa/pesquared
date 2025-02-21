from performance_verifier import PerformanceVerifier
import os
import json
import logging
from tabulate import tabulate

logger = logging.getLogger()
class APEHW1(PerformanceVerifier):

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
    
    def get_performance(self, branch=""):
        # Save current branch name 
        current_branch = os.popen('git -C {} rev-parse --abbrev-ref HEAD'.format(self.codebase_dir)).read().strip()
        
        if branch != "":
            # checkout branch
            branch_cmd = f'git -C {self.codebase_dir} checkout {branch}'
            if os.system(branch_cmd) != 0:
                raise ValueError("Failed to create and checkout new branch")
        

        # todo we can potentially add a separate check to see if the code even compiles (make -j).
        # run and get output of performance generation script
        perf_cmd = f'docker run --privileged -it -v ~/college/cs598ape/598APE-HW1:/host adiprerepa/598ape /bin/bash -c "cd /host && bash perfstatgen.sh -PEG"'
        
        logger.debug(f"Running command: {perf_cmd}")

        perf_output = os.popen(perf_cmd).read().strip()
        logger.debug(f"perf_output: {perf_output}")
        # print(perf_output)

        perf = self.get_counters_from_directory(f"{self.codebase_dir}/perf")

        os.system(f'sudo rm -rf {self.codebase_dir}/perf')

        if branch != "":
            os.system(f'git -C {self.codebase_dir} stash -q && git -C {self.codebase_dir} checkout {current_branch}')

        return perf
    

    def validate_performance(self, perf: dict) -> bool:
        return len(perf) == 3


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


    def tests_pass(self, branch) -> bool:
        pass