from performance_verifier import PerformanceVerifier
import os
import json

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
        
        # run and get output of performance generation script
        perf_cmd = f'docker run --privileged -it -v ~/college/cs598ape/598APE-HW1:/host adiprerepa/598ape /bin/bash -c "cd /host && bash perfstatgen.sh -PEG"'
        
        perf_output = os.popen(perf_cmd).read().strip()
        print(perf_output)

        perf = self.get_counters_from_directory(f"{self.codebase_dir}/perf")

        os.system(f'sudo rm -rf {self.codebase_dir}/perf')

        if branch != "":
            os.system(f'git -C {self.codebase_dir} stash && git -C {self.codebase_dir} checkout {current_branch}')

        return perf


    def compare_performance(self, old: dict, new: dict) -> bool:
        """
        Using cycles as performance metric -- maybe we should use task clock.
        """
        for run in old:
            if float(new[run]['cycles']) >= float(old[run]['cycles']):
                return False
        return True

    def tests_pass(self, branch) -> bool:
        pass