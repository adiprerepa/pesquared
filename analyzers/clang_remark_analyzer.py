import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Type
import sys
from collections import defaultdict

# --- Helper Classes for Structured Data ---

class SourceLocation:
    """Represents a location in source code."""
    __slots__ = ('file', 'line', 'column') # Memory optimization

    def __init__(self, loc_dict: Optional[Dict[str, Any]]):
        if loc_dict is None:
            loc_dict = {}
        self.file: Optional[str] = loc_dict.get('File')
        # Ensure line/column are integers, default to 0 or None if missing/invalid
        try:
            self.line: int = int(loc_dict.get('Line', 0))
        except (ValueError, TypeError):
            self.line: int = 0
        try:
            self.column: int = int(loc_dict.get('Column', 0))
        except (ValueError, TypeError):
            self.column: int = 0

    def __repr__(self) -> str:
        if self.file:
            return f"SourceLocation(file='{self.file}', line={self.line}, column={self.column})"
        else:
            return "SourceLocation(None)"

    def __str__(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}:{self.column}"
        else:
            return "<unknown location>"

class DetailItemWithLocation:
    """Represents a detail item (like GVN ClobberedBy) with a type and optional location."""
    __slots__ = ('type', 'location')

    def __init__(self, detail_dict: Optional[Dict[str, Any]]):
        if detail_dict is None:
            detail_dict = {}
        self.type: Optional[str] = detail_dict.get('Type')
        self.location: Optional[SourceLocation] = SourceLocation(detail_dict.get('DebugLoc')) if 'DebugLoc' in detail_dict else None

    def __repr__(self) -> str:
        return f"DetailItemWithLocation(type='{self.type}', location={self.location!r})"

class CallerCalleeInfo:
    """Represents Caller/Callee information for inlining."""
    __slots__ = ('name', 'location')

    def __init__(self, detail_dict: Optional[Dict[str, Any]]):
        if detail_dict is None:
            detail_dict = {}
        self.name: Optional[str] = detail_dict.get('Name')
        self.location: Optional[SourceLocation] = SourceLocation(detail_dict.get('DebugLoc')) if 'DebugLoc' in detail_dict else None

    def __repr__(self) -> str:
        return f"CallerCalleeInfo(name='{self.name}', location={self.location!r})"

# --- Base and Subclasses for Optimization Remarks ---

class OptimizationRemark:
    """Base class for a single optimization remark."""
    def __init__(self, remark_dict: Dict[str, Any]):
        self.pass_name: str = remark_dict.get('Optimization_Pass', 'N/A')
        self.miss_reason: str = remark_dict.get('Miss_Reason', 'N/A')
        self.function: str = remark_dict.get('Function', 'N/A')
        self.location: SourceLocation = SourceLocation(remark_dict.get('Source_Location'))
        self.original_tag: str = remark_dict.get('Original_Tag', 'N/A')
        self.source_yaml: str = remark_dict.get('Source_Yaml', 'N/A')

        self.raw_details: Dict[str, Any] = remark_dict.get('Details', {})
        # Common detail fields
        self.info: Optional[str] = self.raw_details.get('Info')

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pass='{self.pass_name}', reason='{self.miss_reason}', func='{self.function}', loc='{self.location}')"

    @property
    def file(self) -> Optional[str]:
        """Convenience property to get the file path."""
        return self.location.file

    @property
    def line(self) -> int:
        """Convenience property to get the line number."""
        return self.location.line


class InlineRemark(OptimizationRemark):
    """Remark specific to the 'inline' pass."""
    def __init__(self, remark_dict: Dict[str, Any]):
        super().__init__(remark_dict)
        details = self.raw_details

        self.caller: Optional[CallerCalleeInfo] = CallerCalleeInfo(details.get('Caller')) if 'Caller' in details else None
        self.callee: Optional[CallerCalleeInfo] = CallerCalleeInfo(details.get('Callee')) if 'Callee' in details else None

        # Handle Cost/Threshold which might be number or {'Value': N, 'DebugLoc': ...}
        cost_data = details.get('Cost')
        self.cost: Optional[Union[int, float]] = None
        self.cost_location: Optional[SourceLocation] = None
        if isinstance(cost_data, dict) and 'Value' in cost_data:
            self.cost = cost_data.get('Value')
            self.cost_location = SourceLocation(cost_data.get('DebugLoc'))
        elif cost_data is not None:
            self.cost = cost_data # Already parsed as number by previous script

        thresh_data = details.get('Threshold') # Handles 'Treshold' correction from prev script
        self.threshold: Optional[Union[int, float]] = None
        self.threshold_location: Optional[SourceLocation] = None
        if isinstance(thresh_data, dict) and 'Value' in thresh_data:
            self.threshold = thresh_data.get('Value')
            self.threshold_location = SourceLocation(thresh_data.get('DebugLoc'))
        elif thresh_data is not None:
            self.threshold = thresh_data

        self.reason_value: Optional[str] = None # If Reason was {'Value': ..., 'DebugLoc': ...}
        self.reason_location: Optional[SourceLocation] = None
        reason_data = details.get('Reason')
        if isinstance(reason_data, dict) and 'Value' in reason_data:
             self.reason_value = reason_data.get('Value')
             self.reason_location = SourceLocation(reason_data.get('DebugLoc'))
        elif isinstance(reason_data, str):
             self.reason_value = reason_data


class GVNRemark(OptimizationRemark):
    """Remark specific to the 'gvn' pass."""
    def __init__(self, remark_dict: Dict[str, Any]):
        super().__init__(remark_dict)
        details = self.raw_details
        self.clobbered_by: Optional[DetailItemWithLocation] = DetailItemWithLocation(details.get('ClobberedBy')) if 'ClobberedBy' in details else None
        self.other_access: Optional[DetailItemWithLocation] = DetailItemWithLocation(details.get('OtherAccess')) if 'OtherAccess' in details else None
        self.type: Optional[str] = details.get('Type')


class RegallocRemark(OptimizationRemark):
    """Remark specific to the 'regalloc' pass."""
    def __init__(self, remark_dict: Dict[str, Any]):
        super().__init__(remark_dict)
        details = self.raw_details
        self.num_vr_copies: Optional[Union[int, float]] = details.get('NumVRCopies')
        self.total_copies_cost: Optional[Union[int, float]] = details.get('TotalCopiesCost')
        self.num_spills: Optional[Union[int, float]] = details.get('NumSpills')
        self.total_spills_cost: Optional[Union[int, float]] = details.get('TotalSpillsCost')
        self.num_folded_reloads: Optional[Union[int, float]] = details.get('NumFoldedReloads')
        self.total_folded_reloads_cost: Optional[Union[int, float]] = details.get('TotalFoldedReloadsCost')
        self.num_remats: Optional[Union[int, float]] = details.get('NumRemats')
        self.total_remats_cost: Optional[Union[int, float]] = details.get('TotalRematsCost')


class LICMRemark(OptimizationRemark):
    """Remark specific to the 'licm' pass."""
    # Currently no specific fields beyond base + Info
    def __init__(self, remark_dict: Dict[str, Any]):
        super().__init__(remark_dict)


class SLPvectorizeRemark(OptimizationRemark):
    """Remark specific to the 'slp-vectorizer' pass."""
    # Currently only Cost/Threshold handled like Inline, plus Info
    def __init__(self, remark_dict: Dict[str, Any]):
        super().__init__(remark_dict)
        details = self.raw_details
        # Cost/Threshold might appear here too for 'NotBeneficial'
        cost_data = details.get('Cost')
        self.cost: Optional[Union[int, float]] = None
        if isinstance(cost_data, dict) and 'Value' in cost_data: # Less likely here
            self.cost = cost_data.get('Value')
        elif cost_data is not None:
            self.cost = cost_data

        thresh_data = details.get('Threshold')
        self.threshold: Optional[Union[int, float]] = None
        if isinstance(thresh_data, dict) and 'Value' in thresh_data: # Less likely here
            self.threshold = thresh_data.get('Value')
        elif thresh_data is not None:
            self.threshold = thresh_data


# --- Mapping from Pass Name to Class ---
# Case-insensitive matching could be added if needed
REMARK_CLASS_MAP: Dict[str, Type[OptimizationRemark]] = {
    'inline': InlineRemark,
    'gvn': GVNRemark,
    'regalloc': RegallocRemark,
    'licm': LICMRemark,
    'slp-vectorizer': SLPvectorizeRemark,
    # Add other passes here as needed
}

# --- Main Parsing Function ---

def parse_optimization_summary(yaml_path: Union[str, Path]) -> List[OptimizationRemark]:
    """
    Parses the optimization summary YAML file into a list of OptimizationRemark objects.

    Args:
        yaml_path: Path to the input YAML file generated by the previous script.

    Returns:
        A list of OptimizationRemark objects (or subclasses like InlineRemark, GVNRemark).
        Returns an empty list if the file cannot be read or is empty.
    """
    summary_path = Path(yaml_path)
    if not summary_path.is_file():
        print(f"Error: YAML file not found at {summary_path}", file=sys.stderr)
        return []

    all_remarks: List[OptimizationRemark] = []
    try:
        with open(summary_path, 'r') as f:
            # Skip the schema comment block at the beginning
            content = f.read()
            yaml_content_start = content.find('\n---') # Find separator after schema
            if yaml_content_start == -1:
                yaml_data_str = content # Assume no schema block if separator not found
            else:
                yaml_data_str = content[yaml_content_start + len('\n---'):]

            if not yaml_data_str.strip():
                 print(f"Warning: YAML file {summary_path} appears empty after schema.", file=sys.stderr)
                 return []

            data = yaml.safe_load(yaml_data_str)

        if not isinstance(data, dict):
            print(f"Error: YAML content in {summary_path} is not a dictionary.", file=sys.stderr)
            return []

        # Iterate through files and the remarks within them
        for file_path, remarks_in_file in data.items():
            if not isinstance(remarks_in_file, list):
                print(f"Warning: Expected a list of remarks for file '{file_path}', got {type(remarks_in_file)}. Skipping.", file=sys.stderr)
                continue

            for remark_dict in remarks_in_file:
                if not isinstance(remark_dict, dict):
                     print(f"Warning: Found non-dictionary item in remarks list for file '{file_path}'. Skipping item.", file=sys.stderr)
                     continue

                pass_name = remark_dict.get('Optimization_Pass')
                # Choose the appropriate class based on the pass name, fallback to base class
                RemarkClass = REMARK_CLASS_MAP.get(pass_name, OptimizationRemark) # Default to base class

                try:
                    remark_obj = RemarkClass(remark_dict)
                    all_remarks.append(remark_obj)
                except Exception as e:
                     print(f"Error instantiating remark object for pass '{pass_name}' in file '{file_path}': {e}", file=sys.stderr)
                     print(f"Problematic remark data: {remark_dict}", file=sys.stderr)


    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {summary_path}: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        # Already handled by the initial check, but keep for safety
        print(f"Error: File not found at {summary_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred processing {summary_path}: {e}", file=sys.stderr)
        return []

    return all_remarks

def index_remarks_by_function(remarks: List[OptimizationRemark]) -> Dict[str, List[OptimizationRemark]]:
    """
    Indexes a list of OptimizationRemark objects by function name.

    Args:
        remarks: A list of OptimizationRemark objects (output from parse_optimization_summary).

    Returns:
        A dictionary where keys are function names (strings) and values are lists
        of OptimizationRemark objects associated with that function. Function names
        defaulting to 'N/A' will be grouped under that key.
    """
    function_index = defaultdict(list)
    for remark in remarks:
        # Use the function attribute directly. It defaults to 'N/A' if missing in source.
        function_name = remark.function
        function_index[function_name].append(remark)
    return dict(function_index)

# --- Example Usage ---
if __name__ == "__main__":
    # Assume the summary YAML is named 'missed_optimizations_summary.yaml'
    # in the current directory
    if len(sys.argv) < 2:
        print("Usage: python3 clang_remark_analyzer.py <summary_yaml>")
        sys.exit(1)
    summary_file = sys.argv[

    print(f"Parsing {summary_file}...")
    parsed_remarks = parse_optimization_summary(summary_file)

    if not parsed_remarks:
        print("No remarks parsed or file was empty/not found.")
    else:
        print(f"\nSuccessfully parsed {len(parsed_remarks)} remarks.")

        # Example: Print details for different remark types
        inline_count = 0
        gvn_count = 0
        regalloc_count = 0
        other_count = 0

        index = index_remarks_by_function(remarks=parsed_remarks)
        print(index['genetic::build_program(genetic::program&, genetic::param const&, PhiloxEngine&)'])

        print("\n--- Example Remark Details ---")
        for remark in parsed_remarks[:15]: # Print details for first few
            print(remark.__str__())
            print(f"\nRemark Type: {type(remark).__name__}")
            print(f"  Pass: {remark.pass_name}, Reason: {remark.miss_reason}")
            print(f"  Location: {remark.location}")
            print(f"  Function: {remark.function}")

            if isinstance(remark, InlineRemark):
                inline_count += 1
                print(f"  Inline Cost: {remark.cost}, Threshold: {remark.threshold}")
                if remark.callee: print(f"  Callee: {remark.callee.name}")
            elif isinstance(remark, GVNRemark):
                gvn_count += 1
                if remark.clobbered_by: print(f"  Clobbered By: {remark.clobbered_by.type} at {remark.clobbered_by.location}")
                if remark.other_access: print(f"  Other Access: {remark.other_access.type} at {remark.other_access.location}")
            elif isinstance(remark, RegallocRemark):
                regalloc_count += 1
                if remark.num_spills: print(f"  Num Spills: {remark.num_spills} (Cost: {remark.total_spills_cost})")
                if remark.num_vr_copies: print(f"  Num VR Copies: {remark.num_vr_copies} (Cost: {remark.total_copies_cost})")
            else:
                other_count += 1
                if remark.info: print(f"  Info: {remark.info}")


        print(f"\n--- Summary ---")
        print(f"Total Remarks Parsed: {len(parsed_remarks)}")
        # Update counts for all remarks
        inline_count = sum(1 for r in parsed_remarks if isinstance(r, InlineRemark))
        gvn_count = sum(1 for r in parsed_remarks if isinstance(r, GVNRemark))
        regalloc_count = sum(1 for r in parsed_remarks if isinstance(r, RegallocRemark))
        licm_count = sum(1 for r in parsed_remarks if isinstance(r, LICMRemark))
        slp_count = sum(1 for r in parsed_remarks if isinstance(r, SLPvectorizeRemark))
        other_count = len(parsed_remarks) - (inline_count + gvn_count + regalloc_count + licm_count + slp_count)

        print(f"  Inline Remarks: {inline_count}")
        print(f"  GVN Remarks: {gvn_count}")
        print(f"  Regalloc Remarks: {regalloc_count}")
        print(f"  LICM Remarks: {licm_count}")
        print(f"  SLP Vectorizer Remarks: {slp_count}")
        print(f"  Other/Unknown Remarks: {other_count}")