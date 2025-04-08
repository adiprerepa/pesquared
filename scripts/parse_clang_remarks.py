#!/usr/bin/env python3

import argparse
import glob
import yaml
import os
import sys
import cxxfilt # For demangling C++ names
from pathlib import Path
from collections import defaultdict
import io # To write schema and data sequentially

# --- Helper Function to Demangle ---
def safe_demangle(name):
    """Safely demangles a C++ name, returning original if error."""
    if not name or not isinstance(name, str):
        return name
    try:
        return cxxfilt.demangle(name, external_only=False)
    except Exception:
        return name

# --- Helper to Extract Structured Args (Revised) ---
def parse_args(args_list):
    """
    Parses the Args list from Clang optimization remarks into a structured dictionary.
    Identifies specific patterns (like Callee, Caller, Cost, GVN context, Regalloc info)
    and places them as first-class citizens in the output dictionary.
    Handles common string arguments by adding them to an 'Info' field.
    """
    details = {'Info': []} # Collect general strings/unparsed items here
    if not isinstance(args_list, list):
        return details

    i = 0
    while i < len(args_list):
        item = args_list[i]
        processed = False # Flag to check if item was handled structurally

        if isinstance(item, dict):
            # --- Handle known top-level keys that expect a 'Name' and optional 'DebugLoc' ---
            keys_to_extract_with_name = ['Callee', 'Caller']
            for key in keys_to_extract_with_name:
                if key in item:
                    details[key] = {'Name': safe_demangle(item[key])}
                    # Check for DebugLoc immediately following
                    if i + 1 < len(args_list) and isinstance(args_list[i+1], dict) and 'DebugLoc' in args_list[i+1]:
                        details[key]['DebugLoc'] = args_list[i+1]['DebugLoc']
                        i += 1 # Consume DebugLoc
                    processed = True
                    break # Key handled

            # --- Handle known top-level keys that might have an optional 'DebugLoc' ---
            # Includes Cost, Threshold, Reason, the typo 'Treshold', and Regalloc keys
            if not processed:
                keys_to_extract_simple_value = [
                    'Cost', 'Threshold', 'Treshold', 'Reason', # Common Opt Args
                    'NumVRCopies', 'TotalCopiesCost',          # Regalloc Args
                    'NumSpills', 'TotalSpillsCost',
                    'NumFoldedReloads', 'TotalFoldedReloadsCost',
                    'NumRemats', 'TotalRematsCost'             # More potential Regalloc Args
                    # Add other simple key-value args here if needed
                ]
                for key in keys_to_extract_simple_value:
                    if key in item:
                        value = item[key]
                        debug_loc_associated = None
                        # Check for DebugLoc immediately following
                        if i + 1 < len(args_list) and isinstance(args_list[i+1], dict) and 'DebugLoc' in args_list[i+1]:
                             debug_loc_associated = args_list[i+1]['DebugLoc']
                             i += 1 # Consume DebugLoc

                        # Attempt to convert numeric strings to numbers
                        try:
                            if isinstance(value, str):
                                if '.' in value or 'e' in value or 'E' in value:
                                    numeric_value = float(value)
                                else:
                                    numeric_value = int(value)
                            else: # Already a number?
                                numeric_value = value
                        except (ValueError, TypeError):
                            numeric_value = value # Keep original if conversion fails

                        # Store value, potentially with DebugLoc
                        if debug_loc_associated:
                            details[key] = {'Value': numeric_value, 'DebugLoc': debug_loc_associated}
                        else:
                            details[key] = numeric_value
                        processed = True
                        break # Key handled

            # --- Handle specific context keys (like GVN) that expect a 'Type' and optional 'DebugLoc' ---
            if not processed:
                context_keys_with_debugloc = ['OtherAccess', 'ClobberedBy'] # Add more as needed
                for key in context_keys_with_debugloc:
                    if key in item:
                        context_info = {'Type': item[key]} # Store the value under 'Type'
                        # Check for DebugLoc immediately following
                        if i + 1 < len(args_list) and isinstance(args_list[i+1], dict) and 'DebugLoc' in args_list[i+1]:
                            context_info['DebugLoc'] = args_list[i+1]['DebugLoc']
                            i += 1 # Consume DebugLoc
                        details[key] = context_info # Add structured info
                        processed = True
                        break # Key handled

            # --- Handle simple key-value pairs if not processed otherwise ---
            # Catches {'Type': 'i32'}, {'Line': '40'}, and importantly {'String': '...'}
            if not processed and len(item) == 1:
                key, val = list(item.items())[0]
                # Treat 'String' dicts as items to add to the Info list
                if key == 'String' and isinstance(val, str):
                     details['Info'].append(val)
                     processed = True
                # Avoid overwriting previously parsed structured data for the same key
                elif key not in details:
                   details[key] = val
                   processed = True

        # --- Add unhandled items/strings to Info ---
        # Catches standalone strings or complex dicts we didn't specifically parse
        if not processed:
            # Avoid adding DebugLoc dicts that were already consumed by previous logic
            is_consumed_debugloc = False
            if isinstance(item, dict) and 'DebugLoc' in item:
                 if i > 0 and isinstance(args_list[i-1], dict):
                     # Check if the *previous* item was one that could have consumed this DebugLoc
                     prev_item_keys = args_list[i-1].keys()
                     possible_consumers = keys_to_extract_with_name + keys_to_extract_simple_value + context_keys_with_debugloc
                     if any(k in prev_item_keys for k in possible_consumers):
                         is_consumed_debugloc = True

            if not is_consumed_debugloc:
                 details['Info'].append(str(item)) # Fallback: stringify

        i += 1 # Move to the next item in Args

    # --- Cleanup Info list ---
    cleaned_info_items = []
    for info_item in details.get('Info', []):
        # Basic cleaning
        info_item = info_item.strip()
        info_item = info_item.replace("''", "") # Remove empty quotes often used as placeholders
        # Optionally remove specific boilerplate text (uncomment lines below to activate)
        # info_item = info_item.replace(" because it is clobbered by ", "")
        # info_item = info_item.replace(" because its definition is unavailable", "")
        # info_item = info_item.replace(" generated in function", "")
        # info_item = info_item.replace("failed to move load with loop-invariant address because the loop may invalidate its value", "(LICM: invalidated)")
        # info_item = info_item.replace("failed to hoist load with loop-invariant address because load is conditionally executed", "(LICM: conditional)")
        # info_item = info_item.replace(" with available vectorization factors", "(SLP: no factors)")
        # info_item = info_item.replace(" >= ", "") # Remove SLP comparison string if Cost/Treshold captured

        # Only add non-empty strings, and ignore strings that are just punctuation like ')'
        if info_item and info_item not in [')', ';', ':', '"', "'"]:
            cleaned_info_items.append(info_item)

    joined_info = ' '.join(cleaned_info_items).strip()

    if joined_info:
        details['Info'] = joined_info
    else:
        details.pop('Info', None) # Remove empty info key if nothing substantial remains

    # Rename 'Treshold' to 'Threshold' if found, for consistency
    if 'Treshold' in details:
        if 'Threshold' not in details: # Avoid overwriting if 'Threshold' also exists
            details['Threshold'] = details.pop('Treshold')
        else: # Both exist? Keep the official one, remove typo.
            details.pop('Treshold')

    return details

# --- Define Output YAML Schema (Updated) ---
YAML_SCHEMA = """# YAML Schema for Clang Missed Optimization Summary
#
# This file aggregates missed optimization remarks from Clang's -fsave-optimization-record output.
# Data is grouped by the source file where the missed optimization occurred.
#
# Root: A dictionary where keys are source file paths (strings).
#
# File Path Key (e.g., "src/philox_rng.cpp"):
#   Value: A list of missed optimization records found in that file.
#
# Missed Optimization Record (List Item): A dictionary with the following keys:
#
#   Optimization_Pass: (string) The name of the optimization pass (e.g., "gvn", "inline", "licm", "slp-vectorizer", "regalloc").
#   Miss_Reason: (string) The specific reason code or name for the missed optimization (e.g., "LoadClobbered", "TooCostly", "NoDefinition", "NotPossible", "SpillReloadCopies").
#   Source_Location: (dictionary) Location of the missed optimization in the source code.
#     File: (string) Path to the source file.
#     Line: (integer) Line number.
#     Column: (integer) Column number.
#   Function: (string) Demangled name of the function containing the missed optimization.
#   Details: (dictionary) Additional details extracted from the 'Args' field of the original remark. May contain:
#     # --- Inlining Args ---
#     Caller: (dictionary, optional) Info about the caller function.
#       Name: (string) Demangled name of the caller.
#       DebugLoc: (dictionary, optional) Source location of the call site in the caller.
#     Callee: (dictionary, optional) Info about the callee function.
#       Name: (string) Demangled name of the callee.
#       DebugLoc: (dictionary, optional) Source location of the callee definition.
#     Cost: (number or dictionary, optional) Estimated cost. Parsed as number (int/float) if possible. Might be {'Value': N, 'DebugLoc': ...}.
#     Threshold: (number or dictionary, optional) Threshold used. Parsed as number if possible. Might be {'Value': N, 'DebugLoc': ...}. (Note: 'Treshold' typo is corrected to 'Threshold').
#     Reason: (string or dictionary, optional) Textual reason provided. Might be {'Value': "...", 'DebugLoc': ...}.
#     # --- GVN Args ---
#     OtherAccess: (dictionary, optional) GVN: Details about another memory access.
#       Type: (string) Type of access (e.g., "load", "store").
#       DebugLoc: (dictionary, optional) Source location of the other access.
#     ClobberedBy: (dictionary, optional) GVN: Details about the instruction that clobbers the value.
#       Type: (string) Type of clobbering instruction (e.g., "invoke", "store", "call").
#       DebugLoc: (dictionary, optional) Source location of the clobbering instruction.
#     # --- Regalloc Args ---
#     NumVRCopies: (number, optional) Regalloc: Number of copies of virtual registers.
#     TotalCopiesCost: (number, optional) Regalloc: Estimated cost of copies.
#     NumSpills: (number, optional) Regalloc: Number of spills.
#     TotalSpillsCost: (number, optional) Regalloc: Estimated cost of spills.
#     NumFoldedReloads: (number, optional) Regalloc: Number of reloads folded into instructions.
#     TotalFoldedReloadsCost: (number, optional) Regalloc: Estimated cost of folded reloads.
#     NumRemats: (number, optional) Regalloc: Number of rematerializations.
#     TotalRematsCost: (number, optional) Regalloc: Estimated cost of rematerializations.
#     # --- Generic/Fallback Args ---
#     Type: (string, optional) Data type involved (e.g., "i32", "float"). Extracted from simple {'Type': '...'}.
#     Line: (string or integer, optional) Line number info sometimes present in Args. Extracted from simple {'Line': '...'}.
#     Info: (string, optional) Concatenated string of any remaining arguments or descriptive text originally passed via {'String': '...'}, after basic cleaning. This captures general messages from passes like LICM, SLP, etc.
#   Original_Tag: (string) The original YAML tag from the Clang output (e.g., "Missed").
#   Source_Yaml: (string) The name of the specific *.opt.yaml file this record came from.
#
---
"""


# --- Custom YAML Constructor ---
# (Keep the unknown_tag_constructor function as it was)
def unknown_tag_constructor(loader, tag_suffix, node):
    """Handles unknown tags like !Passed, !Missed. Adds tag as metadata."""
    data = None
    if isinstance(node, yaml.MappingNode):
        data = loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.ScalarNode):
        data = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        data = loader.construct_sequence(node, deep=True)
    else:
        print(f"Warning: Encountered unknown node type {type(node)} for tag {tag_suffix}. Treating as scalar.", file=sys.stderr)
        data = loader.construct_scalar(node) # Fallback

    if isinstance(data, dict):
        data['_YamlTag_'] = tag_suffix # Store the original tag (e.g., '!Missed')
    else:
        # If data isn't a dict, wrap it to store the tag
        print(f"Warning: Data following tag {tag_suffix} was not a dictionary: {data}. Wrapping it.", file=sys.stderr)
        return {'_YamlTag_': tag_suffix, '_RawValue_': data}

    return data

# --- Main Processing Logic ---
# (Keep the process_yaml_files function as it was, ensuring it uses the updated parse_args)
def process_yaml_files(input_dir, output_file):
    """Finds, parses, filters, and summarizes missed optimizations."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)

    yaml_files = list(input_path.glob('*.opt.yaml'))
    if not yaml_files:
        print(f"Warning: No *.opt.yaml files found in '{input_dir}'.", file=sys.stderr)
        return

    print(f"Found {len(yaml_files)} YAML files to process...")

    processed_optimizations = defaultdict(list) # Group by source file path

    TheLoader = yaml.SafeLoader
    TheLoader.add_multi_constructor('!', unknown_tag_constructor)

    total_docs = 0
    processed_count = 0

    for yml_file in yaml_files:
        print(f"Processing {yml_file.name}...")
        file_processed_count = 0
        try:
            with open(yml_file, 'r') as f:
                docs = list(yaml.load_all(f, Loader=TheLoader))
                total_docs += len(docs)

                for doc_index, doc in enumerate(docs):
                    if not isinstance(doc, dict):
                        print(f"  Warning: Skipping non-dictionary document #{doc_index+1} in {yml_file.name}.", file=sys.stderr)
                        continue

                    original_tag = doc.get('_YamlTag_')

                    if original_tag == 'Missed': # Focus on missed optimizations
                        try:
                            pass_name = doc.get('Pass', 'N/A')
                            miss_reason = doc.get('Name', 'N/A')
                            function_name = safe_demangle(doc.get('Function', 'N/A'))
                            args_list = doc.get('Args', [])

                            debug_loc = doc.get('DebugLoc', {})
                            if not isinstance(debug_loc, dict):
                                print(f"  Warning: Skipping record with invalid DebugLoc in {yml_file.name} (Doc #{doc_index+1}): {doc}", file=sys.stderr)
                                continue

                            file = debug_loc.get('File', 'Unknown File')
                            line = debug_loc.get('Line', 0)
                            col = debug_loc.get('Column', 0)

                            # --- <<<< YOUR FILTER LOGIC HERE >>>> ---
                            if file.startswith('src/'): # Example filter
                            # --- End Filter Logic ---

                                processed_count += 1
                                file_processed_count += 1

                                details = parse_args(args_list) # Use the revised parser

                                processed_record = {
                                    'Optimization_Pass': pass_name,
                                    'Miss_Reason': miss_reason,
                                    'Source_Location': {
                                        'File': file,
                                        'Line': line,
                                        'Column': col
                                    },
                                    'Function': function_name,
                                    'Details': details,
                                    'Original_Tag': original_tag,
                                    'Source_Yaml': str(yml_file.name)
                                }
                                processed_optimizations[file].append(processed_record)

                        except Exception as e:
                            print(f"  Warning: Error processing record in {yml_file.name} (Doc #{doc_index+1}): {e}", file=sys.stderr)
                            print(f"  Problematic Record Data: {doc}", file=sys.stderr)

            if file_processed_count > 0:
                print(f"  Found {file_processed_count} relevant 'Missed' records in this file.")

        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yml_file.name}: {e}", file=sys.stderr)
        except FileNotFoundError:
             print(f"Error: File {yml_file.name} not found during processing.", file=sys.stderr)
        except Exception as e:
             print(f"An unexpected error occurred processing {yml_file.name}: {e}", file=sys.stderr)

    print(f"\nProcessed {total_docs} total optimization records.")
    print(f"Found {processed_count} 'Missed' optimization records matching the filter criteria.")

    if not processed_optimizations:
        print("No relevant missed optimizations found to write.")
        return

    # --- Write the consolidated output ---
    try:
        output_path = Path(output_file)
        print(f"Writing summary to {output_path}...")

        string_stream = io.StringIO()
        output_data = dict(processed_optimizations)
        yaml.dump(output_data, string_stream, default_flow_style=False, sort_keys=False, indent=2)

        with open(output_path, 'w') as out_f:
            out_f.write(YAML_SCHEMA) # Write schema first
            out_f.write(string_stream.getvalue()) # Then write data

        print("Done.")
    except Exception as e:
         print(f"Error writing output file {output_file}: {e}", file=sys.stderr)

# --- Command Line Argument Parsing ---
# (Keep the __main__ block as it was)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize Clang Missed Optimization YAML records from *.opt.yaml files.')
    parser.add_argument('input_dir',
                        help='Directory containing the *.opt.yaml files.')
    parser.add_argument('-o', '--output',
                        default='missed_optimizations_summary.yaml',
                        help='Output YAML file name (default: missed_optimizations_summary.yaml)')

    args = parser.parse_args()

    process_yaml_files(args.input_dir, args.output)