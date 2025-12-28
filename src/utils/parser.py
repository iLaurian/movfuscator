import sys
import re
from . import macros

def parse_args():
    """
    Parses sys.argv to find input filename and optional output filename (-o).
    Returns a tuple (input_file, output_file).
    """
    args = sys.argv[1:]
    input_file = None
    output_file = "samples/out/output.s"

    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if arg == "-o":
            if i + 1 < len(args):
                output_file = args[i + 1]
                skip_next = True
            else:
                print("Error: -o flag requires a filename")
                sys.exit(1)
        else:
            if input_file is None:
                input_file = arg
            else:
                pass

    if input_file is None:
        print("Usage: python3 main.py <filename.s> [-o output.s]")
        sys.exit(1)

    return input_file, output_file


def process_data_section(lines):
    """
    Iterates through lines, detects variables in the .data section,
    and injects DUMMY and SELECT variables.
    Returns a list of processed lines.
    """
    processed_lines = []
    in_data_section = False

    var_pattern = re.compile(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*(\.[a-z]+)\s+(.*)')

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('.data'):
            in_data_section = True
            processed_lines.append(line)
            continue
        elif stripped.startswith('.text') or stripped.startswith('.section'):
            in_data_section = False
            processed_lines.append(line)
            continue

        if in_data_section:
            match = var_pattern.match(line)
            if match:
                label = match.group(1)
                directive = match.group(2)
                processed_lines.append(line)

                dummy_line = f"\tDUMMY_{label}: {directive} 0\n"
                processed_lines.append(dummy_line)

                select_line = f"\tSELECT_{label}: .long DUMMY_{label}, {label}\n"
                processed_lines.append(select_line)
                continue

        processed_lines.append(line)

    return processed_lines


def add_macros(lines):
    macro_generators = [
        macros.get_increment_def,
        macros.get_decrement_def,
        macros.get_logical_or_def,
        macros.get_logical_and_def,
        macros.get_logical_not_def
    ]

    all_data_content = []
    all_macro_content = []

    for gen in macro_generators:
        data_part, macro_part = gen()
        if data_part:
            all_data_content.append(data_part)
        if macro_part:
            all_macro_content.append(macro_part)

    final_lines = []

    if all_macro_content:
        final_lines.extend([m + "\n" for m in all_macro_content])

    data_injected = False

    for line in lines:
        final_lines.append(line)
        # Inject Data Tables immediately after .data directive
        if not data_injected and line.strip().startswith('.data'):
            final_lines.extend([d + "\n" for d in all_data_content])
            data_injected = True

    if not data_injected and all_data_content:
        final_lines.append("\n.data\n")
        final_lines.extend([d + "\n" for d in all_data_content])

    return final_lines