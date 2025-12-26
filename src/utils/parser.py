import sys
import re

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

