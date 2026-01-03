from utils.parser import *
from utils.alu import *
import pprint

if __name__ == '__main__':
    input_filename, output_filename = parse_args()

    try:
        with open(input_filename, 'r') as f:
            content = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        sys.exit(1)

    assembly_code = parse_asm_source(content)
    # pprint.pp(assembly_code)

    final_assembly_lines = process_alu_parsed_lines(assembly_code)

    try:
        with open(output_filename, 'w') as f:
            for line in final_assembly_lines:
                f.write(line + '\n')

        print(f"Successfully processed '{input_filename}' -> '{output_filename}'")
        print(f"Output size: {len(final_assembly_lines)} lines.")

    except IOError as e:
        print(f"Error writing to file '{output_filename}': {e}")