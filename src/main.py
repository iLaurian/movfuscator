from utils.parser import *

if __name__ == '__main__':
    input_filename, output_filename = parse_args()

    try:
        with open(input_filename, 'r') as f:
            content = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        sys.exit(1)

    new_content = process_data_section(content)

    try:
        with open(output_filename, 'w') as f:
            f.writelines(new_content)
        print(f"Successfully processed '{input_filename}' -> '{output_filename}'")
    except IOError as e:
        print(f"Error writing to file '{output_filename}': {e}")


