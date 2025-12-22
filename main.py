import sys

if __name__ == '__main__':
    try:
        f = sys.argv[1]
    except IndexError:
        print("Usage: python3 main.py filename")


