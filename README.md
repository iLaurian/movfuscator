The [**M/o/Vfuscator**](https://github.com/xoreaxeaxeax/movfuscator) compiles programs into `mov` instructions only, based on the idea, argued by [Stephen Dolan](docs/mov.pdf), that this instruction is *Turing Complete*.  

This project is a transpiler that converts Assembly x86 AT&T code into equivalent code using only the `mov` instruction.  

The goal of the projects is to manage to translate some sample programs, found in `samples/in`, into `mov` instructions only.

**Usage:**
```shell
# Transpiles programs from samples/in into samples/out
./transpile_all.sh

# To transpile a single program
python3 src/main.py path/to/input.s -o path/to/output.s 

# Tests if input code in samples/in is equivalent (generates same output & exit code) to code transpiled in samples/out
./test.sh
```

**Team:** Laurian Iacob, Neagu Ștefan-Claudel - group 152  

**References:**  
- https://www.youtube.com/watch?v=hsNDLVUzYEs
- [`mov` is Turing-complete](docs/mov.pdf)