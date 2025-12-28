The [**M/o/Vfuscator**](https://github.com/xoreaxeaxeax/movfuscator) compiles programs into `mov` instructions only, based on the idea, argued by [Stephen Dolan](docs/mov.pdf), that this instruction is *Turing Complete*.  

This project is an interpreter that converts Assembly x86 AT&T code into equivalent code using the `mov` instruction as much as possible.  
The goal of the projects is to manage to translate some sample programs, found in `samples/in`, into `mov` instructions only.

**Team:** Laurian Iacob, Neagu Ștefan-Claudel - group 152  

**References:**  
- https://www.youtube.com/watch?v=hsNDLVUzYEs
- [`mov` is Turing-complete](docs/mov.pdf)