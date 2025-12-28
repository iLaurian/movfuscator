# 2025-12-28

```
ALTELE:
int
lea
mov
loop

JUMP-URI:
jmp
jump-uri conditionale (jge, jle, je, jz, jnz)
call

STACK:
push
pop

ALU:
add + sub
cmp
and + or + xor
test
inc + dec
div
mul
shr + shl

.data:
ZFLAG: .byte
DUMMY_ZFLAG: .byte
CFLAG: .byte
CFLAG_ZFLAG: .byte
```

# 2025-12-26

- inject dummy data in `.data` section @done
- inject macros in header @done
- instructions interpreter
  - should interpret opcodes we want to replace, leave everything else the same
  - must be able to handle all types of data accessing methods

# 2025-12-22

For the sake of simplicity we are going to use this `todo.md` file for planning and notes, rather than GitHub issues or projects.

- parse assembly code and interpret instructions
- implement lookup tables for arithmetic
- control flow via execution masking

Key Ideas:
- `mov` can check equality
  ```
  mov $0, x
  mov $1, y
  mov x, %eax
  ```
  - cannot write to arbitrary spots in memory so we limit to 1 byte data and create a 256 byte scratch array for equality testing
- implementing if
  - all paths execute, no matter what
  - add a pointer array to all variables
  ```
  .data
  X:          .long 0
  DUMMY_X:    .long 0
  Y:          .long 0
  DUMMY_Y:    .long 0
  SELECT_X:   .long DUMMY_X, X
  SELECT_Y:   .long DUMMY_Y, Y
  
  .text
  .global main
  main:
    movl    X, %eax
    movl    $0, (%eax)
    movl    Y, %eax
    movl    $4, (%eax)
    movl    X, %eax
  
    movl    SELECT_X(%eax), %eax
    movl    $100, (%eax) 
  ```
- extending the if/else idea
  - on each branch
    - if the branch is taken
      - store the target address
      - turn execution "off"
    - if the branch is not taken
      - leave execution "on"
  - on each operation
    - if execution is on
      - run the operation on real data
    - if execution is off
      - is current address the stored branch target? - yes
        - turn execution on
        - run operation on real data
- check how we can use macros for setup
- to stop the program just access null
  ```
  .data
  nh: .long 0
  h: .long nh, 0
  .text
    movl    b, %eax
    movl    h(%eax), %eax
    movl    (%eax), %eax
  ```
- arithmetic
  - lookup tables
    - increment
      ```
      .data
      incb:
        .set y, 1
        .rept 256
          .byte y & 0xff
          .set y, y+1
        .endr
      
      .text
      movl incb(%eax), %eax
      ```
    - decrement
      ```
      .data
      decb:
        .set y, 255
        .rept 256
          .byte y & 0xff
          .set y, y+1
        .endr
      
      .text
        movl    decb(%eax), %eax
      ```
    - logical or
      ```
      .data
      o:      .long o_0, o_1 
      o_0:    .long 0, 4
      o_1:    .long 4, 4

      .macro logical_or result, src1, src2
          movl    \src1, %eax
          movl    o(%eax), %edx
          movl    \src2, %eax
          movl    (%edx, %eax), %eax
          movl    %eax, \result
      .endm
      ```
    - logical and
      ```
      .data
      a: .long a_0, a_1
      a_0: .long 0, 0
      a_1: .long 0, 4
      
      .macro logic_and arg1, arg2, arg3
        movl \arg2, %eax
        movl a(%eax), %edx
        movl \arg3, %eax
        movl (%edx, %eax), %eax
        movl %eax, \arg1
      .endm
      ```
    - logical not
      ```
      .data
      n: .long 4, 0
      
      .macro logic_not dest, src
        movl    \src, %eax
        movl    n(%eax), %eax
        movl    %eax, \dest
      .endm
      ```
      