def get_increment_def():
    data = """\tincb:
\t  .set y, 1
\t  .rept 256
\t    .byte y & 0xff
\t    .set y, y+1
\t  .endr
"""
    return data, ""

def get_decrement_def():
    data = """\tdecb:
\t  .set y, 255
\t  .rept 256
\t    .byte y & 0xff
\t    .set y, y+1
\t  .endr
"""
    return data, ""

# We use movf prefix for labels to avoid collisions with existing data
def get_logical_or_def():
    data = """\tmovf_or: .long movf_or0, movf_or1 
\tmovf_or0: .long 0, 4
\tmovf_or1: .long 4, 4
"""
    macro = """.macro logical_or result, src1, src2
  movl    \\src1, %eax
  movl    movf_or(%eax), %edx
  movl    \\src2, %eax
  movl    (%edx, %eax), %eax
  movl    %eax, \\result
.endm
"""
    return data, macro

def get_logical_and_def():
    data = """\tmovf_and: .long movf_and0, movf_and1
\tmovf_and0: .long 0, 0
\tmovf_and1: .long 0, 4
"""
    macro = """.macro logic_and arg1, arg2, arg3
  movl \\arg2, %eax
  movl movf_and(%eax), %edx
  movl \\arg3, %eax
  movl (%edx, %eax), %eax
  movl %eax, \\arg1
.endm
"""
    return data, macro

def get_logical_not_def():
    data = """\tmovf_not: .long 4, 0
"""
    macro = """.macro logic_not dest, src
  movl    \\src, %eax
  movl    movf_not(%eax), %eax
  movl    %eax, \\dest
.endm
"""
    return data, macro