def generate_alu_tables():
    """
    Generates the .data section for the MOV-based ALU.
    Includes lookup tables for arithmetic, logic, and stack operations.
    Excludes flow control (branching/execution gating) structures.
    """
    lines = []

    # --- Constants ---
    # Adjust these sizes based on your memory constraints
    SOFT_I_REGS = 4
    SOFT_F_REGS = 4
    SOFT_D_REGS = 4
    STACK_SIZE = 0x4000  # 16KB stack
    DISCARD_SIZE = 0x100

    def emit(s):
        lines.append(s)

    def emit_label(l):
        emit(f"{l}:")

    def build_1d_table(name, dtype, elements, expr_lambda):
        emit(f".align 16")
        emit(f".globl {name}")
        # Generate values comma-separated
        vals = [hex(expr_lambda(x) & (0xFF if dtype == 'byte' else 0xFFFFFFFF)) for x in range(elements)]
        emit(f"{name}: .{dtype} " + ", ".join(vals))
        emit(f".equ {name}_END, .")
        emit("")

    def build_2d_table(name, dtype, height, width, expr_lambda):
        # Table of pointers
        emit(f".align 16")
        emit(f".globl {name}")
        emit(f"{name}: .long " + ", ".join(f"{name}_{x}" for x in range(height)))

        # Table of rows
        for x in range(height):
            emit(f".globl {name}_{x}")
            vals = [hex(expr_lambda(x, y) & (0xFF if dtype == 'byte' else 0xFFFFFFFF)) for y in range(width)]
            emit(f"{name}_{x}: .{dtype} " + ", ".join(vals))
        emit("")

    def build_1d_index_table(name, dtype, height, width, expr_lambda):
        # Used for arithmetic offset optimization
        emit(f".align 16")
        emit(f".globl {name}")
        emit(f"{name}: .long " + ", ".join(f"{name}_{x}" for x in range(height)))

        # Flattened value strip
        for y in range(width):
            emit(f"{name}_{y}: .{dtype} {hex(expr_lambda(y))}")
        emit("")

    # DATA SECTION
    emit(".data")

    # Soft Registers
    emit(".align 16")
    for i in range(SOFT_I_REGS): emit(f".globl R{i}\nR{i}: .long 0")
    for i in range(SOFT_F_REGS): emit(f".globl F{i}\nF{i}: .long 0")
    for i in range(SOFT_D_REGS): emit(f".globl D{i}\nD{i}: .long 0, 0")

    # Logical Tables (1-bit)
    emit(".align 16")
    emit(".globl or, and, xor, xnor")
    emit("or:    .long or_0, or_1")
    emit("or_0:  .long 0, 1")
    emit("or_1:  .long 1, 1")

    emit("and:   .long and_0, and_1")
    emit("and_0: .long 0, 0")
    emit("and_1: .long 0, 1")

    emit("xor:   .long xor_0, xor_1")
    emit("xor_0: .long 0, 1")
    emit("xor_1: .long 1, 0")

    emit("xnor:  .long xnor_0, xnor_1")
    emit("xnor_0: .long 1, 0")
    emit("xnor_1: .long 0, 1")

    # Boolean & Comparison Helpers
    build_1d_table("alu_true", "byte", 512, lambda x: 1 if x else 0)
    build_1d_table("alu_false", "byte", 512, lambda x: 0 if x else 1)

    # Bit extraction (0-7)
    for b in range(8):
        build_1d_table(f"alu_b{b}", "long", 256, lambda x, b=b: (x >> b) & 1)

    # Bit Set/Clear (2D)
    build_2d_table("alu_b_s", "byte", 8, 256, lambda y, x: x | (1 << y))
    build_2d_table("alu_b_c", "byte", 8, 256, lambda y, x: x & ~(1 << y))

    # Equality check (x == y)
    build_2d_table("alu_eq", "byte", 256, 256, lambda x, y: 1 if x == y else 0)

    # Arithmetic Tables

    build_1d_table("alu_mask_FF", "byte", 256, lambda x: 0xFF if x == 1 else 0)

    # Increment/Decrement (Byte)
    emit("incb:")
    emit("\t.set y, 1")
    emit("\t.rept 256")
    emit("\t.byte y & 0xff")
    emit("\t.set y, y+1")
    emit("\t.endr")

    emit("decb:")
    emit("\t.set y, 255")
    emit("\t.rept 256")
    emit("\t.byte y & 0xff")
    emit("\t.set y, y+1")
    emit("\t.endr")

    # ADD
    build_1d_table("alu_add8l", "byte", 512, lambda x: x & 0xff)
    build_1d_table("alu_add8h", "byte", 512, lambda x: (x >> 8) & 0xff)

    # ADD 16 (Offset table optimization)
    build_1d_index_table("alu_add16", "long", 131072, 131072, lambda y: y)

    # INV (Bitwise NOT for subtraction)
    build_1d_table("alu_inv8", "byte", 256, lambda x: ~x)

    # Logical 8-bit (2D)
    build_2d_table("alu_band8", "byte", 256, 256, lambda x, y: x & y)
    build_2d_table("alu_bor8", "byte", 256, 256, lambda x, y: x | y)
    build_2d_table("alu_bxor8", "byte", 256, 256, lambda x, y: x ^ y)

    # SHIFT TABLES

    # Left Shift (Unsigned)
    build_2d_table("alu_lshu8", "long", 33, 256,
                   lambda x, y: 0 if x > 31 else (y << x))

    # Right Shift (Unsigned)
    build_2d_table("alu_rshu8", "long", 33, 256,
                   lambda x, y: 0 if x > 31 else ((y << 24) & 0xFFFFFFFF) >> x)

    # Right Shift (Signed)
    def rshi_func(x, y):
        if x > 31: return 0xFFFFFFFF
        val = (y & 0x80) << 24
        if val:
            mask = (0xFFFFFFFF << (32 - x)) & 0xFFFFFFFF
            return ((val >> x) | mask) & 0xFFFFFFFF
        return 0

    build_2d_table("alu_rshi8s", "long", 33, 256, rshi_func)

    # Clamp shift amount to 32
    build_1d_table("alu_clamp32", "long", 512, lambda x: 32 if x > 32 else x)

    # Multiplication & Division Helpers
    build_1d_table("alu_mul_sum8l", "byte", 256 * 3, lambda x: x & 0xff)
    build_1d_table("alu_mul_sum8h", "byte", 256 * 3, lambda x: (x >> 8) & 0xff)
    build_1d_table("alu_mul_shl2", "long", 256 * 16, lambda x: x * 4)
    build_1d_table("alu_mul_sums", "long", 256 * 16, lambda x: x)

    build_2d_table("alu_mul_mul8l", "byte", 256, 256, lambda x, y: (x * y) & 0xff)
    build_2d_table("alu_mul_mul8h", "byte", 256, 256, lambda x, y: (x * y) >> 8)

    # Division shifts
    build_1d_table("alu_div_shl1_8_c_d", "long", 512, lambda x: x)
    build_1d_table("alu_div_shl1_8_d", "long", 256, lambda x: x * 2)
    build_1d_table("alu_div_shl2_8_d", "long", 256, lambda x: x * 4)
    build_1d_table("alu_div_shl3_8_d", "long", 256, lambda x: x * 8)

    # Sign Extension (Byte to Long)
    build_1d_table("alu_sex8", "long", 256, lambda x: (x - 256) if x > 127 else x)

    # Flags & Scratch

    # Overflow Flag Lookup Tree for Addition (Sign A, Sign B, Sign Result -> OF)
    emit(".align 16")
    emit(".globl alu_cmp_of")
    emit("alu_cmp_of:     .long alu_cmp_of_0,   alu_cmp_of_1")
    emit("alu_cmp_of_0:   .long alu_cmp_of_00,  alu_cmp_of_01")
    emit("alu_cmp_of_1:   .long alu_cmp_of_10,  alu_cmp_of_11")

    # Level 2
    emit("alu_cmp_of_00:  .long alu_cmp_of_000, alu_cmp_of_001")
    emit("alu_cmp_of_01:  .long alu_cmp_of_010, alu_cmp_of_011")
    emit("alu_cmp_of_10:  .long alu_cmp_of_100, alu_cmp_of_101")
    emit("alu_cmp_of_11:  .long alu_cmp_of_110, alu_cmp_of_111")

    # Leaves (1 = Overflow, 0 = No Overflow)
    # Mapping: [SignA][SignB][SignDiff]
    emit("alu_cmp_of_000: .long 0")
    emit("alu_cmp_of_001: .long 0")
    emit("alu_cmp_of_010: .long 0")
    emit("alu_cmp_of_011: .long 1")
    emit("alu_cmp_of_100: .long 1")
    emit("alu_cmp_of_101: .long 0")
    emit("alu_cmp_of_110: .long 0")
    emit("alu_cmp_of_111: .long 0")

    # Boolean scratch
    emit(".align 16")
    emit(".globl b0, b1, b2, b3")
    for i in range(4): emit(f"b{i}: .long 0")

    # ALU Scratch
    emit(".align 16")
    emit(".globl alu_x, alu_y, alu_s, alu_c")
    emit("alu_x: .long 0")
    emit("alu_y: .long 0")
    emit("alu_s: .long 0")
    emit(".long 0")  # Padding
    emit("alu_c: .long 0, 0")

    # Deep Scratch (Shift/Mul/Div)
    emit(".align 16")

    # Padding for shift operations
    # We write to (alu_s0 - 3), so we need valid memory before alu_s0.
    emit("alu_pad_pre: .fill 16, 1, 0")

    scratch_vars = [
        "alu_s0", "alu_s1", "alu_s2", "alu_s3", "alu_ss", "alu_sc", "alu_sx",
        "alu_z0", "alu_z1", "alu_z2", "alu_z3",
        "alu_n", "alu_d", "alu_q", "alu_r", "alu_t",
        "alu_ns", "alu_ds", "alu_qs", "alu_rs",
        "alu_sn", "alu_sd", "alu_sq", "alu_sr"
    ]
    for sv in scratch_vars:
        emit(f".globl {sv}")
        emit(f"{sv}: .long 0")

    # Pointers for Div/Rem Selectors
    emit(".globl alu_sel_r, alu_sel_d, alu_sel_q, alu_sel_n")
    emit("alu_sel_r: .long alu_sr, alu_r")
    emit("alu_sel_d: .long alu_sd, alu_d")
    emit("alu_sel_q: .long alu_sq, alu_q")
    emit("alu_sel_n: .long alu_sn, alu_n")

    emit(".globl alu_psel_r, alu_psel_d, alu_psel_q, alu_psel_n")
    for v in ["r", "d", "q", "n"]: emit(f"alu_psel_{v}: .long 0")

    # CPU Flags
    emit(".align 16")
    emit(".globl zf, sf, of, cf")
    emit("zf: .byte 0")
    emit("sf: .byte 0")
    emit("of: .byte 0")
    emit("cf: .byte 0")

    # Stack
    emit(".align 16")
    emit(".globl stack_temp")
    emit("stack_temp: .long 0, 0")

    emit(".align 16")
    emit(".globl stack")
    emit("stack:")
    emit(f".fill 0x{STACK_SIZE:x}, 1, 0")

    emit(f"pop_guard: .long stack+0x0")
    emit("pushpop: .long " + ", ".join(f"stack+0x{i:x}" for i in range(0, STACK_SIZE + 4, 4)))
    emit(".globl push, pop")
    emit(".equ push, pushpop-stack-4")
    emit(".equ pop,  pushpop-stack+4")
    emit(f"push_guard: .long stack+0x{STACK_SIZE:x}")

    emit(".align 16")
    emit(".globl sp, fp")
    emit(f"sp: .long stack+0x{STACK_SIZE:x}")
    emit(f"fp: .long stack+0x{STACK_SIZE:x}")

    # Discard in BSS
    emit(".section .bss")
    emit(".align 16")
    emit(".globl discard")
    emit(f"discard: .fill 0x{DISCARD_SIZE:x}, 1, 0")

    # Register Backup Storage
    emit(".align 16")
    emit(".globl backup_eax, backup_ebx, backup_ecx, backup_edx, backup_esi, backup_edi")
    for r in ["eax", "ebx", "ecx", "edx", "esi", "edi"]:
        emit(f"backup_{r}: .long 0")

    return lines


def translate_alu_instruction(opcode, operands):
    """
    Translates standard ALU opcodes into MOV sequences.
    """
    lines = []

    def emit(s):
        lines.append(f"\t{s}")

    def mov(src, dst):
        emit(f"movl {src}, {dst}")

    def movb(src, dst):
        emit(f"movb {src}, {dst}")

    def movw(src, dst):
        emit(f"movw {src}, {dst}")

    def movzx_byte_to_reg(src, dst):
        emit(f"movzbl {src}, {dst}")

    def get_op_str(op):
        if op[0] == 'imm': return 'imm', f"${op[1]}"
        if op[0] == 'reg': return 'reg', op[1]
        if op[0] == 'mem': return 'mem', op[1]
        if op[0] == 'label': return 'mem', op[1]
        return 'imm', f"${op[1]}"

    def save_regs():
        emit("# -- context save --")
        # We save EAX, EBX, ECX, EDX, ESI, EDI
        # We generally assume ESP/EBP are handled safely or untouched by ALU internals
        regs = ['%eax', '%ebx', '%ecx', '%edx', '%esi', '%edi']
        for r in regs:
            # backup_eax, backup_ebx...
            mov(r, f"backup_{r[1:]}")

    def restore_regs(skip_reg=None):
        if skip_reg is None: skip_reg = []
        emit("# -- context restore --")
        regs = ['%eax', '%ebx', '%ecx', '%edx', '%esi', '%edi']
        for r in regs:
            if r in skip_reg: continue
            mov(f"backup_{r[1:]}", r)

    # Load operand into memory scratchpad (alu_x or alu_y)
    def load_to_scratch(op_tuple, scratch_loc):
        otype, oval = get_op_str(op_tuple)
        if otype == 'mem':
            # mem -> scratch requires intermediate reg
            mov(oval, "%edx")
            mov("%edx", scratch_loc)
        else:
            mov(oval, scratch_loc)
        return scratch_loc

    # Write Back from memory scratchpad (alu_s) to destination
    def write_back(src_loc, dest_op):
        dtype, dval = get_op_str(dest_op)
        if dtype == 'reg':
            mov(src_loc, dval)
        else:
            mov(src_loc, "%edx")
            mov("%edx", dval)

    # Flag Helpers

    def emit_update_zf_sf(res_loc):
        emit(f"# -- update ZF SF ({res_loc}) --")

        # SF
        mov("$0", "%eax")
        movb(f"{res_loc}+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        movb("%al", "sf")

        # ZF
        mov("$0", "%ebx")
        mov("$0", "%eax")

        movb(f"{res_loc}+0", "%al")
        movb("alu_true(%eax)", "%cl")
        emit("or %cl, %bl")

        movb(f"{res_loc}+1", "%al")
        movb("alu_true(%eax)", "%cl")
        emit("or %cl, %bl")

        movb(f"{res_loc}+2", "%al")
        movb("alu_true(%eax)", "%cl")
        emit("or %cl, %bl")

        movb(f"{res_loc}+3", "%al")
        movb("alu_true(%eax)", "%cl")
        emit("or %cl, %bl")

        mov("$0", "%eax")
        movb("%bl", "%al")
        movb("alu_false(%eax)", "%al")
        movb("%al", "zf")

    def emit_update_of(op1, op2, res):
        emit("# -- update OF --")
        mov("$alu_cmp_of", "%edx")

        mov("$0", "%eax")
        movb(f"{op1}+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        mov("(%edx,%eax,4)", "%edx")

        mov("$0", "%eax")
        movb(f"{op2}+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        mov("(%edx,%eax,4)", "%edx")

        mov("$0", "%eax")
        movb(f"{res}+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        mov("(%edx,%eax,4)", "%edx")

        mov("(%edx)", "%eax")
        movb("%al", "of")

    # ALU Implementations

    def impl_push(src_op):
        emit(f"# push")
        # Load value to temp
        t, v = get_op_str(src_op)
        if t == 'mem':
            mov(v, "%eax")
            mov("%eax", "stack_temp")
        else:
            mov(v, "stack_temp")

        # Decrement SP: sp = push(sp)
        mov("sp", "%eax")
        mov("push(%eax)", "%eax")
        mov("%eax", "sp")

        # sync hardware pointer
        mov("%eax", "%esp")

        # Store data
        mov("stack_temp", "%edx")
        mov("%edx", "(%eax)")

    def impl_pop(dest_op):
        emit(f"# pop")
        # Load from stack
        mov("sp", "%eax")
        mov("(%eax)", "%edx")
        mov("%edx", "stack_temp")

        # Increment SP: sp = pop(sp)
        mov("pop(%eax)", "%eax")
        mov("%eax", "sp")

        # sync hardware pointer
        mov("%eax", "%esp")

        # Write to dest
        dtype, dval = get_op_str(dest_op)
        if dtype == 'reg':
            mov("stack_temp", dval)
        elif dtype == 'mem':
            mov("stack_temp", "%edx")
            mov("%edx", dval)

    def impl_inc_helper(dest_op):
        # INC = ADD dest, 1 (But CF is preserved)
        emit("# -- alu_inc --")

        # 1. Backup CF (INC should not change it)
        # We use b0 as temp storage for CF
        mov("$0", "%eax")
        movb("cf", "%al")
        movb("%al", "b0")

        # 2. Perform ADD
        load_to_scratch(dest_op, "alu_y")  # Dest
        mov("$1", "alu_x")  # Source = 1

        # alu_s = alu_y + alu_x
        impl_alu_add32("alu_s", "alu_y", "alu_x")
        write_back("alu_s", dest_op)

        # 3. Restore CF
        mov("$0", "%eax")
        movb("b0", "%al")
        movb("%al", "cf")

    def impl_dec_helper(dest_op):
        # DEC = SUB dest, 1 (But CF is preserved)
        emit("# -- alu_dec --")

        # 1. Backup CF
        mov("$0", "%eax")
        movb("cf", "%al")
        movb("%al", "b0")

        # 2. Perform SUB
        load_to_scratch(dest_op, "alu_x")  # Dest
        mov("$1", "alu_y")  # Source = 1

        # alu_s = alu_x - alu_y
        impl_alu_sub32("alu_s", "alu_x", "alu_y")
        write_back("alu_s", dest_op)

        # 3. Restore CF
        mov("$0", "%eax")
        movb("b0", "%al")
        movb("%al", "cf")

    def prepare_shift(count_op, dest_op):
        load_to_scratch(count_op, "alu_x")
        load_to_scratch(dest_op, "alu_y")

        mov("alu_x", "%eax")
        mov("alu_clamp32(,%eax,4)", "%eax")
        mov("%eax", "alu_x")

        mov("$0", "alu_s0")
        mov("$0", "alu_s1")
        mov("$0", "alu_s2")
        mov("$0", "alu_s3")
        mov("$0", "alu_ss")

    def combine_scratch_to_s():
        mov("alu_s0", "%eax")
        emit("or alu_s1, %eax")
        emit("or alu_s2, %eax")
        emit("or alu_s3, %eax")
        emit("or alu_ss, %eax")
        mov("%eax", "alu_s")

    def impl_alu_shl(count_op, dest_op):
        emit("# -- alu_shl --")
        prepare_shift(count_op, dest_op)

        mov("alu_x", "%eax")
        mov("alu_lshu8(,%eax,4)", "%edx")

        mov("$0", "%eax")
        movb("alu_y+0", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s0")

        mov("$0", "%eax")
        movb("alu_y+1", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s1")

        mov("$0", "%eax")
        movb("alu_y+2", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s2")

        mov("$0", "%eax")
        movb("alu_y+3", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s3")

        combine_scratch_to_s()
        write_back("alu_s", dest_op)
        emit_update_zf_sf("alu_s")

    def impl_alu_shr(count_op, dest_op):
        emit("# -- alu_shr --")
        prepare_shift(count_op, dest_op)

        mov("alu_x", "%eax")
        mov("alu_rshu8(,%eax,4)", "%edx")

        mov("$0", "%eax")
        movb("alu_y+0", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s0-3")

        mov("$0", "%eax")
        movb("alu_y+1", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s1-2")

        mov("$0", "%eax")
        movb("alu_y+2", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s2-1")

        mov("$0", "%eax")
        movb("alu_y+3", "%al")
        mov("(%edx,%eax,4)", "%ecx")
        mov("%ecx", "alu_s3")

        combine_scratch_to_s()
        write_back("alu_s", dest_op)
        emit_update_zf_sf("alu_s")

    def impl_alu_sar(count_op, dest_op):
        emit("# -- alu_sar --")
        impl_alu_shr(count_op, dest_op)

        mov("alu_x", "%eax")
        mov("alu_rshi8s(,%eax,4)", "%edx")

        mov("$0", "%eax")
        movb("alu_y+3", "%al")
        mov("(%edx,%eax,4)", "%ecx")

        mov("alu_s", "%eax")
        emit("or %ecx, %eax")
        mov("%eax", "alu_s")

        write_back("alu_s", dest_op)
        emit_update_zf_sf("alu_s")


    def impl_lea(dest_op, mem_op):
        emit("# -- lea --")
        op_str = mem_op[1]
        mov("$0", "alu_s")

        import re
        match = re.match(r"^(-?\d+|[a-zA-Z0-9_]+)?(?:\((%[a-z0-9]+)?(?:,\s*(%[a-z0-9]+))?(?:,\s*(\d+))?\))?$", op_str)

        if match:
            disp, base, index, scale = match.groups()

            if index:
                scale_val = int(scale) if scale else 1
                load_to_scratch(('reg', index), "alu_x")

                if scale_val == 1:
                    pass
                elif scale_val == 2:
                    impl_core_add_logic("alu_x", "alu_x", "alu_x")
                elif scale_val == 4:
                    impl_core_add_logic("alu_x", "alu_x", "alu_x")
                    impl_core_add_logic("alu_x", "alu_x", "alu_x")
                elif scale_val == 8:
                    impl_core_add_logic("alu_x", "alu_x", "alu_x")
                    impl_core_add_logic("alu_x", "alu_x", "alu_x")
                    impl_core_add_logic("alu_x", "alu_x", "alu_x")

                impl_core_add_logic("alu_s", "alu_s", "alu_x")

            if base:
                load_to_scratch(('reg', base), "alu_x")
                impl_core_add_logic("alu_s", "alu_s", "alu_x")

            if disp:
                mov(f"${disp}", "alu_x")
                impl_core_add_logic("alu_s", "alu_s", "alu_x")
        else:
            mov(f"${op_str}", "alu_s")

        write_back("alu_s", dest_op)

    def impl_core_add_logic(res, x, y, carry_in_val=None):
        """
        Core 32-bit adder logic: res = x + y + carry_in_val
        Leaves the raw Carry Out in 'alu_t+2' (byte 2 of alu_t).
        """
        # Low 16-bit Add
        mov("$0", "%eax")
        mov("$0", "%ecx")
        movw(f"{x}+0", "%ax")
        movw(f"{y}+0", "%cx")

        # edx = x_low + y_low
        mov("alu_add16(,%eax,4)", "%edx")
        mov("(%edx,%ecx,4)", "%edx")

        # Apply Input Carry (Used for SUB +1)
        if carry_in_val:
            mov(f"{carry_in_val}", "%ecx")  # e.g., $1
            mov("alu_add16(,%edx,4)", "%eax")
            mov("(%eax,%ecx,4)", "%edx")

        # Store Low Result
        movw("%dx", f"{res}+0")

        # Calculate Carry to High
        # We need to save the low result + carry status before reusing registers
        mov("%edx", "alu_t")

        # High 16-bit Add
        mov("$0", "%eax")
        movw(f"{x}+2", "%ax")  # High X

        mov("$0", "%ecx")
        movw(f"{y}+2", "%cx")  # High Y

        # Add High X + High Y
        mov("alu_add16(,%eax,4)", "%edx")
        mov("(%edx,%ecx,4)", "%edx")

        # Add Carry from Low calculation
        mov("%edx", "%eax")
        mov("$0", "%ecx")
        movb("alu_t+2", "%cl")

        mov("alu_add16(,%eax,4)", "%edx")
        mov("(%edx,%ecx,4)", "%edx")

        # Store High Result
        movw("%dx", f"{res}+2")

        # Save Final Carry State
        # It is currently in byte 2 of EDX. We explicitly dump EDX to alu_t so the caller can check alu_t+2.
        mov("%edx", "alu_t")


    def impl_alu_add32(res, x, y):
        emit(f"# -- alu_add32 {res} = {x} + {y} --")

        impl_core_add_logic(res, x, y, carry_in_val=None)

        emit(f"# -- flags (ADD) --")

        # CF
        mov("$0", "%eax")
        movb("alu_t+2", "%al")
        movb("%al", "cf")

        emit_update_zf_sf(res)
        emit_update_of(x, y, res)

    def impl_alu_sub32(res, x, y, is_cmp=False):
        emit(f"# -- alu_sub32 {res} = {x} - {y} --")

        # Invert Y into alu_z0
        for i in range(4):
            mov("$0", "%eax")
            movb(f"{y}+{i}", "%al")
            movb("alu_inv8(%eax)", "%dl")
            movb("%dl", f"alu_z0+{i}")

        # Perform ADD (X + ~Y + 1)
        impl_core_add_logic(res, x, "alu_z0", carry_in_val="$1")

        emit(f"# -- flags (SUB/CMP) --")

        # CF = NOT(AdderCarry)
        mov("$0", "%eax")
        movb("alu_t+2", "%al")
        movb("alu_false(%eax)", "%al")
        movb("%al", "cf")

        emit_update_zf_sf(res)
        emit_update_of(x, y, res)

    def impl_bitwise(op_name, table_name, res, x, y):
        emit(f"# -- alu_{op_name} --")
        for i in range(4):
            mov("$0", "%eax")
            mov("$0", "%ebx")
            movb(f"{x}+{i}", "%al")
            movb(f"{y}+{i}", "%bl")
            mov(f"{table_name}(,%eax,4)", "%ecx")
            movb("(%ecx,%ebx)", "%dl")
            movb("%dl", f"{res}+{i}")

        # Update Flags
        emit(f"# -- flags ({op_name}) --")

        movb("$0", "cf")
        movb("$0", "of")

        # SF
        mov("$0", "%eax")
        movb(f"{res}+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        movb("%al", "sf")

        # ZF
        mov("$0", "%eax")
        mov("$0", "%edx")
        movb(f"{res}+0", "%dl")
        movb("alu_true(%edx)", "%al")
        movb(f"{res}+1", "%dl")
        movb("alu_true(%edx)", "%dl")
        emit("or %dl, %al")
        movb(f"{res}+2", "%dl")
        movb("alu_true(%edx)", "%dl")
        emit("or %dl, %al")
        movb(f"{res}+3", "%dl")
        movb("alu_true(%edx)", "%dl")
        emit("or %dl, %al")
        movb("alu_false(%eax)", "%al")
        movb("%al", "zf")

    def impl_alu_bxor8(res, x, y):
        impl_bitwise("xor", "alu_bxor8", res, x, y)

    def impl_alu_not(res, x):
        emit(f"# alu_not")
        for i in range(4):
            mov("$0", "%eax")
            movb(f"{x}+{i}", "%al")
            movb("alu_inv8(%eax)", "%dl")
            movb("%dl", f"{res}+{i}")

    def impl_alu_neg(res, x):
        emit(f"# -- alu_neg --")
        # Negate is effectively 0 - x
        mov("$0", "alu_z0")
        impl_alu_sub32(res, "alu_z0", x)

    def conditional_negate(target, cond_var, table_label):
        mov(target, "%ecx")
        mov(target, "%edx")
        mov("%edx", "alu_x")
        impl_alu_neg("alu_s", "alu_x")

        emit(".section .data")
        emit(f".align 4")
        emit(f"{table_label}: .long discard, {target}")
        emit(".text")

        mov(cond_var, "%eax")
        mov(f"{table_label}(,%eax,4)", "%eax")
        mov("alu_s", "%edx")
        mov("%edx", "(%eax)")

    def alu_add8n(s, s_off, c, x, x_off, y, y_off, extra_args=None):
        if extra_args is None: extra_args = []

        emit("# alu_add8n")
        mov("$0", "%ebx")
        mov("$0", "%edx")
        mov("$0", "%eax")

        # Initial Sum: val(x) + val(y)
        movb(f"{x}+{x_off}", "%al")
        movb(f"{y}+{y_off}", "%dl")
        mov("alu_mul_shl2(,%eax,4)", "%eax")
        mov("alu_mul_shl2(,%edx,4)", "%edx")
        mov("alu_mul_sums(%eax,%edx)", "%edx")

        # Add extra terms
        for i in range(0, len(extra_args), 2):
            p = extra_args[i]
            o = extra_args[i + 1]
            mov("$0", "%eax")
            movb(f"{p}+{o}", "%al")
            mov("alu_mul_shl2(,%edx,4)", "%edx")
            mov("alu_mul_shl2(,%eax,4)", "%eax")
            mov("alu_mul_sums(%eax,%edx)", "%edx")

        # Store Result
        movb("%dl", f"{s}+{s_off}")
        movb("%dh", f"{c}")

    def alu_mul8(s, s_off, x, x_off, y, y_off, c):
        # 8-bit Multiply: s[off] = x[off] * y[off] + c
        emit("# alu_mul8")
        mov("$0", "%eax")
        mov("$0", "%ebx")
        mov("$0", "%ecx")
        mov("$0", "%edx")

        movb(f"{x}+{x_off}", "%al")
        movb(f"{y}+{y_off}", "%dl")

        # Low byte of product
        mov("alu_mul_mul8l(,%eax,4)", "%ebx")
        movb("(%ebx,%edx)", "%cl")

        # High byte of product
        mov("alu_mul_mul8h(,%eax,4)", "%ebx")
        movb("(%ebx,%edx)", "%al")  # al = high part

        # Add Input Carry
        mov("$0", "%ebx")
        movb(f"{c}", "%dl")
        movb("alu_mul_sum8l(%ecx,%edx)", "%dl")  # low + carry
        movb("%dl", f"{s}+{s_off}")  # Store result byte

        # Calc Output Carry
        movb(f"{c}", "%dl")
        movb("alu_mul_sum8h(%ecx,%edx)", "%dl")  # carry of (low+carry)
        movb("alu_mul_sum8l(%edx,%eax)", "%dl")  # + high part
        movb("%dl", f"{c}")  # Store new carry

    def impl_alu_mul32(s, x, y):
        emit(f"# -- mul32 {s} = {x} * {y} --")
        for z in ["alu_z0", "alu_z1", "alu_z2", "alu_z3"]:
            mov("$0", z)

        c = "alu_c"

        mov("$0", c)
        alu_mul8("alu_z0", 0, x, 0, y, 0, c)
        alu_mul8("alu_z0", 1, x, 1, y, 0, c)
        alu_mul8("alu_z0", 2, x, 2, y, 0, c)
        alu_mul8("alu_z0", 3, x, 3, y, 0, c)

        movb(f"{c}", "%al")
        movb("%al", "alu_s0")

        mov("$0", c)
        alu_mul8("alu_z1", 1, x, 0, y, 1, c)
        alu_mul8("alu_z1", 2, x, 1, y, 1, c)
        alu_mul8("alu_z1", 3, x, 2, y, 1, c)
        movb(f"{c}", "%al")
        emit("or %al, alu_s0")

        mov("$0", c)
        alu_mul8("alu_z2", 2, x, 0, y, 2, c)
        alu_mul8("alu_z2", 3, x, 1, y, 2, c)
        movb(f"{c}", "%al")
        emit("or %al, alu_s0")


        mov("$0", c)
        alu_mul8("alu_z3", 3, x, 0, y, 3, c)
        movb(f"{c}", "%al")
        emit("or %al, alu_s0")

        # Summation
        mov("$0", c)
        alu_add8n(s, 0, c, "alu_z0", 0, "alu_c", 0)
        alu_add8n(s, 0, c, "alu_c", 2, "alu_z0", 0)
        alu_add8n(s, 0, c, "alu_z0", 0, "alu_c", 0)
        alu_add8n(s, 1, c, "alu_z0", 1, "alu_z1", 1, extra_args=["alu_c", 0])
        alu_add8n(s, 2, c, "alu_z0", 2, "alu_z1", 2, extra_args=["alu_z2", 2, "alu_c", 0])
        alu_add8n(s, 3, c, "alu_z0", 3, "alu_z1", 3, extra_args=["alu_z2", 3, "alu_z3", 3, "alu_c", 0])

        emit("# Update MUL Flags")
        mov("$0", "%eax")
        movb("alu_s0", "%al")
        movb("alu_true(%eax)", "%al")
        movb("%al", "cf")
        movb("%al", "of")

    def alu_bit(s, x, n):
        # Extract bit n from x into s
        emit(f"# alu_bit {x}[{n}] -> {s}")
        mov("$0", "%edx")
        byte_off = n // 8
        bit_idx = n % 8
        movb(f"{x}+{byte_off}", "%dl")
        mov(f"alu_b{bit_idx}(,%edx,4)", "%eax")
        mov("%eax", s)

    def alu_div_shl1_8_c(s, s_off, c):
        # Shift byte left 1, shifting in carry c
        emit("# shl1_8_c")
        mov("$0", "%eax")
        mov("$0", "%edx")
        movb(f"{s}+{s_off}", "%al")
        movb(f"{c}", "%dl")
        mov("alu_div_shl3_8_d(,%eax,4)", "%eax")
        mov("alu_div_shl1_8_c_d(%eax,%edx,4)", "%eax")
        movb("%al", f"{s}+{s_off}")
        movb("%ah", f"{c}")

    def alu_div_shl1_32_c(s, c):
        # Shift 32-bit left 1
        alu_div_shl1_8_c(s, 0, c)
        alu_div_shl1_8_c(s, 1, c)
        alu_div_shl1_8_c(s, 2, c)
        alu_div_shl1_8_c(s, 3, c)

    def alu_div_gte32(s, x, y, c_ignored=None):
        emit(f"# -- alu_div_gte32 {s} = ({x} >= {y}) --")

        mov(x, "%eax")
        mov("%eax", "alu_x")
        mov(y, "%eax")
        mov("%eax", "alu_y")

        impl_alu_sub32("alu_t", "alu_x", "alu_y", is_cmp=True)

        mov("$0", "%eax")
        movb("cf", "%al")
        movb("alu_false(%eax)", "%al")

        mov("%eax", s)

    def alu_div_setb32(s, n):
        # Set bit n in s
        byte_off = n // 8
        bit_idx = n % 8
        mov("$0", "%eax")
        movb(f"{s}+{byte_off}", "%al")
        movb(f"alu_b_s_{bit_idx}(%eax)", "%al")
        movb("%al", f"{s}+{byte_off}")

    def impl_alu_divrem(n, d):
        emit(f"# -- divrem n={n} d={d} --")

        mov(n, "%eax")
        mov("%eax", "alu_n")
        mov(d, "%eax")
        mov("%eax", "alu_d")

        mov("$0", "alu_q")
        mov("$0", "alu_r")

        for bit in range(31, -1, -1):
            alu_bit("alu_c", "alu_n", bit)
            alu_div_shl1_32_c("alu_r", "alu_c")

            alu_div_gte32("alu_t", "alu_r", "alu_d", "alu_c")

            mov("alu_t", "%eax")
            mov("alu_sel_r(,%eax,4)", "%edx")
            mov("%edx", "alu_psel_r")
            mov("alu_sel_q(,%eax,4)", "%edx")
            mov("%edx", "alu_psel_q")

            mov("alu_r", "%eax")
            mov("%eax", "alu_x")
            mov("alu_d", "%eax")
            mov("%eax", "alu_y")

            impl_alu_sub32("alu_sr", "alu_x", "alu_y")

            mov("alu_psel_r", "%eax")
            mov("alu_sr", "%edx")
            emit("movl %edx, (%eax)")

            # Set Quotient Bit: Q[bit] = 1 if R>=D
            mov("alu_psel_q", "%eax")
            mov("(%eax)", "%eax")
            mov("%eax", "alu_sq")
            alu_div_setb32("alu_sq", bit)
            mov("alu_psel_q", "%eax")
            mov("alu_sq", "%edx")
            mov("%edx", "(%eax)")

    # Dispatch

    if opcode == 'push':
        save_regs()
        impl_push(operands[0])
        restore_regs()

    elif opcode == 'pop':
        save_regs()
        impl_pop(operands[0])
        # If popping into a register, do NOT overwrite it during restore
        dest_reg = None
        d_type, d_val = get_op_str(operands[0])
        if d_type == 'reg': dest_reg = d_val
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'lea':
        save_regs()
        impl_lea(operands[1], operands[0])
        d_reg = operands[1][1] if operands[1][0] == 'reg' else None
        restore_regs(skip_reg=d_reg)

    elif opcode == 'loop':
        save_regs()
        target_label = operands[0][1]
        emit("# -- loop --")
        impl_dec_helper(('reg', '%ecx'))
        restore_regs(skip_reg='%ecx')
        emit(f"jnz {target_label}")

    elif opcode == 'inc':
        save_regs()
        impl_inc_helper(operands[0])
        d_reg = operands[0][1] if operands[0][0] == 'reg' else None
        restore_regs(skip_reg=d_reg)


    elif opcode == 'dec':
        save_regs()
        impl_dec_helper(operands[0])
        d_reg = operands[0][1] if operands[0][0] == 'reg' else None
        restore_regs(skip_reg=d_reg)

    elif opcode in ['shl', 'sal']:
        save_regs()
        impl_alu_shl(operands[0], operands[1])
        dest_reg = operands[1][1] if operands[1][0] == 'reg' else None
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'shr':
        save_regs()
        impl_alu_shr(operands[0], operands[1])
        dest_reg = operands[1][1] if operands[1][0] == 'reg' else None
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'sar':
        save_regs()
        impl_alu_sar(operands[0], operands[1])
        dest_reg = operands[1][1] if operands[1][0] == 'reg' else None
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'add':
        save_regs()
        load_to_scratch(operands[0], "alu_x")
        load_to_scratch(operands[1], "alu_y")
        impl_alu_add32("alu_s", "alu_y", "alu_x")
        write_back("alu_s", operands[1])

        dest_reg = None
        d_type, d_val = get_op_str(operands[1])
        if d_type == 'reg': dest_reg = d_val
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'sub':
        save_regs()
        load_to_scratch(operands[0], "alu_y")
        load_to_scratch(operands[1], "alu_x")
        impl_alu_sub32("alu_s", "alu_x", "alu_y")
        write_back("alu_s", operands[1])

        dest_reg = None
        d_type, d_val = get_op_str(operands[1])
        if d_type == 'reg': dest_reg = d_val
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'cmp':
        save_regs()

        load_to_scratch(operands[0], "alu_y")  # Src
        load_to_scratch(operands[1], "alu_x")  # Dest
        impl_alu_sub32("alu_s", "alu_x", "alu_y", is_cmp=True)

        restore_regs()

    elif opcode in ['and', 'or', 'xor']:
        save_regs()
        load_to_scratch(operands[0], "alu_x")
        load_to_scratch(operands[1], "alu_y")
        table_map = {'and': 'alu_band8', 'or': 'alu_bor8', 'xor': 'alu_bxor8'}
        impl_bitwise(opcode, table_map[opcode], "alu_s", "alu_x", "alu_y")
        write_back("alu_s", operands[1])

        dest_reg = None
        d_type, d_val = get_op_str(operands[1])
        if d_type == 'reg': dest_reg = d_val
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'not':
        save_regs()
        load_to_scratch(operands[0], "alu_x")
        impl_alu_not("alu_s", "alu_x")
        write_back("alu_s", operands[0])

        dest_reg = None
        d_type, d_val = get_op_str(operands[0])
        if d_type == 'reg': dest_reg = d_val
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'neg':
        save_regs()
        load_to_scratch(operands[0], "alu_x")
        impl_alu_neg("alu_s", "alu_x")
        write_back("alu_s", operands[0])

        dest_reg = None
        d_type, d_val = get_op_str(operands[0])
        if d_type == 'reg': dest_reg = d_val
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'test':
        # test A, B <=> push B
        #               and A, B
        #               pop B
        save_regs()

        impl_push(operands[1])

        load_to_scratch(operands[0], "alu_x")
        load_to_scratch(operands[1], "alu_y")
        impl_bitwise('and', 'alu_band8', "alu_s", "alu_x", "alu_y")

        impl_pop(operands[1])

        restore_regs()

    elif opcode in ['mul', 'imul']:
        save_regs()

        src_op = operands[0]
        dst_op = operands[1] if len(operands) > 1 else ('reg', '%eax')

        load_to_scratch(src_op, "alu_x")
        load_to_scratch(dst_op, "alu_y")

        impl_alu_mul32("alu_s", "alu_x", "alu_y")

        write_back("alu_s", dst_op)
        dest_reg = dst_op[1] if dst_op[0] == 'reg' else None
        restore_regs(skip_reg=dest_reg)

    elif opcode == 'div':
        save_regs()
        src_op = operands[0]

        load_to_scratch(src_op, "alu_d")
        mov("%eax", "alu_n")

        impl_alu_divrem("alu_n", "alu_d")

        mov("alu_q", "%eax")
        mov("alu_r", "%edx")

        restore_regs(skip_reg=['%eax', '%edx'])


    elif opcode == 'idiv':
        save_regs()

        src_op = operands[0]
        load_to_scratch(src_op, "alu_d")
        mov("%eax", "alu_n")
        mov("$0", "%eax")
        movb("alu_n+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        mov("%eax", "alu_ns")
        mov("$0", "%eax")
        movb("alu_d+3", "%al")
        mov("alu_b7(,%eax,4)", "%eax")
        mov("%eax", "alu_ds")

        load_to_scratch(('mem', 'alu_ns'), "alu_x")
        load_to_scratch(('mem', 'alu_ds'), "alu_y")
        impl_alu_bxor8("alu_qs", "alu_x", "alu_y")

        mov("alu_ns", "%eax")
        mov("%eax", "alu_rs")

        conditional_negate("alu_n", "alu_ns", "sel_abs_n")
        conditional_negate("alu_d", "alu_ds", "sel_abs_d")
        impl_alu_divrem("alu_n", "alu_d")
        conditional_negate("alu_q", "alu_qs", "sel_neg_q")
        conditional_negate("alu_r", "alu_rs", "sel_neg_r")

        mov("alu_q", "%eax")
        mov("alu_r", "%edx")
        restore_regs(skip_reg=['%eax', '%edx'])

    else:
        strs = []
        for op in operands:
            t, v = get_op_str(op)
            strs.append(v)
        emit(f"{opcode} {', '.join(strs)}")

    return lines

def process_alu_parsed_lines(parsed_output):
    """
    Takes the output of parse_asm_source and generates the final assembly
    with expanded ALU operations.
    """
    final_lines = []

    # Inject our massive table definitions at the beginning of the file
    final_lines.extend(generate_alu_tables())

    for line_tuple in parsed_output:
        key, val = line_tuple

        if key.startswith('.'):
            if key == '.data':
                final_lines.append(".data")
            else:
                # Reconstruct directive: .globl main
                args = [v[1] for v in val]
                final_lines.append(f"{key} {', '.join(args)}")
            continue

        # Handling Labels
        if key.endswith(':'):
            final_lines.append(key)
            continue

        # Handling Data Declarations (n: .long 6)
        # In the parser, these came as (label, [dtype, values])
        if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and val[0].startswith('.'):
            dtype = val[0]
            values = val[1]
            final_lines.append(f"\t{key}: {dtype} {', '.join(values)}")
            continue

        # Handling Instructions
        # Format: (opcode, [(type, val), ...])
        opcode = key
        operands = val

        # Pass to translation layer
        expanded = translate_alu_instruction(opcode, operands)
        final_lines.extend(expanded)

    return final_lines