from enum import Enum

class Labels(Enum):
  # Locations in memory used to temporarly store data (or just to provide a valid
  # address to write something to)
  TMP1 = "MVF_TMP1"
  TMP2 = "MVF_TMP2"
  TMP3 = "MVF_TMP3"
  TMP4 = "MVF_TMP4"

  # Special places in memory used to temporarly save some registers
  REGISTER_EAX = "MVF_REGISTER_EAX"
  REGISTER_EBX = "MVF_REGISTER_EBX"
  REGISTER_ECX = "MVF_REGISTER_ECX"
  REGISTER_EDX = "MVF_REGISTER_EDX"
  REGISTER_ESI = "MVF_REGISTER_ESI"
  REGISTER_EDI = "MVF_REGISTER_EDI"
  REGISTER_EBP = "MVF_REGISTER_EBP"
  REGISTER_ESP = "MVF_REGISTER_ESP"

  # ALU flags
  FLAG_ZERO = "MVF_FLAG_ZERO"
  FLAG_SIGN = "MVF_FLAG_SIGN"
  FLAG_OVERFLOW = "MVF_FLAG_OVERFLOW"
  FLAG_CARRY = "MVF_FLAG_CARRY"

  # Some function names for the runtime
  SETUP_FUNCTION_NAME = "MVF_SETUP"
  DISPATCHER_FUNCTION_NAME = "MVF_DISPATCHER"

  # Parameters for the dispatcher
  DISPATCHER_JUMP_ADDRESS = "MVF_DISPATCHER_JUMP_ADDRESS"

  # Address of the lookup table used for conditional jumps (it provides
  # an invalid address at position 0, a valid one at position 1 and again a invalid one at position 2)
  CONDITIONAL_JUMP_LOOKUP_TABLE = "MVF_CONDITIONAL_JUMP_LUT"

  SIMPLE_AND_LOOKUP_TABLE = "MVF_1BIT_AND_LUT"
  SIMPLE_AND_0_LOOKUP_TABLE = "MVF_1BIT_AND_0_LUT"
  SIMPLE_AND_1_LOOKUP_TABLE = "MVF_1BIT_AND_1_LUT"

  SIMPLE_OR_LOOKUP_TABLE = "MVF_1BIT_OR_LUT"
  SIMPLE_OR_0_LOOKUP_TABLE = "MVF_1BIT_OR_0_LUT"
  SIMPLE_OR_1_LOOKUP_TABLE = "MVF_1BIT_OR_1_LUT"

  SIMPLE_XOR_LOOKUP_TABLE = "MVF_1BIT_XOR_LUT"
  SIMPLE_XOR_0_LOOKUP_TABLE = "MVF_1BIT_XOR_0_LUT"
  SIMPLE_XOR_1_LOOKUP_TABLE = "MVF_1BIT_XOR_1_LUT"

  SIMPLE_NOT_LOOKUP_TABLE = "MVF_1BIT_NOT_LUT"


class Instruction:
  def __init__(self, instruction_tuple):
    self.mnemonic = instruction_tuple[0].strip().lower()
    self.operands = []
    self.suffix = ''

    if len(instruction_tuple) >= 2:
      self.operands = instruction_tuple[1]

    for instr in ['jmp', 'cmp']:
      if self.mnemonic.startswith(instr) and len(self.mnemonic) == len(instr) + 1 and self.mnemonic[-1] in ['b', 'w', 'l']:
        self.suffix = self.mnemonic[-1]
        self.mnemonic = self.mnemonic[:-1]
        break

  def att_form(self):
    operands = ', '.join(op[1] for op in self.operands)
    return f"{self.mnemonic}{self.suffix} {operands}"

class BranchingFuscator:
  def __init__(self, file):
    self.file = file
    self.lines = []

  def emit(self, code: str):
    self.lines.append(code)
    # self.file.write(code + '\n')
  
  def initialize_startup_routines(self):
    self.emit(f"""
      .data
      {Labels.TMP1.value}: .long 0
      {Labels.TMP2.value}: .long 0
      {Labels.TMP3.value}: .long 0
      {Labels.TMP4.value}: .long 0
              
      {Labels.REGISTER_EAX.value}: .long 0
      {Labels.REGISTER_EBX.value}: .long 0
      {Labels.REGISTER_ECX.value}: .long 0
      {Labels.REGISTER_EDX.value}: .long 0
      {Labels.REGISTER_ESI.value}: .long 0
      {Labels.REGISTER_EDI.value}: .long 0
      {Labels.REGISTER_EBP.value}: .long 0
      {Labels.REGISTER_ESP.value}: .long 0
              
      {Labels.FLAG_ZERO.value}: .byte 0
      {Labels.FLAG_SIGN.value}: .byte 0
      {Labels.FLAG_OVERFLOW.value}: .byte 0
      {Labels.FLAG_CARRY.value}: .byte 0

      {Labels.SIMPLE_AND_LOOKUP_TABLE.value}: .long {Labels.SIMPLE_AND_0_LOOKUP_TABLE.value}, {Labels.SIMPLE_AND_1_LOOKUP_TABLE.value}
      {Labels.SIMPLE_AND_0_LOOKUP_TABLE.value}: .long 0, 0
      {Labels.SIMPLE_AND_1_LOOKUP_TABLE.value}: .long 0, 1

      {Labels.SIMPLE_OR_LOOKUP_TABLE.value}: .long {Labels.SIMPLE_OR_0_LOOKUP_TABLE.value}, {Labels.SIMPLE_OR_1_LOOKUP_TABLE.value}
      {Labels.SIMPLE_OR_0_LOOKUP_TABLE.value}: .long 0, 1
      {Labels.SIMPLE_OR_1_LOOKUP_TABLE.value}: .long 1, 1

      {Labels.SIMPLE_XOR_LOOKUP_TABLE.value}: .long {Labels.SIMPLE_XOR_0_LOOKUP_TABLE.value}, {Labels.SIMPLE_XOR_1_LOOKUP_TABLE.value}
      {Labels.SIMPLE_XOR_0_LOOKUP_TABLE.value}: .long 0, 1
      {Labels.SIMPLE_XOR_1_LOOKUP_TABLE.value}: .long 1, 0

      {Labels.SIMPLE_NOT_LOOKUP_TABLE.value}: .long 1, 0
              
      {Labels.DISPATCHER_JUMP_ADDRESS.value}: .long 0
      __ARRIVED_AT_DISPATCHER_DEBUG_STR: .asciz "Arrived at the MVF_DISPATCHER"

      {Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}: .long 0, {Labels.TMP1.value}, 0

      .text
      # ############################## THE DISPATCHER ##############################
      {Labels.DISPATCHER_FUNCTION_NAME.value}:
      push $__ARRIVED_AT_DISPATCHER_DEBUG_STR
      call puts
      add $4, %esp

      # %eax now contains a pointer to a ucontext_t struct
      # which stores the values of all the registers at the moment
      # of the crash (including EIP)
      mov 12(%esp), %eax

      # %ebx now stores the address where we should redirect execution to
      mov {Labels.DISPATCHER_JUMP_ADDRESS.value}, %ebx
      
      # This sets the EIP to whatever is stored in {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov %ebx, 76(%eax)

      # The way we currently implement conditional jumps uses
      # register %edi (it sets it to a valid or invalid address and then attempts to dereference it),
      # so we must restore its value (we also suppose that the intentional segfaulty
      # code that brought us here saved the value of %edi in MVF_REGISTER_EDI before using it to trigger
      # the segfault)
      mov MVF_REGISTER_EDI, %edi
      mov %edi, 36(%eax)

      ret
      {Labels.DISPATCHER_FUNCTION_NAME.value}_END:
      # #############################################################################

      {Labels.SETUP_FUNCTION_NAME.value}:
      # This:
      #  - registers MVF_DISPATCHER to be a SIGSEV handler

      # Allocating enough space for:
      #  - struct sigaction (offset: 0 - 140)
      push %ebp
      mov %esp, %ebp
      sub $140, %esp

      # sa_resolver = NULL
      # sa_flags = SIGSEV
      # sa_sigaction = &MVF_DISPATCHER
      movl $0, -4(%ebp)
      movl $4, -8(%ebp)
      movl $MVF_DISPATCHER, -140(%ebp)

      # Putting in eax the address of the struct
      mov %ebp, %eax
      sub $140, %eax

      # Registering the handler: sigaction(SIGSEV, struct sigaction, NULL)
      push $0
      push %eax
      push $11
      call sigaction
      add $12, %esp

      # Poping the frame and returning
      add $140, %esp
      pop %ebp
      ret # can be removed
      {Labels.SETUP_FUNCTION_NAME.value}_END:

      .section .init_array
      .long {Labels.SETUP_FUNCTION_NAME.value}
    """)

  def emit_final_assembly(self, instructions: list):
    self.initialize_startup_routines()
    self.emit("# ################ CODE STARTS HERE ################")
    for raw_instruction in instructions:
      if type(raw_instruction) == str:
        self.emit(raw_instruction)
      else:
        instruction = Instruction(raw_instruction)
        self.translate_instruction(instruction)

  def copy_ALU_flags(self, support_custom_alu = True):
    self.emit(f"""
      setz {Labels.FLAG_ZERO.value}
      seto {Labels.FLAG_OVERFLOW.value}
      sets {Labels.FLAG_SIGN.value}
      setc {Labels.FLAG_CARRY.value}
    """)
    if support_custom_alu:
      self.emit(f"""
        mov %eax, {Labels.REGISTER_EAX.value}
        
        mov $zf, %eax
        movb (%eax), {Labels.FLAG_ZERO.value}

        mov $of, %eax
        movb (%eax), {Labels.FLAG_OVERFLOW.value}

        mov $sf, %eax
        movb (%eax), {Labels.FLAG_SIGN.value}

        mov $cf, %eax
        movb (%eax), {Labels.FLAG_CARRY.value}

        mov {Labels.REGISTER_EAX.value}, %eax
      """)


  def translate_instruction(self, instruction: Instruction):
    match instruction.mnemonic:
      # Jumps using EFLAGS
      case "jnc":
        self.translate_flag_jmp(instruction, Labels.FLAG_CARRY, False)
      case "jc":
        self.translate_flag_jmp(instruction, Labels.FLAG_CARRY, True)
      case "jns":
        self.translate_flag_jmp(instruction, Labels.FLAG_SIGN, False)
      case "js":
        self.translate_flag_jmp(instruction, Labels.FLAG_SIGN, True)
      case "jno":
        self.translate_flag_jmp(instruction, Labels.FLAG_OVERFLOW, False)
      case "jo":
        self.translate_flag_jmp(instruction, Labels.FLAG_OVERFLOW, True)
      case "jnz":
        self.translate_flag_jmp(instruction, Labels.FLAG_ZERO, False)
      case "jz":
        self.translate_flag_jmp(instruction, Labels.FLAG_ZERO, True)

      # Conditional jumps
      ## Equality (je same as jz) and Inequality (jne same as jnz)
      case "jne":
        self.translate_flag_jmp(instruction, Labels.FLAG_ZERO, False)
      case "je":
        self.translate_flag_jmp(instruction, Labels.FLAG_ZERO, True)

      ## Signed inequalities
      case "jl":
        self.translate_jl(instruction)  # SF != OF
      case "jle":
        self.translate_jle(instruction) # ZF == 1 or SF != OF
      case "jg":
        self.translate_jg(instruction)  # ZF == 0 and SF == OF
      case "jge":
        self.translate_jge(instruction) # SF == OF

      ## Unsigned inequalities
      case "jb":
        # CF == 1 (so same as jc)
        self.translate_flag_jmp(instruction, Labels.FLAG_CARRY, True)
      case "jbe":
        self.translate_jbe(instruction) # CF == 1 or ZF == 1
      case "ja":
        self.translate_ja(instruction)  # CF == 0 and ZF == 0
      case "jae":
        # CF == 0 (so same as jnc)
        self.translate_flag_jmp(instruction, Labels.FLAG_CARRY, False)

      case "cmp":
        self.translate_cmp(instruction)
      case "jmp":
        self.translate_jmp(instruction)
      case _:
        self.emit(instruction.att_form())

  def translate_ja(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %ebx, {Labels.REGISTER_EBX.value}
      mov %ecx, {Labels.REGISTER_ECX.value}
      mov %edi, {Labels.REGISTER_EDI.value}

      mov $0, %eax
      mov $0, %ecx
      movb {Labels.FLAG_CARRY.value}, %al
      movb {Labels.FLAG_ZERO.value}, %cl

      # %eax: CF ^ 1 (same as CF == 0)
      mov ${Labels.SIMPLE_XOR_1_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %eax

      # %ecx: ZF ^ 1 (same as ZF == 0)
      mov (%ebx, %ecx, 4), %ecx

      # %eax: (CF ^ 1) and (ZF ^ 1) -> same as (CF == 0) and (ZF == 0)
      mov ${Labels.SIMPLE_AND_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %ebx
      mov (%ebx, %ecx, 4), %eax

      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %ebx
      mov 4(%ebx, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      mov {Labels.REGISTER_EBX.value}, %ebx
      mov {Labels.REGISTER_ECX.value}, %ecx

      movl $0xCAFE, (%edi)
      mov {Labels.REGISTER_EDI.value}, %edi

      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_jbe(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %ebx, {Labels.REGISTER_EBX.value}
      mov %ecx, {Labels.REGISTER_ECX.value}
      mov %edi, {Labels.REGISTER_EDI.value}

      mov $0, %eax
      mov $0, %ecx
      movb {Labels.FLAG_CARRY.value}, %al
      movb {Labels.FLAG_ZERO.value}, %cl

      # %eax: CF & 1 (same as CF == 1)
      mov ${Labels.SIMPLE_AND_1_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %eax

      # %ecx: ZF & 1 (same as ZF == 1)
      mov (%ebx, %ecx, 4), %ecx

      # %eax: (CF & 1) or (ZF & 1) -> same as (CF == 1) or (ZF == 1)
      mov ${Labels.SIMPLE_OR_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %ebx
      mov (%ebx, %ecx, 4), %eax

      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %ebx
      mov 4(%ebx, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      mov {Labels.REGISTER_EBX.value}, %ebx
      mov {Labels.REGISTER_ECX.value}, %ecx

      movl $0xCAFE, (%edi)
      mov {Labels.REGISTER_EDI.value}, %edi

      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_jl(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %ebx, {Labels.REGISTER_EBX.value}
      mov %edi, {Labels.REGISTER_EDI.value}

      # %eax = SF ^ OF -> same value as (SF != OF)
      mov $0, %eax
      movb {Labels.FLAG_SIGN.value}, %al
      mov ${Labels.SIMPLE_XOR_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %ebx
      mov $0, %eax
      movb {Labels.FLAG_OVERFLOW.value}, %al
      mov (%ebx, %eax, 4), %eax

      # Preparing the jump via the dispatcher
      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %ebx
      mov 4(%ebx, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      mov {Labels.REGISTER_EBX.value}, %ebx

      movl $0xDEAD, (%edi)
      mov {Labels.REGISTER_EDI.value}, %edi
      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_jge(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %ebx, {Labels.REGISTER_EBX.value}
      mov %edi, {Labels.REGISTER_EDI.value}

      # %ebx = SF ^
      mov $0, %eax
      movb {Labels.FLAG_SIGN.value}, %al
      mov ${Labels.SIMPLE_XOR_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %ebx

      # %eax = SF ^ OF
      mov $0, %eax
      movb {Labels.FLAG_OVERFLOW.value}, %al
      mov (%ebx, %eax, 4), %eax

      # %eax = !(SF ^ OF) -> same value as SF == OF
      mov ${Labels.SIMPLE_NOT_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %eax

      # 3...2...1...
      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %ebx
      mov 4(%ebx, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      mov {Labels.REGISTER_EBX.value}, %ebx

      # ...JUMP!
      movl $0x31337, (%edi)
      mov {Labels.REGISTER_EDI.value}, %edi
      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_jg(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %ebx, {Labels.REGISTER_EBX.value}
      mov %ecx, {Labels.REGISTER_ECX.value}
      mov %edi, {Labels.REGISTER_EDI.value}

      # %eax = ZF ^ 1 (which has the same value as ZF == 0)
      mov $0, %ecx
      movb {Labels.FLAG_ZERO.value}, %cl
      mov ${Labels.SIMPLE_XOR_1_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %ecx, 4), %eax

      # %eax = (ZF ^ 1) and
      mov ${Labels.SIMPLE_AND_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %eax

      # %ebx = SF ^
      mov $0, %ecx
      movb {Labels.FLAG_SIGN.value}, %cl
      mov ${Labels.SIMPLE_XOR_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %ecx, 4), %ebx

      # %ecx = SF ^ OF
      mov $0, %ecx
      movb {Labels.FLAG_OVERFLOW.value}, %cl
      mov (%ebx, %ecx, 4), %ecx

      # %ecx = !(SF ^ OF) -> same value as SF == OF
      mov ${Labels.SIMPLE_NOT_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %ecx, 4), %ecx

      # %eax = (ZF ^ 1) and !(SF ^ OF) -> same value as ZF == 0 and SF == OF
      mov (%eax, %ecx, 4), %eax

      # 3...2...1...
      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %ebx
      mov 4(%ebx, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      mov {Labels.REGISTER_EBX.value}, %ebx
      mov {Labels.REGISTER_ECX.value}, %ecx

      # ...JUMP!
      movl $0x31337, (%edi)
      mov {Labels.REGISTER_EDI.value}, %edi
      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_jle(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %ebx, {Labels.REGISTER_EBX.value}
      mov %ecx, {Labels.REGISTER_ECX.value}
      mov %edi, {Labels.REGISTER_EDI.value}

      # %eax = (ZF == 1) = (ZF & 1)
      mov $0, %ecx
      movb {Labels.FLAG_ZERO.value}, %cl
      mov ${Labels.SIMPLE_AND_1_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %ecx, 4), %eax

      # %eax: stores the address for the OR LUT 
      # based on the previous value of (ZF == 1)
      mov ${Labels.SIMPLE_OR_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %eax, 4), %eax

      # %ecx = (SF != OF) = (SF ^ OF)
      mov $0, %ecx
      movb {Labels.FLAG_SIGN.value}, %cl
      mov ${Labels.SIMPLE_XOR_LOOKUP_TABLE.value}, %ebx
      mov (%ebx, %ecx, 4), %ebx
      mov $0, %ecx
      movb {Labels.FLAG_OVERFLOW.value}, %cl
      mov (%ebx, %ecx, 4), %ecx

      # %eax = (ZF == 1) OR (SF != OF)
      mov (%eax, %ecx, 4), %eax

      # Preparing the jump via the dispatcher
      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %ebx
      mov 4(%ebx, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      mov {Labels.REGISTER_EBX.value}, %ebx
      mov {Labels.REGISTER_ECX.value}, %ecx

      movl $0xB00B5, (%edi)
      mov {Labels.REGISTER_EDI.value}, %edi
      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_cmp(self, instruction: Instruction):
    self.emit(instruction.att_form())
    self.copy_ALU_flags()

  def translate_flag_jmp(self, instruction: Instruction, flag_label: Labels, version = False):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      # ### BEGIN TRANSLATION FOR: {instruction.att_form()} ###
      mov %eax, {Labels.REGISTER_EAX.value}
      mov %edi, {Labels.REGISTER_EDI.value}
      
      mov $0, %eax
      mov {flag_label.value}, %al
      
      mov ${Labels.CONDITIONAL_JUMP_LOOKUP_TABLE.value}, %edi
      mov {4 if version else 0}(%edi, %eax, 4), %edi
      mov {Labels.REGISTER_EAX.value}, %eax
      
      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      movl $0xDEAD, (%edi)
      
      mov {Labels.REGISTER_EDI.value}, %edi
      # ###  END TRANSLATION FOR: {instruction.att_form()}  ###
    """)

  def translate_jmp(self, instruction: Instruction):
    destination = BranchingFuscator.get_destination_string_for_jump(instruction)
    self.emit(f"""
      movl {destination}, {Labels.DISPATCHER_JUMP_ADDRESS.value}
      mov %edi, {Labels.REGISTER_EDI.value}
      movl $0xDEAD, 0
    """)

  @staticmethod
  def get_destination_string_for_jump(instruction: Instruction):
    addressing_mode, target = instruction.operands[0]
    return ("$" + target) if addressing_mode == "label" else target



assembly_code__JMP = [
    ".data",
    # Indented data definitions
    '\tstr1: .asciz "Will jump to LABEL_A via the dispatcher"',
    '\tstr2: .asciz "Arrived at LABEL_B"',
    '\tstr3: .asciz "Arrived at LABEL_A"',
    '\tstr4: .asciz "Will jump to LABEL_B via the dispatcher"',
    '\tstr5: .asciz "Arrived at the MVF_DISPATCHER"',
    
    "\n.text",
    ".global main",
    
    "main:",
    # Instructions with '\t' prefix for indentation
    ("\tpush", [("imm", "$str1")]),
    ("\tcall", [("label", "puts")]),
    ("\tpop",  [("reg", "%eax")]),
    
    # Triggering SIGSEGV
    ("\tjmp",  [("label", "LABEL_A")]),
    
    "\nLABEL_B:",
    ("\tpush", [("imm", "$str2")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\tmov",  [("imm", "$0"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")]),
    
    "\nLABEL_A:",
    ("\tpush", [("imm", "$str3")]),
    ("\tcall", [("label", "puts")]),
    ("\tpop",  [("reg", "%eax")]),
    
    ("\tpush", [("imm", "$str4")]),
    ("\tcall", [("label", "puts")]),
    ("\tpop",  [("reg", "%eax")]),
    
    # Triggering SIGSEGV
    ("\tjmp",  [("label", "LABEL_B")])
]

assembly_code__JZ = [
    ".data",
    '\tstr1: .asciz "[!] Arrived at LABEL_B. This shouldn\'t happen"',
    '\tstr2: .asciz "[+] Arrived at LABEL_A, as expected"',
    '\tstr3: .asciz "[!] Arrived at LABEL_C. This shouldn\'t happen"',
    '\tvalue_zero: .long 0',

    "\n.text",
    ".global main",
    "main:",
    ("\tcmpl", [("imm", "$0"), ("mem", "value_zero")]),
    ("\tjz",   [("label", "LABEL_A")]),

    "\nLABEL_B:",
    ("\tpush", [("imm", "$str1")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")]),

    "\nLABEL_A:",
    ("\tpush", [("imm", "$str2")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    ("\tcmpl", [("imm", "$1"), ("mem", "value_zero")]),
    ("\tjz",   [("label", "LABEL_C")]),

    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_C:",
    ("\tpush", [("imm", "$str3")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")])
]

# Test created by Gemini
assembly_code__JNZ = [
    ".data",
    '\tstr_fail: .asciz "[!] JNZ jumped when it should have fallen through!"',
    '\tstr_pass1: .asciz "[+] Fallthrough check passed (ZF=1, JNZ didn\'t jump)"',
    '\tstr_pass2: .asciz "[+] Jump check passed (ZF=0, JNZ jumped)"',
    '\tvalue_zero: .long 0',

    "\n.text",
    ".global main",
    "main:",
    
    # TEST 1: Compare 0 with 0. Result is 0 (ZF=1).
    # JNZ should NOT jump. If it does, go to error.
    ("\tcmpl", [("imm", "$0"), ("mem", "value_zero")]),
    ("\tjnz",  [("label", "LABEL_FAIL")]),

    # If we are here, JNZ correctly fell through.
    ("\tpush", [("imm", "$str_pass1")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    # TEST 2: Compare 1 with 0. Result is not 0 (ZF=0).
    # JNZ SHOULD jump.
    ("\tcmpl", [("imm", "$1"), ("mem", "value_zero")]),
    ("\tjnz",  [("label", "LABEL_SUCCESS")]),

    # If we are here, JNZ failed to jump. Fall through to fail message.

    "\nLABEL_FAIL:",
    ("\tpush", [("imm", "$str_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_SUCCESS:",
    ("\tpush", [("imm", "$str_pass2")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Another test by Gemini (USE IT WHEN WE HAVE A ALU)
assembly_code__JZ_Thorough = [
    ".data",
    '\tstr_math_pass: .asciz "[+] Math check passed (sub created Zero)"',
    '\tstr_math_fail: .asciz "[!] Math check failed (sub didn\'t trigger JZ)"',
    '\tstr_test_pass: .asciz "[+] Logic check passed (test saw Zero)"',
    '\tstr_test_fail: .asciz "[!] Logic check failed (test didn\'t trigger JZ)"',

    "\n.text",
    ".global main",
    "main:",

    # TEST 1: Arithmetic subtraction
    # mov eax, 5; sub eax, 5 -> Result is 0, ZF should be 1
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tsub",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tjz",   [("label", "LABEL_MATH_OK")]),

    # Fail block for Math
    ("\tpush", [("imm", "$str_math_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_MATH_OK:",
    ("\tpush", [("imm", "$str_math_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    # TEST 2: Bitwise TEST
    # xor ebx, ebx (sets ebx to 0)
    # test ebx, ebx (updates flags based on AND result). Result 0, ZF=1
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\ttest", [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tjz",   [("label", "LABEL_TEST_OK")]),

    # Fail block for Test
    ("\tpush", [("imm", "$str_test_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_TEST_OK:",
    ("\tpush", [("imm", "$str_test_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__JZ_CmpOnly = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Imm cmp) passed: JZ jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Imm cmp) failed: JZ did not jump"',
    '\tstr_t2_pass: .asciz "[+] Test 2 (Reg cmp) passed: JZ jumped correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Reg cmp) failed: JZ did not jump"',

    "\n.text",
    ".global main",
    "main:",

    # TEST 1: Immediate Comparison
    # Check if register (0) equals Immediate (0)
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjz",   [("label", "LABEL_TEST1_PASS")]),

    # --- FAIL CASE 1 ---
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_TEST1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    # TEST 2: Register Comparison
    # Load 0x42 into EAX and EBX. Compare them.
    ("\tmov",  [("imm", "$0x42"), ("reg", "%eax")]),
    ("\tmov",  [("imm", "$0x42"), ("reg", "%ebx")]),
    ("\tcmp",  [("reg", "%eax"), ("reg", "%ebx")]),
    ("\tjz",   [("label", "LABEL_TEST2_PASS")]),

    # --- FAIL CASE 2 ---
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_TEST2_PASS:",
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

assembly_code__JLE = [
    ".data",
    '\tstr1: .asciz "[!] Arrived at LABEL_B. This shouldn\'t happen"',
    '\tstr2: .asciz "[+] Arrived at LABEL_A, as expected"',
    '\tstr3: .asciz "[!] Arrived at LABEL_C. This shouldn\'t happen"',
    '\tvalue_zero: .long 0',

    "\n.text",
    ".global main",
    "main:",
    
    # cmpl $10, value_zero (0 - 10 = -10)
    ("\tcmpl", [("imm", "$10"), ("mem", "value_zero")]),
    
    # jle should jump because -10 is Less or Equal to 0
    ("\tjle",  [("label", "LABEL_A")]),

    "\nLABEL_B:",
    ("\tpush", [("imm", "$str1")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")]),

    "\nLABEL_A:",
    ("\tpush", [("imm", "$str2")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    # cmpl $1, value_zero (0 - 1 = -1)
    ("\tcmpl", [("imm", "$1"), ("mem", "value_zero")]),
    
    # jz should NOT jump because result is not 0
    ("\tjz",   [("label", "LABEL_C")]),

    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_C:",
    ("\tpush", [("imm", "$str3")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")])
]

# Test created by Gemini
assembly_code__JLE_Test = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Less Than) passed: JLE jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Less Than) failed: JLE did not jump"',
    
    '\tstr_t2_pass: .asciz "[+] Test 2 (Equal To) passed: JLE jumped correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Equal To) failed: JLE did not jump"',
    
    '\tstr_t3_pass: .asciz "[+] Test 3 (Greater Than) passed: JLE fell through correctly"',
    '\tstr_t3_fail: .asciz "[!] Test 3 (Greater Than) failed: JLE jumped incorrectly"',

    "\n.text",
    ".global main",
    "main:",

    # -------------------------------------------------------------
    # TEST 1: LESS THAN
    # We compare 0 ($eax) - 10 ($imm). Result is -10.
    # Logic: -10 <= 0 is True. JLE Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjle",  [("label", "LABEL_T1_PASS")]),

    # Fail Block T1
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 2: EQUAL TO
    # We compare 5 ($eax) - 5 ($imm). Result is 0.
    # Logic: 0 <= 0 is True. JLE Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjle",  [("label", "LABEL_T2_PASS")]),

    # Fail Block T2
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T2_PASS:",
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 3: GREATER THAN
    # We compare 20 ($eax) - 10 ($imm). Result is 10.
    # Logic: 10 <= 0 is False. JLE Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$20"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjle",  [("label", "LABEL_T3_FAIL")]),

    # Success Block T3 (Fallthrough)
    ("\tpush", [("imm", "$str_t3_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T3_FAIL:",
    ("\tpush", [("imm", "$str_t3_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__JG_Test = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Greater Than) passed: JG jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Greater Than) failed: JG did not jump"',
    
    '\tstr_t2_pass: .asciz "[+] Test 2 (Equal To) passed: JG fell through correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Equal To) failed: JG jumped incorrectly"',
    
    '\tstr_t3_pass: .asciz "[+] Test 3 (Less Than) passed: JG fell through correctly"',
    '\tstr_t3_fail: .asciz "[!] Test 3 (Less Than) failed: JG jumped incorrectly"',

    "\n.text",
    ".global main",
    "main:",

    # -------------------------------------------------------------
    # TEST 1: GREATER THAN
    # We compare 10 ($eax) - 0 ($imm). Result is +10.
    # Flags: SF=0, OF=0 (Equal), ZF=0.
    # Logic: 10 > 0 is True. JG Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjg",   [("label", "LABEL_T1_PASS")]),

    # Fail Block T1
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 2: EQUAL TO
    # We compare 5 ($eax) - 5 ($imm). Result is 0.
    # Flags: ZF=1.
    # Logic: 5 > 5 is False. JG Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjg",   [("label", "LABEL_T2_FAIL")]),

    # Success Block T2 (Fallthrough)
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    # Jump over the fail block to Test 3
    ("\tjmp",  [("label", "TEST_3")]),

    "\nLABEL_T2_FAIL:",
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),


    # -------------------------------------------------------------
    # TEST 3: LESS THAN
    # We compare 0 ($eax) - 10 ($imm). Result is -10.
    # Flags: SF=1, OF=0 (Not Equal).
    # Logic: 0 > 10 is False. JG Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    "\nTEST_3:",
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjg",   [("label", "LABEL_T3_FAIL")]),

    # Success Block T3 (Fallthrough)
    ("\tpush", [("imm", "$str_t3_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T3_FAIL:",
    ("\tpush", [("imm", "$str_t3_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__JGE_Test = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Greater Than) passed: JGE jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Greater Than) failed: JGE did not jump"',
    
    '\tstr_t2_pass: .asciz "[+] Test 2 (Equal To) passed: JGE jumped correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Equal To) failed: JGE did not jump"',
    
    '\tstr_t3_pass: .asciz "[+] Test 3 (Less Than) passed: JGE fell through correctly"',
    '\tstr_t3_fail: .asciz "[!] Test 3 (Less Than) failed: JGE jumped incorrectly"',

    "\n.text",
    ".global main",
    "main:",

    # -------------------------------------------------------------
    # TEST 1: GREATER THAN
    # We compare 10 ($eax) - 0 ($imm). Result is +10.
    # Flags: SF=0, OF=0. (SF == OF)
    # Logic: 10 >= 0 is True. JGE Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjge",  [("label", "LABEL_T1_PASS")]),

    # Fail Block T1
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 2: EQUAL TO
    # We compare 5 ($eax) - 5 ($imm). Result is 0.
    # Flags: SF=0, OF=0. (SF == OF)
    # Logic: 5 >= 5 is True. JGE Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjge",  [("label", "LABEL_T2_PASS")]),

    # Fail Block T2
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T2_PASS:",
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 3: LESS THAN
    # We compare 0 ($eax) - 10 ($imm). Result is -10.
    # Flags: SF=1, OF=0. (SF != OF)
    # Logic: 0 >= 10 is False. JGE Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjge",  [("label", "LABEL_T3_FAIL")]),

    # Success Block T3 (Fallthrough)
    ("\tpush", [("imm", "$str_t3_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T3_FAIL:",
    ("\tpush", [("imm", "$str_t3_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__JL_Test = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Less Than) passed: JL jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Less Than) failed: JL did not jump"',
    
    '\tstr_t2_pass: .asciz "[+] Test 2 (Equal To) passed: JL fell through correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Equal To) failed: JL jumped incorrectly"',
    
    '\tstr_t3_pass: .asciz "[+] Test 3 (Greater Than) passed: JL fell through correctly"',
    '\tstr_t3_fail: .asciz "[!] Test 3 (Greater Than) failed: JL jumped incorrectly"',

    "\n.text",
    ".global main",
    "main:",

    # -------------------------------------------------------------
    # TEST 1: LESS THAN
    # We compare 0 ($eax) - 10 ($imm). Result is -10.
    # Flags: SF=1, OF=0. (SF != OF)
    # Logic: 0 < 10 is True. JL Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjl",   [("label", "LABEL_T1_PASS")]),

    # Fail Block T1
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 2: EQUAL TO
    # We compare 5 ($eax) - 5 ($imm). Result is 0.
    # Flags: SF=0, OF=0. (SF == OF)
    # Logic: 5 < 5 is False. JL Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjl",   [("label", "LABEL_T2_FAIL")]),

    # Success Block T2 (Fallthrough)
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    # Jump to Test 3
    ("\tjmp",  [("label", "TEST_3")]),

    "\nLABEL_T2_FAIL:",
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),


    # -------------------------------------------------------------
    # TEST 3: GREATER THAN
    # We compare 10 ($eax) - 0 ($imm). Result is +10.
    # Flags: SF=0, OF=0. (SF == OF)
    # Logic: 10 < 0 is False. JL Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    "\nTEST_3:",
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjl",   [("label", "LABEL_T3_FAIL")]),

    # Success Block T3 (Fallthrough)
    ("\tpush", [("imm", "$str_t3_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T3_FAIL:",
    ("\tpush", [("imm", "$str_t3_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__JA_Test = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Above) passed: JA jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Above) failed: JA did not jump"',
    
    '\tstr_t2_pass: .asciz "[+] Test 2 (Equal To) passed: JA fell through correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Equal To) failed: JA jumped incorrectly"',
    
    '\tstr_t3_pass: .asciz "[+] Test 3 (Below) passed: JA fell through correctly"',
    '\tstr_t3_fail: .asciz "[!] Test 3 (Below) failed: JA jumped incorrectly"',

    "\n.text",
    ".global main",
    "main:",

    # -------------------------------------------------------------
    # TEST 1: ABOVE (Unsigned >)
    # We compare 10 ($eax) - 5 ($imm). Result is 5.
    # Flags: CF=0 (No Borrow), ZF=0 (Not Zero).
    # Logic: 10 > 5 is True. JA Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tja",   [("label", "LABEL_T1_PASS")]),

    # Fail Block T1
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 2: EQUAL TO
    # We compare 5 ($eax) - 5 ($imm). Result is 0.
    # Flags: CF=0, ZF=1.
    # Logic: 5 > 5 is False. JA Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tja",   [("label", "LABEL_T2_FAIL")]),

    # Success Block T2 (Fallthrough)
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    # Jump to Test 3
    ("\tjmp",  [("label", "TEST_3")]),

    "\nLABEL_T2_FAIL:",
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),


    # -------------------------------------------------------------
    # TEST 3: BELOW (Unsigned <)
    # We compare 5 ($eax) - 10 ($imm). 
    # Logic: 5 - 10 causes a Borrow (Unsigned Underflow).
    # Flags: CF=1, ZF=0.
    # Logic: 5 > 10 is False. JA Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    "\nTEST_3:",
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tja",   [("label", "LABEL_T3_FAIL")]),

    # Success Block T3 (Fallthrough)
    ("\tpush", [("imm", "$str_t3_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T3_FAIL:",
    ("\tpush", [("imm", "$str_t3_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__JBE_Test = [
    ".data",
    '\tstr_t1_pass: .asciz "[+] Test 1 (Below) passed: JBE jumped correctly"',
    '\tstr_t1_fail: .asciz "[!] Test 1 (Below) failed: JBE did not jump"',
    
    '\tstr_t2_pass: .asciz "[+] Test 2 (Equal To) passed: JBE jumped correctly"',
    '\tstr_t2_fail: .asciz "[!] Test 2 (Equal To) failed: JBE did not jump"',
    
    '\tstr_t3_pass: .asciz "[+] Test 3 (Above) passed: JBE fell through correctly"',
    '\tstr_t3_fail: .asciz "[!] Test 3 (Above) failed: JBE jumped incorrectly"',

    "\n.text",
    ".global main",
    "main:",

    # -------------------------------------------------------------
    # TEST 1: BELOW (Unsigned <)
    # We compare 5 ($eax) - 10 ($imm). 
    # Logic: 5 - 10 causes a Borrow (CF=1).
    # Logic: 5 <= 10 is True. JBE Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjbe",  [("label", "LABEL_T1_PASS")]),

    # Fail Block T1
    ("\tpush", [("imm", "$str_t1_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T1_PASS:",
    ("\tpush", [("imm", "$str_t1_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 2: EQUAL TO
    # We compare 10 ($eax) - 10 ($imm). Result is 0.
    # Logic: ZF=1.
    # Logic: 10 <= 10 is True. JBE Should Jump.
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjbe",  [("label", "LABEL_T2_PASS")]),

    # Fail Block T2
    ("\tpush", [("imm", "$str_t2_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T2_PASS:",
    ("\tpush", [("imm", "$str_t2_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),


    # -------------------------------------------------------------
    # TEST 3: ABOVE (Unsigned >)
    # We compare 10 ($eax) - 5 ($imm). Result is 5.
    # Logic: No Borrow (CF=0), Not Zero (ZF=0).
    # Logic: 10 <= 5 is False. JBE Should NOT Jump (Fallthrough).
    # -------------------------------------------------------------
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjbe",  [("label", "LABEL_T3_FAIL")]),

    # Success Block T3 (Fallthrough)
    ("\tpush", [("imm", "$str_t3_pass")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nLABEL_T3_FAIL:",
    ("\tpush", [("imm", "$str_t3_fail")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

# Test created by Gemini
assembly_code__ALL_JUMPS = [
    ".data",
    '\tmsg_start: .asciz "[*] Starting Ultimate Jump Test..."',
    '\tmsg_ok:    .asciz "[+] All tests passed successfully!"',
    
    # Error Messages
    '\terr_jmp: .asciz "[!] JMP failed"',
    '\terr_je:  .asciz "[!] JE (Equal) failed"',
    '\terr_jne: .asciz "[!] JNE (Not Equal) failed"',
    '\terr_jz:  .asciz "[!] JZ (Zero) failed"',
    '\terr_jnz: .asciz "[!] JNZ (Not Zero) failed"',
    '\terr_js:  .asciz "[!] JS (Sign/Negative) failed"',
    '\terr_jns: .asciz "[!] JNS (No Sign/Positive) failed"',
    '\terr_jo:  .asciz "[!] JO (Overflow) failed"',
    '\terr_jno: .asciz "[!] JNO (No Overflow) failed"',
    '\terr_jc:  .asciz "[!] JC (Carry) failed"',
    '\terr_jnc: .asciz "[!] JNC (No Carry) failed"',
    '\terr_jb:  .asciz "[!] JB (Below) failed"',
    '\terr_jae: .asciz "[!] JAE (Above/Equal) failed"',
    '\terr_ja:  .asciz "[!] JA (Above) failed"',
    '\terr_jbe: .asciz "[!] JBE (Below/Equal) failed"',
    '\terr_jl:  .asciz "[!] JL (Less) failed"',
    '\terr_jle: .asciz "[!] JLE (Less/Equal) failed"',
    '\terr_jg:  .asciz "[!] JG (Greater) failed"',
    '\terr_jge: .asciz "[!] JGE (Greater/Equal) failed"',

    "\n.text",
    ".global main",
    "main:",
    
    # Print Start Message
    ("\tpush", [("imm", "$msg_start")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    # ---------------------------------------------------------
    # 1. Unconditional JMP
    # ---------------------------------------------------------
    ("\tjmp",  [("label", "TEST_JE")]),
    # Fail
    ("\tpush", [("imm", "$err_jmp")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # 2. Equality (JE, JZ)
    # ---------------------------------------------------------
    "\nTEST_JE:",
    # 5 == 5 -> ZF=1
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tje",   [("label", "TEST_JZ")]),
    # Fail
    ("\tpush", [("imm", "$err_je")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JZ:",
    # 0 == 0 -> ZF=1
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjz",   [("label", "TEST_JNE")]),
    # Fail
    ("\tpush", [("imm", "$err_jz")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # 3. Inequality (JNE, JNZ)
    # ---------------------------------------------------------
    "\nTEST_JNE:",
    # 5 != 4 -> ZF=0
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$4"), ("reg", "%eax")]),
    ("\tjne",  [("label", "TEST_JNZ")]),
    # Fail
    ("\tpush", [("imm", "$err_jne")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JNZ:",
    # 1 != 0 -> ZF=0
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjnz",  [("label", "TEST_JS")]),
    # Fail
    ("\tpush", [("imm", "$err_jnz")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # 4. Sign Flags (JS, JNS)
    # ---------------------------------------------------------
    "\nTEST_JS:",
    # 0 - 5 = -5 (Negative) -> SF=1
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjs",   [("label", "TEST_JNS")]),
    # Fail
    ("\tpush", [("imm", "$err_js")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JNS:",
    # 10 - 5 = 5 (Positive) -> SF=0
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjns",  [("label", "TEST_JO")]),
    # Fail
    ("\tpush", [("imm", "$err_jns")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # 5. Overflow Flags (JO, JNO)
    # ---------------------------------------------------------
    "\nTEST_JO:",
    # MinInt (0x80000000) - 1. 
    # Negative - Positive = Positive (Overflow) -> OF=1
    ("\tmov",  [("imm", "$0x80000000"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$1"), ("reg", "%eax")]),
    ("\tjo",   [("label", "TEST_JNO")]),
    # Fail
    ("\tpush", [("imm", "$err_jo")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JNO:",
    # 10 - 1 = 9. No Overflow -> OF=0
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$1"), ("reg", "%eax")]),
    ("\tjno",  [("label", "TEST_JC")]),
    # Fail
    ("\tpush", [("imm", "$err_jno")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # 6. Unsigned Jumps (JC/JB, JNC/JAE, JA, JBE)
    # ---------------------------------------------------------
    "\nTEST_JC:",
    # 5 - 10 = Borrow (Carry) -> CF=1
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjc",   [("label", "TEST_JB")]),
    # Fail
    ("\tpush", [("imm", "$err_jc")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JB:",
    # Same as JC: 5 < 10 Unsigned
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjb",   [("label", "TEST_JNC")]),
    # Fail
    ("\tpush", [("imm", "$err_jb")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JNC:",
    # 10 - 5 = No Borrow -> CF=0
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjnc",  [("label", "TEST_JAE")]),
    # Fail
    ("\tpush", [("imm", "$err_jnc")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JAE:",
    # Same as JNC: 10 >= 5 Unsigned
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjae",  [("label", "TEST_JA")]),
    # Fail
    ("\tpush", [("imm", "$err_jae")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JA:",
    # 10 > 5 Unsigned -> CF=0, ZF=0
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tja",   [("label", "TEST_JBE")]),
    # Fail
    ("\tpush", [("imm", "$err_ja")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JBE:",
    # 5 <= 10 Unsigned -> CF=1
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjbe",  [("label", "TEST_JL")]),
    # Fail
    ("\tpush", [("imm", "$err_jbe")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # 7. Signed Jumps (JL, JLE, JG, JGE)
    # ---------------------------------------------------------
    "\nTEST_JL:",
    # 0 < 10 Signed -> SF=1, OF=0 (SF!=OF)
    ("\tmov",  [("imm", "$0"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$10"), ("reg", "%eax")]),
    ("\tjl",   [("label", "TEST_JLE")]),
    # Fail
    ("\tpush", [("imm", "$err_jl")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JLE:",
    # 5 <= 5 Signed -> ZF=1
    ("\tmov",  [("imm", "$5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$5"), ("reg", "%eax")]),
    ("\tjle",  [("label", "TEST_JG")]),
    # Fail
    ("\tpush", [("imm", "$err_jle")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JG:",
    # 10 > 0 Signed -> SF=0, OF=0, ZF=0
    ("\tmov",  [("imm", "$10"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$0"), ("reg", "%eax")]),
    ("\tjg",   [("label", "TEST_JGE")]),
    # Fail
    ("\tpush", [("imm", "$err_jg")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    "\nTEST_JGE:",
    # -5 >= -10 Signed -> SF=0, OF=0
    ("\tmov",  [("imm", "$-5"), ("reg", "%eax")]),
    ("\tcmpl", [("imm", "$-10"), ("reg", "%eax")]),
    ("\tjge",  [("label", "FINAL_SUCCESS")]),
    # Fail
    ("\tpush", [("imm", "$err_jge")]),
    ("\tcall", [("label", "puts")]),
    ("\tjmp",  [("label", "EXIT")]),

    # ---------------------------------------------------------
    # Success & Exit
    # ---------------------------------------------------------
    "\nFINAL_SUCCESS:",
    ("\tpush", [("imm", "$msg_ok")]),
    ("\tcall", [("label", "puts")]),
    ("\tadd",  [("imm", "$4"), ("reg", "%esp")]),

    "\nEXIT:",
    ("\tmov",  [("imm", "$1"), ("reg", "%eax")]),
    ("\txor",  [("reg", "%ebx"), ("reg", "%ebx")]),
    ("\tint",  [("imm", "$0x80")])
]

def process_branching_parsed_lines(assembly_code):
  obfuscator = BranchingFuscator(None)
  obfuscator.emit_final_assembly(assembly_code)
  return obfuscator.lines

# with open("test1.s", "w") as f:
#   obfuscator = BranchingFuscator(f)
#   obfuscator.emit_final_assembly(assembly_code__ALL_JUMPS)


