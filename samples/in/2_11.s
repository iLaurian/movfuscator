.data
    n: .long 6
    s: .long 0
.text
.global main
main:
    mov n, %ecx
    xor %eax, %eax
    xor %ebx, %ebx
et_loop:
    cmp $0, %ecx
    je et_exit
    mov %ecx, %eax
    mul %eax
    add %eax, %ebx
    dec %ecx
    jmp et_loop
et_exit:
    mov %ebx, s
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80