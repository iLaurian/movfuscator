.text
.global main
main:
    mov $0b11011111111110111111, %eax
    xor %ebx, %ebx
    xor %ecx, %ecx
    mov $32, %edx

et_loop:
    test $1, %eax
    jz et_zero
    inc %ecx
    cmp %ebx, %ecx
    jle et_next
    mov %ecx, %ebx

et_next:
    shr $1, %eax
    dec %edx
    jnz et_loop
    jmp et_exit

et_zero:
    xor %ecx, %ecx
    jmp et_next

et_exit:
    mov %ebx, %eax
    mov $1, %ebx
    mov $1, %eax
    int $0x80