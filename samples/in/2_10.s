.data
    x: .long 77
.text
.global main
main:
    mov x, %eax
    mov $1, %ebx
et_loop:
    cmp %eax, %ebx
    ja et_exit
    shl $1, %ebx
    jmp et_loop
et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80