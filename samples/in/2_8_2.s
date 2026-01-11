.data
    n: .long 10
    t1: .long 0
    t2: .long 1
.text
.global main
main:
    mov n, %ecx
    sub $2, %ecx
et_loop:
    mov t2, %eax
    mov t1, %ebx
    add t2, %ebx
    mov %ebx, t2
    mov %eax, t1
    loop et_loop
et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80