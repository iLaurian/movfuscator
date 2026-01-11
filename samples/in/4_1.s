.data
    n: .long 9
    v: .long 12, 15, 5, 15, 4, 1, 7, 15, 1
    maxim: .long 0
    ap: .long 0

.text
.global main
main:
    mov n, %ecx
    mov $v, %edi

et_loop:
    mov n, %ebx
    sub %ecx, %ebx
    mov (%edi, %ebx, 4), %edx
    cmp maxim, %edx
    jg et_max_nou
    loop et_loop
    jmp cont_max

et_max_nou:
    mov %edx, maxim
    loop et_loop

cont_max:
    mov n, %ecx
    mov $v, %edi

loop_ap:
    mov n, %ebx
    sub %ecx, %ebx
    mov (%edi, %ebx, 4), %edx
    cmp maxim, %edx
    je et_egale
    loop loop_ap
    jmp et_exit

et_egale:
    inc ap
    loop loop_ap

et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80