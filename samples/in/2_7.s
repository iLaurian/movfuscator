.data
    a: .ascii "Assembly"
    b: .word 0x25
    c: .asciz "x86"
    d: .asciz ";;;"
    e: .long 0x15

.text
    mov $4, %eax
    mov $1, %ebx
    mov $a, %ecx
    mov $b, %edi
    sub %ecx, %edi
    or %edi, e
    mov e, %edx
    int $0x80