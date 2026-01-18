.data
    v: .long 26, 12, 3, 56, 3, 18, 27, 35, 15
    n: .long 9
    max1: .long 0
    max2: .long 0
    formatAf: .ascii "%ld\n\0"

.text
.global main
main:
    mov n, %ecx
    lea v, %edi
    xor %eax, %eax

et_parcurgere:
    cmp $0, %ecx
    je et_afisare
    mov (%edi, %eax, 4), %edx
    mov max1, %ebx
    cmp %ebx, %edx
    jle verifica_max2

    mov %ebx, max2
    mov %edx, max1
    jmp et_cont_parcurgere

verifica_max2:
    mov max2, %ebx
    cmp %ebx, %edx
    jle et_cont_parcurgere
    mov max1, %esi
    cmp %esi, %edx
    jge et_cont_parcurgere

    mov %edx, max2

et_cont_parcurgere:
    inc %eax
    dec %ecx
    jmp et_parcurgere

et_afisare:
    push max2
    push $formatAf
    call printf
    add $8, %esp

    push $0
    call fflush
    add $4, %esp

et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80