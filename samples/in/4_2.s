.data
    n: .long 5234
    s: .long 0
    formatAfSuma: .ascii "Suma cifrelor numarului este: %ld\n\0"

.text
.global main
main:
    mov n, %ecx
    mov $10, %ebx

et_loop:
    cmp $0, %ecx
    je et_afisare
    mov %ecx, %eax
    xor %edx, %edx
    div %ebx
    add %edx, s
    mov %eax, %ecx
    jmp et_loop

et_afisare:
    push s
    push $formatAfSuma
    call printf
    pop %ebx
    pop %ebx
    push stdout
    call fflush
    add $4, %esp

et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80