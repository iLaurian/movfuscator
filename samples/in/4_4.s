.data
    v: .long 26, 12, 3, 56, 3, 18, 27, 35, 15
    n: .long 9
    frecv: .space 404
    formatAfisare: .asciz "Numarul care apare cele mai multe ori: %d\n"

.text
.global main
.extern printf

main:
    mov n, %ecx
    lea v, %edi
    xor %eax, %eax

et_tablou_frecv:
    cmp $0, %ecx
    je et_parcurgere
    mov (%edi, %eax, 4), %ebx
    lea frecv, %esi
    mov (%esi, %ebx, 4), %edx
    add $1, %edx
    mov %edx, (%esi, %ebx, 4)
    inc %eax
    dec %ecx
    jmp et_tablou_frecv

et_parcurgere:
    mov $0, %eax
    mov $0, %ebx
    mov $0, %edx
    lea frecv, %esi

et_cautare:
    cmp $100, %eax
    jg afisare
    mov (%esi, %eax, 4), %ecx
    cmp %edx, %ecx
    jle et_cont_parcurgere
    mov %ecx, %edx
    mov %eax, %ebx

et_cont_parcurgere:
    inc %eax
    jmp et_cautare

afisare:
    push %ebx
    push $formatAfisare
    call printf
    add $8, %esp

et_exit:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80