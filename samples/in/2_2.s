.data
.text
.global main
main:
    mov $0, %ecx
    mov $3, %eax
    mov $5, %ebx
    cmp %ebx, %eax
    jge greater_or_equal

    mov $1, %ecx
    jmp end

greater_or_equal:
    mov $2, %ecx

end:
    mov $1, %eax
    xor %ebx, %ebx
    int $0x80