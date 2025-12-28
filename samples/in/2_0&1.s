.data
    str1: .ascii "ASC"
    str2: .long 0X31323334

.text
.global main
main:
    mov $4, %eax
    mov $1, %ebx
    mov $str1, %ecx
    mov $5, %edx
    int $0x80