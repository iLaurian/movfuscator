#!/bin/bash

IN_DIR="samples/in"
OUT_DIR="samples/out"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}       ALU TRANSPILER VERIFICATION TEST RUNNER        ${NC}"
echo -e "${BLUE}======================================================${NC}"

if [ ! -d "$IN_DIR" ]; then
    echo -e "${RED}Error: Directory '$IN_DIR' not found.${NC}"
    exit 1
fi
if [ ! -d "$OUT_DIR" ]; then
    echo -e "${RED}Error: Directory '$OUT_DIR' not found.${NC}"
    exit 1
fi

for in_source in "$IN_DIR"/*.s; do
    [ -e "$in_source" ] || continue

    filename=$(basename "$in_source")
    out_source="$OUT_DIR/$filename"

    exe_in="./exe_ref_${filename%.s}"
    exe_out="./exe_alu_${filename%.s}"

    echo -e "\nProcessing: ${BLUE}$filename${NC}"

    if [ ! -f "$out_source" ]; then
        echo -e "${RED}[MISSING]${NC} Transformed file '$out_source' does not exist."
        continue
    fi

    gcc -m32 "$in_source" -o "$exe_in" -w -no-pie
    if [ $? -ne 0 ]; then
        echo -e "${RED}[COMPILE ERROR]${NC} Failed to compile reference '$in_source'"
        continue
    fi

    gcc -m32 "$out_source" -o "$exe_out" -w -no-pie
    if [ $? -ne 0 ]; then
        echo -e "${RED}[COMPILE ERROR]${NC} Failed to compile transformed '$out_source'"
        rm -f "$exe_in"
        continue
    fi

    output_ref=$("$exe_in" 2>&1)
    ret_ref=$?

    output_alu=$("$exe_out" 2>&1)
    ret_alu=$?

    fail=0

    if [ $ret_ref -ne $ret_alu ]; then
        echo -e "  ${RED}FAIL: Exit Code Mismatch${NC}"
        echo "    INPUT SAMPLE: $ret_ref"
        echo "    OUTPUT SAMPLE:  $ret_alu"
        fail=1
    fi

    if [ "$output_ref" != "$output_alu" ]; then
        echo -e "  ${RED}FAIL: Stdout/Stderr Mismatch${NC}"
        echo "    --- INPUT SAMPLE  ---"
        echo "$output_ref"
        echo "    --- OUTPUT SAMPLE ---"
        echo "$output_alu"
        echo "    ---------------------"
        fail=1
    fi

    if [ $fail -eq 0 ]; then
        echo -e "  ${GREEN}[PASS]${NC} Output: '$output_alu' | Exit Code: $ret_alu"
    fi

    rm -f "$exe_in" "$exe_out"

done

echo -e "\n${BLUE}======================================================${NC}"
echo -e "${BLUE}                   TEST RUN COMPLETE                  ${NC}"
echo -e "${BLUE}======================================================${NC}"