#!/bin/bash

IN_DIR="samples/in"
OUT_DIR="samples/out"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

TEMP_FILE=".temp_runner_output"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}          TRANSPILER VERIFICATION TEST RUNNER          ${NC}"
echo -e "${BLUE}=======================================================${NC}"

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
    exe_out="./exe_transpiled_${filename%.s}"

    echo -e "\nProcessing: ${BLUE}$filename${NC}"

    if [ ! -f "$out_source" ]; then
        echo -e "${RED}[MISSING]${NC} Transformed file '$out_source' does not exist."
        continue
    fi

    compile_out_ref=$(gcc -m32 "$in_source" -o "$exe_in" -w -no-pie 2>&1)
    if [ $? -ne 0 ]; then
        echo -e "${RED}[COMPILE ERROR]${NC} Failed to compile reference '$in_source'"
        echo "$compile_out_ref"
        continue
    fi

    compile_out_transpiled=$(gcc -m32 "$out_source" -o "$exe_out" -w -no-pie 2>&1)
    if [ $? -ne 0 ]; then
        echo -e "${RED}[COMPILE ERROR]${NC} Failed to compile transformed '$out_source'"
        echo "$compile_out_transpiled"
        rm -f "$exe_in"
        continue
    fi

    "$exe_in" > "$TEMP_FILE" 2>&1
    ret_ref=$?
    output_ref=$(cat "$TEMP_FILE" | tr -d '\0')

    "$exe_out" > "$TEMP_FILE" 2>&1
    ret_transpiled=$?
    output_transpiled=$(cat "$TEMP_FILE" | tr -d '\0')

    fail=0

    if [ $ret_ref -ne $ret_transpiled ]; then
        echo -e "  ${RED}FAIL: Exit Code Mismatch${NC}"
        echo "    INPUT SAMPLE: $ret_ref"
        echo "    OUTPUT SAMPLE:  $ret_transpiled"
        fail=1
    fi

    if [ "$output_ref" != "$output_transpiled" ]; then
        echo -e "  ${RED}FAIL: Stdout/Stderr Mismatch${NC}"
        echo "    --- INPUT SAMPLE  ---"
        echo "$output_ref"
        echo "    --- OUTPUT SAMPLE ---"
        echo "$output_transpiled"
        echo "    ---------------------"
        fail=1
    fi

    if [ $fail -eq 0 ]; then
        echo -e "  ${GREEN}[PASS]${NC} Output: '$output_transpiled' | Exit Code: $ret_transpiled"
    fi

    rm -f "$exe_in" "$exe_out" "$TEMP_FILE"

done

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${BLUE}                   TEST RUN COMPLETE                   ${NC}"
echo -e "${BLUE}=======================================================${NC}"