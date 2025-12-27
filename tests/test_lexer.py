"""
Tests for the MOVFUSCATOR Lexer.

Run with: pytest tests/test_lexer.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import Lexer, Token, TokenType, LexerError, tokenize, tokenize_file


class TestBasicTokenization:
    """Test basic tokenization functionality."""

    def test_empty_input(self):
        """Empty input should return only EOF."""
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only(self):
        """Whitespace only should return only EOF."""
        tokens = tokenize("   \t  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_single_newline(self):
        """Single newline should produce NEWLINE + EOF."""
        tokens = tokenize("\n")
        assert len(tokens) == 2
        assert tokens[0].type == TokenType.NEWLINE
        assert tokens[1].type == TokenType.EOF


class TestDirectives:
    """Test directive tokenization."""

    def test_common_directives(self):
        """Test common assembler directives."""
        directives = [
            ".data",
            ".text",
            ".global",
            ".long",
            ".word",
            ".ascii",
            ".asciz",
            ".space",
            ".extern",
        ]
        for directive in directives:
            tokens = tokenize(directive)
            assert tokens[0].type == TokenType.DIRECTIVE
            assert tokens[0].value == directive

    def test_directive_with_args(self):
        """Test directive with arguments."""
        tokens = tokenize(".global main")
        assert tokens[0].type == TokenType.DIRECTIVE
        assert tokens[0].value == ".global"
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "main"

    def test_long_directive(self):
        """Test .long with number."""
        tokens = tokenize(".long 10")
        assert tokens[0].type == TokenType.DIRECTIVE
        assert tokens[0].value == ".long"
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "10"

    def test_long_directive_multiple_values(self):
        """Test .long with multiple comma-separated values."""
        tokens = tokenize(".long 26, 12, 3")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].type == TokenType.DIRECTIVE
        assert filtered[1].type == TokenType.NUMBER
        assert filtered[1].value == "26"
        assert filtered[2].type == TokenType.COMMA
        assert filtered[3].type == TokenType.NUMBER
        assert filtered[3].value == "12"


class TestRegisters:
    """Test register tokenization."""

    def test_32bit_registers(self):
        """Test 32-bit general purpose registers."""
        registers = ["%eax", "%ebx", "%ecx", "%edx", "%esi", "%edi", "%esp", "%ebp"]
        for reg in registers:
            tokens = tokenize(reg)
            assert tokens[0].type == TokenType.REGISTER
            assert tokens[0].value == reg

    def test_16bit_registers(self):
        """Test 16-bit registers."""
        registers = ["%ax", "%bx", "%cx", "%dx"]
        for reg in registers:
            tokens = tokenize(reg)
            assert tokens[0].type == TokenType.REGISTER
            assert tokens[0].value == reg

    def test_8bit_registers(self):
        """Test 8-bit registers."""
        registers = ["%al", "%ah", "%bl", "%bh", "%cl", "%ch", "%dl", "%dh"]
        for reg in registers:
            tokens = tokenize(reg)
            assert tokens[0].type == TokenType.REGISTER
            assert tokens[0].value == reg


class TestImmediates:
    """Test immediate value tokenization."""

    def test_decimal_immediate(self):
        """Test decimal immediate values."""
        tokens = tokenize("$5")
        assert tokens[0].type == TokenType.IMMEDIATE
        assert tokens[0].value == "$5"

    def test_negative_immediate(self):
        """Test negative immediate values."""
        tokens = tokenize("$-1")
        assert tokens[0].type == TokenType.IMMEDIATE
        assert tokens[0].value == "$-1"

    def test_hex_immediate(self):
        """Test hexadecimal immediate values."""
        tokens = tokenize("$0x80")
        assert tokens[0].type == TokenType.IMMEDIATE
        assert tokens[0].value == "$0x80"

    def test_negative_hex_immediate(self):
        """Test negative hexadecimal immediate."""
        tokens = tokenize("$-0xFF")
        assert tokens[0].type == TokenType.IMMEDIATE
        assert tokens[0].value == "$-0xFF"

    def test_binary_immediate(self):
        """Test binary immediate values."""
        tokens = tokenize("$0b1010")
        assert tokens[0].type == TokenType.IMMEDIATE
        assert tokens[0].value == "$0b1010"

    def test_label_as_immediate(self):
        """Test label reference as immediate."""
        tokens = tokenize("$formatAfSuma")
        assert tokens[0].type == TokenType.IMMEDIATE
        assert tokens[0].value == "$formatAfSuma"


class TestNumbers:
    """Test number tokenization (without $ prefix)."""

    def test_decimal_number(self):
        """Test decimal numbers."""
        tokens = tokenize("10")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "10"

    def test_negative_number(self):
        """Test negative numbers."""
        tokens = tokenize("-10")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "-10"

    def test_hex_number(self):
        """Test hexadecimal numbers."""
        tokens = tokenize("0x80")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0x80"


class TestIdentifiers:
    """Test identifier tokenization."""

    def test_simple_identifier(self):
        """Test simple identifiers."""
        tokens = tokenize("main")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "main"

    def test_identifier_with_underscore(self):
        """Test identifiers with underscores."""
        tokens = tokenize("et_loop")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "et_loop"

    def test_identifier_with_numbers(self):
        """Test identifiers with numbers."""
        tokens = tokenize("label2")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "label2"


class TestLabels:
    """Test label tokenization."""

    def test_label_definition(self):
        """Test label definition with colon."""
        tokens = tokenize("main:")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "main"
        assert tokens[1].type == TokenType.COLON

    def test_label_with_underscore(self):
        """Test label with underscore."""
        tokens = tokenize("et_loop:")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "et_loop"
        assert tokens[1].type == TokenType.COLON


class TestInstructions:
    """Test instruction tokenization."""

    def test_mov_instruction(self):
        """Test MOV instruction."""
        tokens = tokenize("movl $5, %eax")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].type == TokenType.IDENTIFIER
        assert filtered[0].value == "movl"
        assert filtered[1].type == TokenType.IMMEDIATE
        assert filtered[1].value == "$5"
        assert filtered[2].type == TokenType.COMMA
        assert filtered[3].type == TokenType.REGISTER
        assert filtered[3].value == "%eax"

    def test_add_instruction(self):
        """Test ADD instruction."""
        tokens = tokenize("addl %ebx, %eax")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].value == "addl"
        assert filtered[1].value == "%ebx"
        assert filtered[3].value == "%eax"

    def test_cmp_instruction(self):
        """Test CMP instruction."""
        tokens = tokenize("cmp %ebx, %eax")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].value == "cmp"
        assert filtered[1].value == "%ebx"
        assert filtered[3].value == "%eax"

    def test_jump_instructions(self):
        """Test jump instructions."""
        jumps = ["jmp", "je", "jne", "jz", "jnz", "jg", "jge", "jl", "jle", "ja", "jb"]
        for jmp in jumps:
            tokens = tokenize(f"{jmp} label")
            assert tokens[0].type == TokenType.IDENTIFIER
            assert tokens[0].value == jmp

    def test_int_instruction(self):
        """Test INT instruction."""
        tokens = tokenize("int $0x80")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].value == "int"
        assert filtered[1].value == "$0x80"


class TestMemoryAddressing:
    """Test memory addressing modes."""

    def test_simple_indirect(self):
        """Test simple indirect addressing: (%eax)."""
        tokens = tokenize("(%eax)")
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.REGISTER
        assert tokens[1].value == "%eax"
        assert tokens[2].type == TokenType.RPAREN

    def test_base_index_scale(self):
        """Test base + index * scale addressing: (%edi, %ecx, 4)."""
        tokens = tokenize("(%edi, %ecx, 4)")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].type == TokenType.LPAREN
        assert filtered[1].type == TokenType.REGISTER
        assert filtered[1].value == "%edi"
        assert filtered[2].type == TokenType.COMMA
        assert filtered[3].type == TokenType.REGISTER
        assert filtered[3].value == "%ecx"
        assert filtered[4].type == TokenType.COMMA
        assert filtered[5].type == TokenType.NUMBER
        assert filtered[5].value == "4"
        assert filtered[6].type == TokenType.RPAREN

    def test_displacement_base(self):
        """Test displacement + base: 8(%ebp)."""
        tokens = tokenize("8(%ebp)")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].type == TokenType.NUMBER
        assert filtered[0].value == "8"
        assert filtered[1].type == TokenType.LPAREN
        assert filtered[2].type == TokenType.REGISTER
        assert filtered[2].value == "%ebp"
        assert filtered[3].type == TokenType.RPAREN

    def test_label_index_scale(self):
        """Test label + index * scale: v(, %eax, 4)."""
        tokens = tokenize("v(, %eax, 4)")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].type == TokenType.IDENTIFIER
        assert filtered[0].value == "v"
        assert filtered[1].type == TokenType.LPAREN
        assert filtered[2].type == TokenType.COMMA
        assert filtered[3].type == TokenType.REGISTER
        assert filtered[4].type == TokenType.COMMA
        assert filtered[5].type == TokenType.NUMBER

    def test_mov_with_memory(self):
        """Test MOV with memory operand."""
        tokens = tokenize("movl (%edi, %ecx, 4), %eax")
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].value == "movl"
        # Memory operand
        assert filtered[1].type == TokenType.LPAREN
        assert filtered[2].value == "%edi"
        assert filtered[3].type == TokenType.COMMA
        assert filtered[4].value == "%ecx"
        assert filtered[5].type == TokenType.COMMA
        assert filtered[6].value == "4"
        assert filtered[7].type == TokenType.RPAREN
        # Comma between operands
        assert filtered[8].type == TokenType.COMMA
        # Destination register
        assert filtered[9].value == "%eax"


class TestComments:
    """Test comment tokenization."""

    def test_hash_comment(self):
        """Test # style comment."""
        tokens = tokenize("# this is a comment")
        assert tokens[0].type == TokenType.COMMENT
        assert "this is a comment" in tokens[0].value

    def test_semicolon_comment(self):
        """Test ; style comment."""
        tokens = tokenize("; this is a comment")
        assert tokens[0].type == TokenType.COMMENT
        assert "this is a comment" in tokens[0].value

    def test_inline_comment(self):
        """Test inline comment after instruction."""
        tokens = tokenize("movl $5, %eax  # load 5 into eax")
        types = [t.type for t in tokens]
        assert TokenType.COMMENT in types


class TestStrings:
    """Test string literal tokenization."""

    def test_simple_string(self):
        """Test simple string literal."""
        tokens = tokenize('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == '"hello"'

    def test_string_with_escape(self):
        """Test string with escape sequences."""
        tokens = tokenize(r'"hello\n"')
        assert tokens[0].type == TokenType.STRING

    def test_ascii_directive_with_string(self):
        """Test .ascii directive with string."""
        tokens = tokenize('.ascii "Hello World\\n\\0"')
        filtered = [
            t for t in tokens if t.type not in (TokenType.EOF, TokenType.NEWLINE)
        ]
        assert filtered[0].type == TokenType.DIRECTIVE
        assert filtered[0].value == ".ascii"
        assert filtered[1].type == TokenType.STRING


class TestLineAndColumnTracking:
    """Test line and column tracking."""

    def test_line_numbers(self):
        """Test that line numbers are tracked correctly."""
        code = """line1
line2
line3"""
        tokens = tokenize(code)
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert identifiers[0].line == 1
        assert identifiers[1].line == 2
        assert identifiers[2].line == 3

    def test_column_numbers(self):
        """Test that column numbers are tracked correctly."""
        tokens = tokenize("  movl $5, %eax")
        # movl should be at column 3 (after 2 spaces)
        movl_token = tokens[0]
        assert movl_token.value == "movl"
        assert movl_token.column == 3


class TestErrorHandling:
    """Test error handling."""

    def test_unexpected_character(self):
        """Test that unexpected characters raise LexerError."""
        with pytest.raises(LexerError) as exc_info:
            tokenize("mov @invalid")
        assert "Unexpected character" in str(exc_info.value)

    def test_error_location(self):
        """Test that error location is reported correctly."""
        with pytest.raises(LexerError) as exc_info:
            tokenize("line1\nline2 @error")
        assert exc_info.value.line == 2


class TestCompletePrograms:
    """Test tokenization of complete programs."""

    def test_simple_program(self):
        """Test tokenization of a simple program."""
        code = """.data
.text
.global main
main:
    movl $5, %eax
    int $0x80
"""
        tokens = tokenize(code)
        # Should not raise any errors
        assert any(t.type == TokenType.EOF for t in tokens)

    def test_program_with_data(self):
        """Test program with .data section."""
        code = """.data
    n: .long 10
.text
.global main
main:
    movl n, %ecx
"""
        tokens = tokenize(code)
        # Find the label definition
        for i, t in enumerate(tokens):
            if t.type == TokenType.IDENTIFIER and t.value == "n":
                if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.COLON:
                    break
        else:
            pytest.fail("Label 'n:' not found")

    def test_program_with_loop(self):
        """Test program with loop."""
        code = """et_loop:
    mul %ecx
    loop et_loop
"""
        tokens = tokenize(code)
        identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        assert "et_loop" in identifiers
        assert "mul" in identifiers
        assert "loop" in identifiers


class TestFilterTokens:
    """Test token filtering."""

    def test_filter_comments(self):
        """Test filtering out comments."""
        tokens = tokenize("movl $5, %eax # comment")
        filtered = Lexer.filter_tokens(tokens)
        assert not any(t.type == TokenType.COMMENT for t in filtered)

    def test_filter_multiple_types(self):
        """Test filtering multiple token types."""
        tokens = tokenize("movl $5, %eax\n# comment")
        filtered = Lexer.filter_tokens(tokens, [TokenType.COMMENT, TokenType.NEWLINE])
        assert not any(
            t.type in (TokenType.COMMENT, TokenType.NEWLINE) for t in filtered
        )


class TestTokenizeFile:
    """Test file tokenization."""

    def test_tokenize_sample_file(self, tmp_path):
        """Test tokenizing a file."""
        # Create a temporary file
        asm_file = tmp_path / "test.s"
        asm_file.write_text(".data\n.text\nmain:\n    movl $0, %eax")

        tokens = tokenize_file(str(asm_file))
        assert len(tokens) > 0
        assert tokens[-1].type == TokenType.EOF


# Integration tests with actual sample files
class TestSampleFiles:
    """Test tokenization of actual sample files from samples/in/."""

    @pytest.fixture
    def samples_dir(self):
        """Get the samples directory."""
        return Path(__file__).parent.parent / "samples" / "in"

    def test_tokenize_2_2(self, samples_dir):
        """Test tokenizing 2_2.s sample file."""
        sample_file = samples_dir / "2_2.s"
        if sample_file.exists():
            tokens = tokenize_file(str(sample_file))
            # Should complete without errors
            assert tokens[-1].type == TokenType.EOF
            # Should contain expected tokens
            identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
            assert "main" in identifiers
            assert "movl" in identifiers or "mov" in identifiers
            assert "cmp" in identifiers

    def test_tokenize_2_8_1(self, samples_dir):
        """Test tokenizing 2_8_1.s sample file (has loop)."""
        sample_file = samples_dir / "2_8_1.s"
        if sample_file.exists():
            tokens = tokenize_file(str(sample_file))
            assert tokens[-1].type == TokenType.EOF
            identifiers = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
            assert "loop" in identifiers
            assert "mul" in identifiers

    def test_tokenize_4_0(self, samples_dir):
        """Test tokenizing 4_0.s sample file (has memory addressing)."""
        sample_file = samples_dir / "4_0.s"
        if sample_file.exists():
            tokens = tokenize_file(str(sample_file))
            assert tokens[-1].type == TokenType.EOF
            # Should have LPAREN/RPAREN for memory addressing
            assert any(t.type == TokenType.LPAREN for t in tokens)
            assert any(t.type == TokenType.RPAREN for t in tokens)

    def test_tokenize_4_4(self, samples_dir):
        """Test tokenizing 4_4.s sample file (complex)."""
        sample_file = samples_dir / "4_4.s"
        if sample_file.exists():
            tokens = tokenize_file(str(sample_file))
            assert tokens[-1].type == TokenType.EOF
            # Should have strings
            assert any(t.type == TokenType.STRING for t in tokens)

    def test_tokenize_all_samples(self, samples_dir):
        """Test that all sample files tokenize without errors."""
        if samples_dir.exists():
            for sample_file in samples_dir.glob("*.s"):
                try:
                    tokens = tokenize_file(str(sample_file))
                    assert tokens[-1].type == TokenType.EOF, f"Failed for {sample_file}"
                except LexerError as e:
                    pytest.fail(f"Lexer error in {sample_file}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
