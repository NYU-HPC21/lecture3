int main() {
  int a = 5;
  int b = 1;
  int c = a*a + b;
  return c;
}

// g++ disassemble-simple.cpp -c && objdump --disassemble disassemble-simple.o
//
// disassemble-simple.o:     file format mach-o-x86-64
//
//
// Disassembly of section .text:
//
// 0000000000000000 <_main>:
//    0:   55                      push   %rbp
//    1:   48 89 e5                mov    %rsp,%rbp
//    4:   c7 45 fc 05 00 00 00    movl   $0x5,-0x4(%rbp)
//    b:   c7 45 f8 01 00 00 00    movl   $0x1,-0x8(%rbp)
//   12:   8b 45 fc                mov    -0x4(%rbp),%eax
//   15:   0f af 45 fc             imul   -0x4(%rbp),%eax
//   19:   8b 55 f8                mov    -0x8(%rbp),%edx
//   1c:   01 d0                   add    %edx,%eax
//   1e:   89 45 f4                mov    %eax,-0xc(%rbp)
//   21:   8b 45 f4                mov    -0xc(%rbp),%eax
//   24:   5d                      pop    %rbp
//   25:   c3                      retq
//
//
//
//
// http://gcc.godbolt.org
