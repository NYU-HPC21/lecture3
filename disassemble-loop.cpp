int main() {
  int sum = 0;
  for (int i = 0; i < 10; ++i) {
    sum += i;
  }
  return sum;
}

// $ g++ -O0 disassemble-loop.cpp -c && objdump --disassemble disassemble-loop.o
//
// disassemble-loop.o:     file format mach-o-x86-64
//
//
// Disassembly of section .text:
//
// 0000000000000000 <_main>:
//    0:   55                      push   %rbp
//    1:   48 89 e5                mov    %rsp,%rbp
//    4:   c7 45 fc 00 00 00 00    movl   $0x0,-0x4(%rbp)
//    b:   c7 45 f8 00 00 00 00    movl   $0x0,-0x8(%rbp)
//   12:   83 7d f8 09             cmpl   $0x9,-0x8(%rbp)
//   16:   7f 0c                   jg     24 <_main+0x24>
//   18:   8b 45 f8                mov    -0x8(%rbp),%eax
//   1b:   01 45 fc                add    %eax,-0x4(%rbp)
//   1e:   83 45 f8 01             addl   $0x1,-0x8(%rbp)
//   22:   eb ee                   jmp    12 <_main+0x12>
//   24:   8b 45 fc                mov    -0x4(%rbp),%eax
//   27:   5d                      pop    %rbp
//   28:   c3                      retq
//
//
//
//
// http://gcc.godbolt.org
