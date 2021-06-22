// RUN: %clang_cc1 -cl-std=CL2.0 -emit-llvm -o - -triple spir-unknown-unknown %s | FileCheck %s

typedef short short4 __attribute__((ext_vector_type(4)));

// CHECK-DAG: declare spir_func <4 x i16> @_Z5clampDv4_sS_S_(<4 x i16>, <4 x i16>, <4 x i16>)
short4 __attribute__ ((overloadable)) clamp(short4 x, short4 minval, short4 maxval);
// CHECK-DAG: declare spir_func <4 x i16> @_Z5clampDv4_sss(<4 x i16>, i16 signext, i16 signext)
short4 __attribute__ ((overloadable)) clamp(short4 x, short minval, short maxval);
void __attribute__((overloadable)) foo(global int *a, global int *b);
void __attribute__((overloadable)) foo(generic int *a, generic int *b);
void __attribute__((overloadable)) bar(generic int *global *a, generic int *global *b);
void __attribute__((overloadable)) bar(generic int *generic *a, generic int *generic *b);

// Checking address space resolution
void kernel test1() {
  global int *a;
  global int *b;
  generic int *c;
  local int *d;
  generic int *generic *gengen;
  generic int *local *genloc;
  generic int *global *genglob;
  // CHECK-DAG: call spir_func void @_Z3fooPU3AS1iS0_(i32 addrspace(1)* undef, i32 addrspace(1)* undef)
  foo(a, b);
  // CHECK-DAG: call spir_func void @_Z3fooPU3AS4iS0_(i32 addrspace(4)* undef, i32 addrspace(4)* undef)
  foo(b, c);
  // CHECK-DAG: call spir_func void @_Z3fooPU3AS4iS0_(i32 addrspace(4)* undef, i32 addrspace(4)* undef)
  foo(a, d);

  // CHECK-DAG: call spir_func void @_Z3barPU3AS4PU3AS4iS2_(i32 addrspace(4)* addrspace(4)* undef, i32 addrspace(4)* addrspace(4)* undef)
  bar(gengen, genloc);
  // CHECK-DAG: call spir_func void @_Z3barPU3AS4PU3AS4iS2_(i32 addrspace(4)* addrspace(4)* undef, i32 addrspace(4)* addrspace(4)* undef)
  bar(gengen, genglob);
  // CHECK-DAG: call spir_func void @_Z3barPU3AS1PU3AS4iS2_(i32 addrspace(4)* addrspace(1)* undef, i32 addrspace(4)* addrspace(1)* undef)
  bar(genglob, genglob);
}

// Checking vector vs scalar resolution
void kernel test2() {
  short4 e0=0;

  // CHECK-DAG: call spir_func <4 x i16> @_Z5clampDv4_sss(<4 x i16> zeroinitializer, i16 signext 0, i16 signext 255)
  clamp(e0, 0, 255);
  // CHECK-DAG: call spir_func <4 x i16> @_Z5clampDv4_sS_S_(<4 x i16> zeroinitializer, <4 x i16> zeroinitializer, <4 x i16> zeroinitializer)
  clamp(e0, e0, e0);
}
