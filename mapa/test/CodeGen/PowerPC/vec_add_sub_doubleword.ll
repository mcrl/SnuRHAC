; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s --check-prefixes=ALL,VSX
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s --check-prefixes=ALL,NOVSX

; Check VMX 64-bit integer operations

define <2 x i64> @test_add(<2 x i64> %x, <2 x i64> %y) nounwind {
; ALL-LABEL: test_add:
; ALL:       # %bb.0:
; ALL-NEXT:    vaddudm 2, 2, 3
; ALL-NEXT:    blr
  %result = add <2 x i64> %x, %y
  ret <2 x i64> %result
}

define <2 x i64> @increment_by_one(<2 x i64> %x) nounwind {
; VSX-LABEL: increment_by_one:
; VSX:       # %bb.0:
; VSX-NEXT:    addis 3, 2, .LCPI1_0@toc@ha
; VSX-NEXT:    addi 3, 3, .LCPI1_0@toc@l
; VSX-NEXT:    lxvd2x 35, 0, 3
; VSX-NEXT:    vaddudm 2, 2, 3
; VSX-NEXT:    blr
;
; NOVSX-LABEL: increment_by_one:
; NOVSX:       # %bb.0:
; NOVSX-NEXT:    addis 3, 2, .LCPI1_0@toc@ha
; NOVSX-NEXT:    addi 3, 3, .LCPI1_0@toc@l
; NOVSX-NEXT:    lvx 3, 0, 3
; NOVSX-NEXT:    vaddudm 2, 2, 3
; NOVSX-NEXT:    blr
  %result = add <2 x i64> %x, <i64 1, i64 1>
  ret <2 x i64> %result
}

define <2 x i64> @increment_by_val(<2 x i64> %x, i64 %val) nounwind {
; VSX-LABEL: increment_by_val:
; VSX:       # %bb.0:
; VSX-NEXT:    mtvsrd 0, 5
; VSX-NEXT:    xxspltd 35, 0, 0
; VSX-NEXT:    vaddudm 2, 2, 3
; VSX-NEXT:    blr
;
; NOVSX-LABEL: increment_by_val:
; NOVSX:       # %bb.0:
; NOVSX-NEXT:    addi 3, 1, -16
; NOVSX-NEXT:    std 5, -8(1)
; NOVSX-NEXT:    std 5, -16(1)
; NOVSX-NEXT:    lvx 3, 0, 3
; NOVSX-NEXT:    vaddudm 2, 2, 3
; NOVSX-NEXT:    blr
  %tmpvec = insertelement <2 x i64> <i64 0, i64 0>, i64 %val, i32 0
  %tmpvec2 = insertelement <2 x i64> %tmpvec, i64 %val, i32 1
  %result = add <2 x i64> %x, %tmpvec2
  ret <2 x i64> %result
; FIXME: This is currently generating the following instruction sequence
;   std 5, -8(1)
;   std 5, -16(1)
;   addi 3, 1, -16
;   ori 2, 2, 0
;   lxvd2x 35, 0, 3
;   vaddudm 2, 2, 3
;   blr
;   This will almost certainly cause a load-hit-store hazard.
;   Since val is a value parameter, it should not need to be
;   saved onto the stack at all (unless we're using this to set
;   up the vector register). Instead, it would be better to splat
;   the value into a vector register.
}

define <2 x i64> @test_sub(<2 x i64> %x, <2 x i64> %y) nounwind {
; ALL-LABEL: test_sub:
; ALL:       # %bb.0:
; ALL-NEXT:    vsubudm 2, 2, 3
; ALL-NEXT:    blr
  %result = sub <2 x i64> %x, %y
  ret <2 x i64> %result
}

define <2 x i64> @decrement_by_one(<2 x i64> %x) nounwind {
; VSX-LABEL: decrement_by_one:
; VSX:       # %bb.0:
; VSX-NEXT:    xxleqv 35, 35, 35
; VSX-NEXT:    vsubudm 2, 2, 3
; VSX-NEXT:    blr
;
; NOVSX-LABEL: decrement_by_one:
; NOVSX:       # %bb.0:
; NOVSX-NEXT:    addis 3, 2, .LCPI4_0@toc@ha
; NOVSX-NEXT:    addi 3, 3, .LCPI4_0@toc@l
; NOVSX-NEXT:    lvx 3, 0, 3
; NOVSX-NEXT:    vsubudm 2, 2, 3
; NOVSX-NEXT:    blr
  %result = sub <2 x i64> %x, <i64 -1, i64 -1>
  ret <2 x i64> %result
}

define <2 x i64> @decrement_by_val(<2 x i64> %x, i64 %val) nounwind {
; VSX-LABEL: decrement_by_val:
; VSX:       # %bb.0:
; VSX-NEXT:    mtvsrd 0, 5
; VSX-NEXT:    xxspltd 35, 0, 0
; VSX-NEXT:    vsubudm 2, 2, 3
; VSX-NEXT:    blr
;
; NOVSX-LABEL: decrement_by_val:
; NOVSX:       # %bb.0:
; NOVSX-NEXT:    addi 3, 1, -16
; NOVSX-NEXT:    std 5, -8(1)
; NOVSX-NEXT:    std 5, -16(1)
; NOVSX-NEXT:    lvx 3, 0, 3
; NOVSX-NEXT:    vsubudm 2, 2, 3
; NOVSX-NEXT:    blr
  %tmpvec = insertelement <2 x i64> <i64 0, i64 0>, i64 %val, i32 0
  %tmpvec2 = insertelement <2 x i64> %tmpvec, i64 %val, i32 1
  %result = sub <2 x i64> %x, %tmpvec2
  ret <2 x i64> %result
}