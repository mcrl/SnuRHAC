; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i686-- -x86-asm-syntax=intel | FileCheck %s

@G1 = internal global i8 0              ; <i8*> [#uses=1]
@G2 = internal global i8 0              ; <i8*> [#uses=1]

define i16 @test1() {
; CHECK-LABEL: test1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movzx eax, byte ptr [G1]
; CHECK-NEXT:    # kill: def $ax killed $ax killed $eax
; CHECK-NEXT:    ret
        %tmp.0 = load i8, i8* @G1           ; <i8> [#uses=1]
        %tmp.3 = zext i8 %tmp.0 to i16          ; <i16> [#uses=1]
        ret i16 %tmp.3
}

define i16 @test2() {
; CHECK-LABEL: test2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movsx eax, byte ptr [G2]
; CHECK-NEXT:    # kill: def $ax killed $ax killed $eax
; CHECK-NEXT:    ret
        %tmp.0 = load i8, i8* @G2           ; <i8> [#uses=1]
        %tmp.3 = sext i8 %tmp.0 to i16          ; <i16> [#uses=1]
        ret i16 %tmp.3
}

