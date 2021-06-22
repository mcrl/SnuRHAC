/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2021 Seoul National University.                             */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   THUNDER Research Group                                                  */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 08826, Korea                           */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jaehoon Jung, Daeyoung Park, Gangwon Jo, Jungho Park, and Jaejin Lee    */
/*                                                                           */
/*****************************************************************************/

#include "clang/SnuPreprocess/ASTDuplicator.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuPreprocess/ASTBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

namespace snu {

Stmt *ASTDuplicator::VisitFunctionBody(Stmt *Node, VarDecl *Ret) {
  ConvertReturnToExpr = true;
  ReturnVar = Ret;
  ReturnLabel = NULL;

  CompoundStmt *CS = dyn_cast<CompoundStmt>(Node);
  if (!CS) {
    CS = CompoundStmt::Create(
        ASTCtx, Node, Node->getBeginLoc(), Node->getEndLoc());
  }

  SmallVector<Stmt *, 16> NewStmts;
  VisitCompoundStmtBody(CS, true, NewStmts);
  if (ReturnLabel) {
    // RetLabel: ;
    Stmt *NS = new (ASTCtx) NullStmt(SourceLocation());
    Stmt *LS = new (ASTCtx) LabelStmt(SourceLocation(), ReturnLabel, NS);
    NewStmts.push_back(LS);
  }

  ConvertReturnToExpr = false;
  ReturnVar = NULL;
  ReturnLabel = NULL;

  return CompoundStmt::Create(
      ASTCtx, NewStmts, Node->getBeginLoc(), Node->getEndLoc());
}

// Statements

Stmt *ASTDuplicator::VisitNullStmt(NullStmt *Node) {
  return new (ASTCtx) NullStmt(Node->getSemiLoc(), Node->hasLeadingEmptyMacro());
}

Stmt *ASTDuplicator::VisitCompoundStmt(CompoundStmt *Node) {
  SmallVector<Stmt *, 16> NewStmts;
  VisitCompoundStmtBody(Node, false, NewStmts);
  return CompoundStmt::Create(
      ASTCtx, NewStmts, Node->getBeginLoc(), Node->getEndLoc());
}

void ASTDuplicator::VisitCompoundStmtBody(CompoundStmt *CS, bool Outermost,
                                          SmallVectorImpl<Stmt *> &NewStmts) {
  for (CompoundStmt::body_iterator S = CS->body_begin(), SEnd = CS->body_end();
       S != SEnd; ++S) {
    NewStmts.push_back(Visit(*S));

    // return E; => RetVal = E; goto RetLabel;
    //                          --------------
    if (ConvertReturnToExpr && !(Outermost && S + 1 == SEnd)) {
      if (isa<ReturnStmt>((*S)->stripLabelLikeStatements())) {
        GotoStmt *GS = new (ASTCtx) GotoStmt(
          LookupOrCreateReturnLabelDecl(), SourceLocation(), SourceLocation());
        NewStmts.push_back(GS);
      }
    }
  }
}

Stmt *ASTDuplicator::VisitLabelStmt(LabelStmt *Node) {
  LabelDecl *NewLabel = LookupOrDuplicateLabelDecl(Node->getDecl(), Node->getBeginLoc());
  Stmt *NewSubStmt = Visit(Node->getSubStmt());
  return new (ASTCtx) LabelStmt(Node->getBeginLoc(), NewLabel, NewSubStmt);
}

Stmt *ASTDuplicator::VisitAttributedStmt(AttributedStmt *Node) {
  ArrayRef<const Attr *> Attrs = Node->getAttrs();
  SmallVector<const Attr *, 16> NewAttrs;
  for (ArrayRef<const Attr *>::iterator A = Attrs.begin(), AEnd = Attrs.end();
       A != AEnd; ++A) {
    NewAttrs.push_back((*A)->clone(ASTCtx));
  }
  Stmt *NewSubStmt = Visit(Node->getSubStmt());
  return AttributedStmt::Create(ASTCtx, Node->getAttrLoc(), NewAttrs, NewSubStmt);
}

Stmt *ASTDuplicator::VisitIfStmt(IfStmt *Node) {
  VarDecl *NewCondVar = NULL;
  if (Node->getConditionVariable()) {
    NewCondVar = DuplicateVarDecl(Node->getConditionVariable());
  }
  Stmt *NewInit = NULL;
  if (Node->getInit()) {
    NewInit = VisitBlock(Node->getInit());
  }
  Expr *NewCond = VisitExpr(Node->getCond());
  Stmt *NewThen = VisitBlock(Node->getThen());
  Stmt *NewElse = NULL;
  if (Node->getElse()) {
    NewElse = VisitBlock(Node->getElse());
  }
  return IfStmt::Create(
      ASTCtx, Node->getIfLoc(), Node->isConstexpr(), NewInit,
      NewCondVar, NewCond, NewThen, Node->getElseLoc(), NewElse);
}

Stmt *ASTDuplicator::VisitSwitchStmt(SwitchStmt *Node) {
  Stmt *NewInit = NULL;
  if (Node->getInit()) {
    NewInit = VisitBlock(Node->getInit());
  }
  VarDecl *NewCondVar = NULL;
  if (Node->getConditionVariable()) {
    NewCondVar = DuplicateVarDecl(Node->getConditionVariable());
  }
  Expr *NewCond = VisitExpr(Node->getCond());
  SwitchStmt *NewNode = SwitchStmt::Create(
      ASTCtx, NewInit, NewCondVar, NewCond);

  SwitchStmt *PreviousSwitchStmt = CurrentSwitchStmt;
  CurrentSwitchStmt = NewNode;
  NewNode->setBody(VisitBlock(Node->getBody()));
  CurrentSwitchStmt = PreviousSwitchStmt;

  return NewNode;
}

Stmt *ASTDuplicator::VisitWhileStmt(WhileStmt *Node) {
  VarDecl *NewCondVar = NULL;
  if (Node->getConditionVariable()) {
    NewCondVar = DuplicateVarDecl(Node->getConditionVariable());
  }
  Expr *NewCond = VisitExpr(Node->getCond());
  Stmt *NewBody = VisitBlock(Node->getBody());
  return WhileStmt::Create(
    ASTCtx, NewCondVar, NewCond, NewBody,
    Node->getWhileLoc());
}

Stmt *ASTDuplicator::VisitDoStmt(DoStmt *Node) {
  Stmt *NewBody = VisitBlock(Node->getBody());
  Expr *NewCond = VisitExpr(Node->getCond());
  return new (ASTCtx) DoStmt(
    NewBody, NewCond,
    Node->getDoLoc(), Node->getWhileLoc(), Node->getRParenLoc());
}

Stmt *ASTDuplicator::VisitForStmt(ForStmt *Node) {
  Stmt *NewInit = NULL;
  if (Node->getInit()) {
    NewInit = Visit(Node->getInit());
  }
  VarDecl *NewCondVar = NULL;
  if (Node->getConditionVariable()) {
    NewCondVar = DuplicateVarDecl(Node->getConditionVariable());
  }
  Expr *NewCond = NULL;
  if (Node->getCond()) {
    NewCond = VisitExpr(Node->getCond());
  }
  Expr *NewInc = NULL;
  if (Node->getInc()) {
    NewInc = VisitExpr(Node->getInc());
  }
  Stmt *NewBody = VisitBlock(Node->getBody());
  return new (ASTCtx) ForStmt(
    ASTCtx, NewInit, NewCond, NewCondVar, NewInc, NewBody,
    Node->getForLoc(), Node->getLParenLoc(), Node->getRParenLoc());
}

Stmt *ASTDuplicator::VisitGotoStmt(GotoStmt *Node) {
  LabelDecl *NewLabel = LookupOrDuplicateLabelDecl(Node->getLabel(), Node->getLabelLoc());
  return new (ASTCtx) GotoStmt(NewLabel, Node->getGotoLoc(), Node->getLabelLoc());
}

Stmt *ASTDuplicator::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Expr *NewTarget = VisitExpr(Node->getTarget());
  return new (ASTCtx) IndirectGotoStmt(Node->getGotoLoc(), Node->getStarLoc(), NewTarget);
}

Stmt *ASTDuplicator::VisitContinueStmt(ContinueStmt *Node) {
  return new (ASTCtx) ContinueStmt(Node->getContinueLoc());
}

Stmt *ASTDuplicator::VisitBreakStmt(BreakStmt *Node) {
  return new (ASTCtx) BreakStmt(Node->getBreakLoc());
}

Stmt *ASTDuplicator::VisitReturnStmt(ReturnStmt *Node) {
  Expr *NewRetValue = NULL;
  if (Node->getRetValue()) {
    NewRetValue = VisitExpr(Node->getRetValue());
  }

  if (ConvertReturnToExpr) {
    if (ReturnVar && NewRetValue) {
      // return E; => RetVar = E;
      return Builder.GetAssignToVarExpr(ReturnVar, NewRetValue);
    } else if (NewRetValue) {
      // (in a void function) return E; => E;
      return NewRetValue;
    } else {
      // return; => ;
      return new (ASTCtx) NullStmt(Node->getReturnLoc());
    }
  } else {
    if (NewRetValue) {
      return ReturnStmt::Create(
          ASTCtx, Node->getReturnLoc(), NewRetValue, NULL);
    } else {
      return ReturnStmt::Create(
          ASTCtx, Node->getReturnLoc(), NULL, NULL);
    }
  }
}

Stmt *ASTDuplicator::VisitDeclStmt(DeclStmt *Node) {
  SmallVector<Decl *, 16> NewDecls;
  for (DeclStmt::decl_iterator D = Node->decl_begin(), DEnd = Node->decl_end();
       D != DEnd; ++D) {
    if (VarDecl *VD = dyn_cast<VarDecl>(*D)) {
      NewDecls.push_back(DuplicateVarDecl(VD));
    } else if (BlockDecl *BD = dyn_cast<BlockDecl>(*D)) {
      // TODO: duplicate a block literal
      NewDecls.push_back(BD);
    } else if (LabelDecl *LD = dyn_cast<LabelDecl>(*D)) {
      // Ignore a declaration of a GNU local label.
      LookupOrDuplicateLabelDecl(LD, SourceLocation());
    } else if (EmptyDecl *ED = dyn_cast<EmptyDecl>(*D)) {
      NewDecls.push_back(EmptyDecl::Create(ASTCtx, &DeclCtx, ED->getLocation()));
    } else {
      // TypedefDecl, EnumDecl, and RecordDecl are not duplicated.
      NewDecls.push_back(*D);
    } 
  }
  if (NewDecls.empty()) {
    return new (ASTCtx) NullStmt(Node->getEndLoc(), false);
  }
  DeclGroupRef DG = DeclGroupRef::Create(ASTCtx, NewDecls.data(), NewDecls.size());
  return new (ASTCtx) DeclStmt(DG, Node->getBeginLoc(), Node->getEndLoc());
}

Stmt *ASTDuplicator::VisitCaseStmt(CaseStmt *Node) {
  Expr *NewLHS = VisitExpr(Node->getLHS());
  Expr *NewRHS = NULL;
  if (Node->getRHS()) {
    NewRHS = VisitExpr(Node->getRHS());
  }
  CaseStmt *NewNode = CaseStmt::Create(
      ASTCtx, NewLHS, NewRHS,
      Node->getCaseLoc(), Node->getEllipsisLoc(), Node->getColonLoc());
  if (CurrentSwitchStmt) {
    CurrentSwitchStmt->addSwitchCase(NewNode);
  }

  Stmt *NewSubStmt = Visit(Node->getSubStmt());
  NewNode->setSubStmt(NewSubStmt);

  return NewNode;
}

Stmt *ASTDuplicator::VisitDefaultStmt(DefaultStmt *Node) {
  Stmt *NewSubStmt = Visit(Node->getSubStmt());
  DefaultStmt *NewNode = new (ASTCtx) DefaultStmt(
    Node->getDefaultLoc(), Node->getColonLoc(), NewSubStmt);
  if (CurrentSwitchStmt) {
    CurrentSwitchStmt->addSwitchCase(NewNode);
  }
  return NewNode;
}

Stmt *ASTDuplicator::VisitCapturedStmt(CapturedStmt *Node) {
  SmallVector<CapturedStmt::Capture, 16> NewCaptures;
  for (CapturedStmt::capture_iterator C = Node->capture_begin(), CEnd = Node->capture_end();
       C != CEnd; ++C) {
    VarDecl *NewVarDecl = NULL;
    if (C->capturesVariable()) {
      NewVarDecl = LookupVarDecl(C->getCapturedVar());
    }
    NewCaptures.push_back(CapturedStmt::Capture(
      C->getLocation(), C->getCaptureKind(), NewVarDecl));
  }

  SmallVector<Expr *, 16> NewCaptureInits;
  for (CapturedStmt::capture_init_iterator E = Node->capture_init_begin(), EEnd = Node->capture_init_end();
       E != EEnd; ++E) {
    NewCaptureInits.push_back(VisitExpr(*E));
  }

  // CapturedDecl and RecordDecl are not duplicated.
  CapturedDecl *NewCapturedDecl = Node->getCapturedDecl();
  RecordDecl *NewCapturedRecordDecl = const_cast<RecordDecl *>(Node->getCapturedRecordDecl());

  Stmt *NewCapturedStmt = VisitBlock(Node->getCapturedStmt());
  return CapturedStmt::Create(
    ASTCtx, NewCapturedStmt, Node->getCapturedRegionKind(),
    NewCaptures, NewCaptureInits, NewCapturedDecl, NewCapturedRecordDecl);
}

Stmt *ASTDuplicator::VisitBlock(Stmt *Node) {
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node)) {
    return VisitCompoundStmt(CS);
  }

  if (isa<ReturnStmt>(Node->stripLabelLikeStatements())) {
    SmallVector<Stmt *, 2> NewStmts;
    NewStmts.push_back(Visit(Node));
    NewStmts.push_back(new (ASTCtx) GotoStmt(
      LookupOrCreateReturnLabelDecl(), SourceLocation(), SourceLocation()));
    return CompoundStmt::Create(ASTCtx, NewStmts, Node->getBeginLoc(), Node->getEndLoc());
  }

  return Visit(Node);
}

// Asm statements

Stmt *ASTDuplicator::VisitGCCAsmStmt(GCCAsmStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitMSAsmStmt(MSAsmStmt *Node) { return Node; }

// Obj-C statements

Stmt *ASTDuplicator::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) { return Node;}
Stmt *ASTDuplicator::VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCForCollectionStmt(ObjCForCollectionStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCAutoreleasePoolStmt(ObjCAutoreleasePoolStmt *Node) { return Node; }

// C++ statements

Stmt *ASTDuplicator::VisitCXXCatchStmt(CXXCatchStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXTryStmt(CXXTryStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXForRangeStmt(CXXForRangeStmt *Node) { return Node; }

// C++ Coroutines TS statements

Stmt *ASTDuplicator::VisitCoroutineBodyStmt(CoroutineBodyStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitCoreturnStmt(CoreturnStmt *Node) { return Node; }

// Expressions

Stmt *ASTDuplicator::VisitPredefinedExpr(PredefinedExpr *Node) {
  return PredefinedExpr::Create(
      ASTCtx, Node->getLocation(), Node->getType(), Node->getIdentKind(),
      Node->getFunctionName());
}

Stmt *ASTDuplicator::VisitDeclRefExpr(DeclRefExpr *Node) {
  return DeclRefExpr::Create(
    ASTCtx, Node->getQualifierLoc(), SourceLocation(),
    reinterpret_cast<ValueDecl*>(LookupDecl(Node->getDecl())),
    Node->refersToEnclosingVariableOrCapture(),
    Node->getNameInfo(), Node->getType(), Node->getValueKind(),
    (Node->getDecl() == Node->getFoundDecl() ? NULL : LookupDecl(Node->getFoundDecl())),
    NULL);
}

Stmt *ASTDuplicator::VisitIntegerLiteral(IntegerLiteral *Node) {
  return new (ASTCtx) IntegerLiteral(
    ASTCtx, Node->getValue(), Node->getType(), Node->getLocation());
}

Stmt *ASTDuplicator::VisitFixedPointLiteral(FixedPointLiteral *Node) {
  assert(0);
  return Node;
}

Stmt *ASTDuplicator::VisitFloatingLiteral(FloatingLiteral *Node) {
  return FloatingLiteral::Create(
    ASTCtx, Node->getValue(), Node->isExact(), Node->getType(),
    Node->getLocation());
}

Stmt *ASTDuplicator::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return new (ASTCtx) ImaginaryLiteral(NewSubExpr, Node->getType());
}

Stmt *ASTDuplicator::VisitStringLiteral(StringLiteral *Node) {
  return StringLiteral::Create(
    ASTCtx, Node->getString(), Node->getKind(), Node->isPascal(),
    Node->getType(), Node->tokloc_begin(), Node->getNumConcatenated());
}

Stmt *ASTDuplicator::VisitCharacterLiteral(CharacterLiteral *Node) {
  return new (ASTCtx) CharacterLiteral(
    Node->getValue(), Node->getKind(), Node->getType(), Node->getLocation());
}

Stmt *ASTDuplicator::VisitParenExpr(ParenExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return new (ASTCtx) ParenExpr(Node->getLParen(), Node->getRParen(), NewSubExpr);
}

Stmt *ASTDuplicator::VisitUnaryOperator(UnaryOperator *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return new (ASTCtx) UnaryOperator(
    NewSubExpr, Node->getOpcode(), Node->getType(), Node->getValueKind(),
    Node->getObjectKind(), Node->getOperatorLoc(), Node->canOverflow());
}

Stmt *ASTDuplicator::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  SmallVector<OffsetOfNode, 16> NewComps;
  for (unsigned Index = 0, NumComps = Node->getNumComponents();
       Index != NumComps; ++Index) {
    NewComps.push_back(Node->getComponent(Index));
  }

  SmallVector<Expr *, 16> NewExprs;
  for (unsigned Index = 0, NumExprs = Node->getNumExpressions();
       Index != NumExprs; ++Index) {
    NewExprs.push_back(VisitExpr(Node->getIndexExpr(Index)));
  }

  return OffsetOfExpr::Create(
    ASTCtx, Node->getType(), Node->getOperatorLoc(), Node->getTypeSourceInfo(),
    NewComps, NewExprs, Node->getRParenLoc());
}

Stmt *ASTDuplicator::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) {
  if (Node->isArgumentType()) {
    return new (ASTCtx) UnaryExprOrTypeTraitExpr(
      Node->getKind(), Node->getArgumentTypeInfo(), Node->getType(),
      Node->getOperatorLoc(), Node->getRParenLoc());
  } else {
    Expr *NewArgumentExpr = VisitExpr(Node->getArgumentExpr());
    return new (ASTCtx) UnaryExprOrTypeTraitExpr(
      Node->getKind(), NewArgumentExpr, Node->getType(),
      Node->getOperatorLoc(), Node->getRParenLoc());
  }
}

Stmt *ASTDuplicator::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Expr *NewLHS = VisitExpr(Node->getLHS());
  Expr *NewRHS = VisitExpr(Node->getRHS());
  return new (ASTCtx) ArraySubscriptExpr(
    NewLHS, NewRHS, Node->getType(),
    Node->getValueKind(), Node->getObjectKind(), Node->getRBracketLoc());
}

Stmt *ASTDuplicator::VisitCallExpr(CallExpr *Node) {
  Expr *NewCallee = VisitExpr(Node->getCallee());
  SmallVector<Expr *, 16> NewArgs;
  for (CallExpr::arg_iterator A = Node->arg_begin(), AEnd = Node->arg_end();
       A != AEnd; ++A) {
    NewArgs.push_back(VisitExpr(*A));
  }
  return CallExpr::Create(
    ASTCtx, NewCallee, NewArgs, Node->getType(), Node->getValueKind(),
    Node->getRParenLoc(), 0, Node->getADLCallKind());
}

Stmt *ASTDuplicator::VisitMemberExpr(MemberExpr *Node) {
  Expr *NewBase = VisitExpr(Node->getBase());
  return MemberExpr::Create(
    ASTCtx, NewBase, Node->isArrow(), Node->getOperatorLoc(),
    Node->getQualifierLoc(), SourceLocation(), Node->getMemberDecl(),
    Node->getFoundDecl(), Node->getMemberNameInfo(), NULL, Node->getType(),
    Node->getValueKind(), Node->getObjectKind(), Node->isNonOdrUse());
}

Stmt *ASTDuplicator::VisitBinaryOperator(BinaryOperator *Node) {
  Expr *NewLHS = VisitExpr(Node->getLHS());
  Expr *NewRHS = VisitExpr(Node->getRHS());
  return new (ASTCtx) BinaryOperator(
    NewLHS, NewRHS, Node->getOpcode(), Node->getType(),
    Node->getValueKind(), Node->getObjectKind(), Node->getOperatorLoc(),
    Node->getFPFeatures());
}

Stmt *ASTDuplicator::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Expr *NewLHS = VisitExpr(Node->getLHS());
  Expr *NewRHS = VisitExpr(Node->getRHS());
  return new (ASTCtx) CompoundAssignOperator(
    NewLHS, NewRHS, Node->getOpcode(), Node->getType(),
    Node->getValueKind(), Node->getObjectKind(),
    Node->getComputationLHSType(), Node->getComputationResultType(),
    Node->getOperatorLoc(), Node->getFPFeatures());
}

Stmt *ASTDuplicator::VisitConditionalOperator(ConditionalOperator *Node) {
  Expr *NewCond = VisitExpr(Node->getCond());
  Expr *NewTrue = VisitExpr(Node->getTrueExpr());
  Expr *NewFalse = VisitExpr(Node->getFalseExpr());
  return new (ASTCtx) ConditionalOperator(
    NewCond, Node->getQuestionLoc(), NewTrue, Node->getColonLoc(), NewFalse,
    Node->getType(), Node->getValueKind(), Node->getObjectKind());
}

Stmt *ASTDuplicator::VisitBinaryConditionalOperator(BinaryConditionalOperator *Node) {
  Expr *NewCommon = VisitExpr(Node->getCommon());
  OpaqueValueExpr *NewOpaqueValue = new (ASTCtx) OpaqueValueExpr(
    NewCommon->getExprLoc(), NewCommon->getType(),
    NewCommon->getValueKind(), NewCommon->getObjectKind(), NewCommon);
  Expr *NewFalse = VisitExpr(Node->getFalseExpr());

  // OpaqueValue, Cond, and TrueExpr are the same.
  // See Sema::ActOnConditionalOp @ lib/Sema/SemaExpr.cpp.
  return new (ASTCtx) BinaryConditionalOperator(
    NewCommon, NewOpaqueValue, NewOpaqueValue, NewOpaqueValue, NewFalse,
    Node->getQuestionLoc(), Node->getColonLoc(), Node->getType(),
    Node->getValueKind(), Node->getObjectKind());
}

Stmt *ASTDuplicator::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return ImplicitCastExpr::Create(
    ASTCtx, Node->getType(), Node->getCastKind(), NewSubExpr, NULL,
    Node->getValueKind());
}

Stmt *ASTDuplicator::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return CStyleCastExpr::Create(
    ASTCtx, Node->getType(), Node->getValueKind(), Node->getCastKind(),
    NewSubExpr, NULL, Node->getTypeInfoAsWritten(),
    Node->getLParenLoc(), Node->getRParenLoc());
}

Stmt *ASTDuplicator::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Expr *NewInitializer = VisitExpr(Node->getInitializer());
  return new (ASTCtx) CompoundLiteralExpr(
    Node->getLParenLoc(), Node->getTypeSourceInfo(), Node->getType(),
    Node->getValueKind(), NewInitializer, Node->isFileScope());
}

Stmt *ASTDuplicator::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Expr *NewBase = VisitExpr(Node->getBase());
  return new (ASTCtx) ExtVectorElementExpr(
    Node->getType(), Node->getValueKind(), NewBase, Node->getAccessor(),
    Node->getAccessorLoc());
}

Stmt *ASTDuplicator::VisitInitListExpr(InitListExpr *Node) {
  SmallVector<Expr *, 16> NewInits;
  for (unsigned Index = 0, NumInits = Node->getNumInits();
       Index != NumInits; ++Index) {
    if (Node->getInit(Index)) {
      NewInits.push_back(VisitExpr(Node->getInit(Index)));
    } else {
      NewInits.push_back(NULL);
    }
  }
  InitListExpr *NewExpr = new (ASTCtx) InitListExpr(
    ASTCtx, Node->getLBraceLoc(), NewInits, Node->getRBraceLoc());
  NewExpr->setType(Node->getType());
  // Discard the alternative form
  return NewExpr;
}

Stmt *ASTDuplicator::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  SmallVector<Expr *, 16> NewIndexExprs;
  // the first subexpression is the initializer
  for (unsigned Index = 1, NumSubExprs = Node->getNumSubExprs();
       Index != NumSubExprs; ++Index) {
    NewIndexExprs.push_back(VisitExpr(Node->getSubExpr(Index)));
  }
  Expr *NewInit = VisitExpr(Node->getInit());
  return DesignatedInitExpr::Create(
    ASTCtx, Node->designators(), NewIndexExprs,
    Node->getEqualOrColonLoc(), Node->usesGNUSyntax(), NewInit);
}

Stmt *ASTDuplicator::VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *Node) {
  assert(0);
  return Node;
}

Stmt *ASTDuplicator::VisitImplicitValueInitExpr(ImplicitValueInitExpr *Node) {
  return new (ASTCtx) ImplicitValueInitExpr(Node->getType());
}

Stmt *ASTDuplicator::VisitNoInitExpr(NoInitExpr *Node) {
  return new (ASTCtx) NoInitExpr(Node->getType());
}

Stmt *ASTDuplicator::VisitArrayInitLoopExpr(ArrayInitLoopExpr *Node) {
  assert(0);
  return Node;
}

Stmt *ASTDuplicator::VisitArrayInitIndexExpr(ArrayInitIndexExpr *Node) {
  assert(0);
  return Node;
}

Stmt *ASTDuplicator::VisitParenListExpr(ParenListExpr *Node) {
  SmallVector<Expr *, 16> NewExprs;
  for (unsigned Index = 0, NumExprs = Node->getNumExprs();
       Index != NumExprs; ++Index) {
    if (Node->getExpr(Index)) {
      NewExprs.push_back(VisitExpr(Node->getExpr(Index)));
    } else {
      NewExprs.push_back(NULL);
    }
  }
  return ParenListExpr::Create(
    ASTCtx, Node->getLParenLoc(), NewExprs, Node->getRParenLoc());
}

Stmt *ASTDuplicator::VisitVAArgExpr(VAArgExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return new (ASTCtx) VAArgExpr(
    Node->getBuiltinLoc(), NewSubExpr, Node->getWrittenTypeInfo(),
    Node->getRParenLoc(), Node->getType(), Node->isMicrosoftABI());
}

Stmt *ASTDuplicator::VisitGenericSelectionExpr(GenericSelectionExpr *Node) { return Node; }

Stmt *ASTDuplicator::VisitPseudoObjectExpr(PseudoObjectExpr *Node) {
  SmallVector<Expr *, 16> NewSemantics;
  for (PseudoObjectExpr::semantics_iterator S = Node->semantics_begin(),
      SEnd = Node->semantics_end(); S != SEnd; ++S) {
    NewSemantics.push_back(VisitExpr(*S));
  }
  return PseudoObjectExpr::Create(ASTCtx, Node->getSyntacticForm(),
      NewSemantics, Node->getResultExprIndex());
}

Stmt *ASTDuplicator::VisitSourceLocExpr(SourceLocExpr *Node) {
  return new (ASTCtx) SourceLocExpr(ASTCtx, Node->getIdentKind(),
      Node->getBeginLoc(), Node->getEndLoc(), Node->getParentContext());
}

// Wrapper expressions

Stmt *ASTDuplicator::VisitConstantExpr(ConstantExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return ConstantExpr::Create(ASTCtx, NewSubExpr, Node->getAPValueResult());
}

// Atomic expressions

Stmt *ASTDuplicator::VisitAtomicExpr(AtomicExpr *Node) { return Node; }

// GNU extensions

Stmt *ASTDuplicator::VisitAddrLabelExpr(AddrLabelExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitStmtExpr(StmtExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitChooseExpr(ChooseExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitGNUNullExpr(GNUNullExpr *Node) { return Node; }

// C++ expressions

Stmt *ASTDuplicator::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  return new (ASTCtx) CXXBoolLiteralExpr(
    Node->getValue(), Node->getType(), Node->getLocation());
}

Stmt *ASTDuplicator::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
  Expr *NewCallee = VisitExpr(Node->getCallee());
  SmallVector<Expr *, 16> NewArgs;
  for (CallExpr::arg_iterator A = Node->arg_begin(), AEnd = Node->arg_end();
       A != AEnd; ++A) {
    NewArgs.push_back(VisitExpr(*A));
  }
  return CXXOperatorCallExpr::Create(
    ASTCtx, Node->getOperator(), NewCallee, NewArgs, Node->getType(), Node->getValueKind(),
    Node->getRParenLoc(), Node->getFPFeatures(), Node->getADLCallKind());
}

Stmt *ASTDuplicator::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXConstCastExpr(CXXConstCastExpr *Node) { return Node; }

Stmt *ASTDuplicator::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  CXXCastPath BasePaths(Node->path_begin(), Node->path_end());
  return CXXFunctionalCastExpr::Create(
      ASTCtx, Node->getType(), Node->getValueKind(),
      Node->getTypeInfoAsWritten(), Node->getCastKind(), NewSubExpr,
      &BasePaths, Node->getLParenLoc(), Node->getRParenLoc());
}

Stmt *ASTDuplicator::VisitCXXTypeidExpr(CXXTypeidExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitUserDefinedLiteral(UserDefinedLiteral *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXThisExpr(CXXThisExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXThrowExpr(CXXThrowExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXDefaultInitExpr(CXXDefaultInitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXNewExpr(CXXNewExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXDeleteExpr(CXXDeleteExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXPseudoDestructorExpr(CXXPseudoDestructorExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitTypeTraitExpr(TypeTraitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitArrayTypeTraitExpr(ArrayTypeTraitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitExpressionTraitExpr(ExpressionTraitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXConstructExpr(CXXConstructExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXInheritedCtorInitExpr(CXXInheritedCtorInitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) { return Node; }

Stmt *ASTDuplicator::VisitExprWithCleanups(ExprWithCleanups *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return ExprWithCleanups::Create(ASTCtx, NewSubExpr,
      Node->cleanupsHaveSideEffects(), Node->getObjects());
}

Stmt *ASTDuplicator::VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXUnresolvedConstructExpr(CXXUnresolvedConstructExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitUnresolvedMemberExpr(UnresolvedMemberExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXNoexceptExpr(CXXNoexceptExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitPackExpansionExpr(PackExpansionExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitSizeOfPackExpr(SizeOfPackExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitSubstNonTypeTemplateParmPackExpr(SubstNonTypeTemplateParmPackExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitFunctionParmPackExpr(FunctionParmPackExpr *Node) { return Node; }

Stmt *ASTDuplicator::VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *Node) {
  Expr *NewSubExpr = VisitExpr(Node->getSubExpr());
  return new (ASTCtx) MaterializeTemporaryExpr(Node->getType(), NewSubExpr,
      (Node->getValueKind() == VK_LValue));
}

Stmt *ASTDuplicator::VisitLambdaExpr(LambdaExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXFoldExpr(CXXFoldExpr *Node) { return Node; }

// C++ Coroutines TS expressions

Stmt *ASTDuplicator::VisitCoawaitExpr(CoawaitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitDependentCoawaitExpr(DependentCoawaitExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCoyieldExpr(CoyieldExpr *Node) { return Node; }

// C++20 Concepts expressions

Stmt *ASTDuplicator::VisitConceptSpecializationExpr(ConceptSpecializationExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitRequiresExpr(RequiresExpr *Node) { return Node; }

// Obj-C expressions

Stmt *ASTDuplicator::VisitObjCStringLiteral(ObjCStringLiteral *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCBoxedExpr(ObjCBoxedExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCArrayLiteral(ObjCArrayLiteral *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCMessageExpr(ObjCMessageExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCIsaExpr(ObjCIsaExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCIndirectCopyRestoreExpr(ObjCIndirectCopyRestoreExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *Node) { return Node; }

// Obj-C ARC expressions

Stmt *ASTDuplicator::VisitObjCBridgedCastExpr(ObjCBridgedCastExpr *Node) { return Node; }

// CUDA expressions

Stmt *ASTDuplicator::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) { return Node; }

// Clang extensions

Stmt *ASTDuplicator::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitConvertVectorExpr(ConvertVectorExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitBlockExpr(BlockExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitOpaqueValueExpr(OpaqueValueExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitTypoExpr(TypoExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitBuiltinBitCastExpr(BuiltinBitCastExpr *Node) { return Node; }

// Microsoft extensions

Stmt *ASTDuplicator::VisitMSPropertyRefExpr(MSPropertyRefExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitMSPropertySubscriptExpr(MSPropertySubscriptExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitCXXUuidofExpr(CXXUuidofExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitSEHTryStmt(SEHTryStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitSEHExceptStmt(SEHExceptStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitSEHFinallyStmt(SEHFinallyStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitSEHLeaveStmt(SEHLeaveStmt *Node) { return Node; }
Stmt *ASTDuplicator::VisitMSDependentExistsStmt(MSDependentExistsStmt *Node) { return Node; }

// OpenCL extensions

Stmt *ASTDuplicator::VisitAsTypeExpr(AsTypeExpr *Node) {
  Expr *NewSrcExpr = VisitExpr(Node->getSrcExpr());
  return new (ASTCtx) AsTypeExpr(
    NewSrcExpr, Node->getType(), Node->getValueKind(), Node->getObjectKind(),
    Node->getBuiltinLoc(), Node->getRParenLoc());
}

// OpenMP directives

Stmt *ASTDuplicator::VisitOMPParallelDirective(OMPParallelDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTeamsDirective(OMPTeamsDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTaskyieldDirective(OMPTaskyieldDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTaskwaitDirective(OMPTaskwaitDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTaskgroupDirective(OMPTaskgroupDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTaskDirective(OMPTaskDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetUpdateDirective(OMPTargetUpdateDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetTeamsDirective(OMPTargetTeamsDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetParallelForDirective(OMPTargetParallelForDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetParallelDirective(OMPTargetParallelDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetExitDataDirective(OMPTargetExitDataDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetEnterDataDirective(OMPTargetEnterDataDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetDirective(OMPTargetDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetDataDirective(OMPTargetDataDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPSingleDirective(OMPSingleDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPFlushDirective(OMPFlushDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPCriticalDirective(OMPCriticalDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPCancellationPointDirective(OMPCancellationPointDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPCancelDirective(OMPCancelDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPBarrierDirective(OMPBarrierDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPAtomicDirective(OMPAtomicDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPSectionsDirective(OMPSectionsDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPSectionDirective(OMPSectionDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPParallelSectionsDirective(OMPParallelSectionsDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPParallelMasterDirective(OMPParallelMasterDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPOrderedDirective(OMPOrderedDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPMasterDirective(OMPMasterDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTeamsDistributeSimdDirective(OMPTeamsDistributeSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTeamsDistributeParallelForSimdDirective(OMPTeamsDistributeParallelForSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTeamsDistributeParallelForDirective(OMPTeamsDistributeParallelForDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTeamsDistributeDirective(OMPTeamsDistributeDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTaskLoopSimdDirective(OMPTaskLoopSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTaskLoopDirective(OMPTaskLoopDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetTeamsDistributeSimdDirective(OMPTargetTeamsDistributeSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetTeamsDistributeParallelForSimdDirective(OMPTargetTeamsDistributeParallelForSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetTeamsDistributeParallelForDirective(OMPTargetTeamsDistributeParallelForDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPMasterTaskLoopSimdDirective(OMPMasterTaskLoopSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPMasterTaskLoopDirective(OMPMasterTaskLoopDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPForSimdDirective(OMPForSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPForDirective(OMPForDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetTeamsDistributeDirective(OMPTargetTeamsDistributeDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetSimdDirective(OMPTargetSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPTargetParallelForSimdDirective(OMPTargetParallelForSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPSimdDirective(OMPSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPParallelMasterTaskLoopSimdDirective(OMPParallelMasterTaskLoopSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPParallelMasterTaskLoopDirective(OMPParallelMasterTaskLoopDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPParallelForSimdDirective(OMPParallelForSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPParallelForDirective(OMPParallelForDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPArraySectionExpr(OMPArraySectionExpr *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPDistributeSimdDirective(OMPDistributeSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPDistributeParallelForSimdDirective(OMPDistributeParallelForSimdDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPDistributeParallelForDirective(OMPDistributeParallelForDirective *Node) { return Node; }
Stmt *ASTDuplicator::VisitOMPDistributeDirective(OMPDistributeDirective *Node) { return Node; }

// Public methods

void ASTDuplicator::ReplaceDecl(NamedDecl *From, NamedDecl *To) {
  DupDecl[From] = To;
}

// Private methods

NamedDecl *ASTDuplicator::LookupDecl(NamedDecl *D) {
  if (DupDecl.find(D) == DupDecl.end()) {
    return D;
  } else {
    return DupDecl[D];
  }
}

VarDecl *ASTDuplicator::LookupVarDecl(VarDecl *VD) {
  return reinterpret_cast<VarDecl*>(LookupDecl(VD));
}

VarDecl *ASTDuplicator::DuplicateVarDecl(VarDecl *VD) {
  IdentifierInfo *NewII = Builder.GetUniqueDeclName(VD->getIdentifier());
  VarDecl *NewVD = VarDecl::Create(
    ASTCtx, &DeclCtx, VD->getInnerLocStart(), VD->getLocation(),
    NewII, VD->getType(), VD->getTypeSourceInfo(), VD->getStorageClass());

  for (VarDecl::attr_iterator A = VD->attr_begin(), AEnd = VD->attr_end();
       A != AEnd; ++A) {
    NewVD->addAttr((*A)->clone(ASTCtx));
  }

  if (VD->hasInit()) {
    NewVD->setInit(VisitExpr(VD->getInit()));
  }

  DeclCtx.addDecl(NewVD);
  DupDecl[VD] = NewVD;
  return NewVD;
}

LabelDecl *ASTDuplicator::LookupOrDuplicateLabelDecl(LabelDecl *LD, SourceLocation Loc) {
  if (DupDecl.find(LD) == DupDecl.end()) {
    IdentifierInfo *NewII = Builder.GetUniqueDeclName(LD->getIdentifier());
    LabelDecl *NewLD = LabelDecl::Create(ASTCtx, &DeclCtx, Loc, NewII);

    for (VarDecl::attr_iterator A = LD->attr_begin(), AEnd = LD->attr_end();
         A != AEnd; ++A) {
      NewLD->addAttr((*A)->clone(ASTCtx));
    }

    DeclCtx.addDecl(NewLD);
    DupDecl[LD] = NewLD;
  }
  return reinterpret_cast<LabelDecl*>(DupDecl[LD]);
}

LabelDecl *ASTDuplicator::LookupOrCreateReturnLabelDecl() {
  if (!ReturnLabel) {
    IdentifierInfo *NewII = Builder.GetUniqueDeclName("__func_exit");
    ReturnLabel = LabelDecl::Create(ASTCtx, &DeclCtx, SourceLocation(), NewII);
    DeclCtx.addDecl(ReturnLabel);
  }
  return ReturnLabel;
}

} // namespace snu

} // namespace clang
