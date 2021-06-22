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

#include "clang/SnuMAPA/MemoryAccessPattern.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WAST.h"
#include "clang/SnuAST/WCFG.h"
#include "clang/SnuAnalysis/MemoryAccess.h"
#include "clang/SnuAnalysis/PointerAnalysis.h"
#include "clang/SnuMAPA/SymbolicAnalysis.h"
#include "clang/SnuSupport/OrderedDenseADT.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace clang {

namespace snu {

namespace {

bool IsConstantExpr(const SymbolicExpr *E);
bool IsAffineIn(bool (*Cond)(const SEVariable *), const SymbolicExpr *E);
  bool IAI_IDs(const SEVariable *Var);
  bool IAI_Iter(const SEVariable *Var);
  bool IAI_SimpleBoundIter(const SEVariable *Var);
  bool IAI_FixedBoundIter(const SEVariable *Var);
  bool IAI_IDsAndIter(const SEVariable *Var);
  bool IAI_IDsAndSimpleBoundIter(const SEVariable *Var);
  bool IAI_IDsAndFixedBoundIter(const SEVariable *Var);
bool IsFixedBoundInductionVariable(const SEInductionVariable *IndVar);
unsigned GetNestingLevel(const SEInductionVariable *IndVar);

bool IsConstantExpr(const SymbolicExpr *E) {
  return isa<SEInvariant>(E);
}

bool IsAffineIn(bool (*Cond)(const SEVariable *), const SymbolicExpr *E) {
  if (const SEAffine *Affine = dyn_cast<SEAffine>(E)) {
    for (unsigned Index = 0, NumTerms = Affine->getNumTerms();
         Index != NumTerms; ++Index) {
      if (!Cond(Affine->getVariable(Index))) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

bool IAI_IDs(const SEVariable *Var) {
  return Var == NULL || isa<SENDRangeIndex>(Var);
}

bool IAI_Iter(const SEVariable *Var) {
  return Var == NULL || isa<SEInductionVariable>(Var);
}

bool IAI_SimpleBoundIter(const SEVariable *Var) {
  if (Var == NULL) return true;
  if (const SEInductionVariable *IndVar = dyn_cast<SEInductionVariable>(Var)) {
    return IsFixedBoundInductionVariable(IndVar) &&
           GetNestingLevel(IndVar) == 0;
  }
  return false;
}

bool IAI_FixedBoundIter(const SEVariable *Var) {
  if (Var == NULL) return true;
  if (const SEInductionVariable *IndVar = dyn_cast<SEInductionVariable>(Var)) {
    return IsFixedBoundInductionVariable(IndVar);
  }
  return false;
}

bool IAI_IDsAndIter(const SEVariable *Var) {
  return Var == NULL || isa<SENDRangeIndex>(Var) ||
         isa<SEInductionVariable>(Var);
}

bool IAI_IDsAndSimpleBoundIter(const SEVariable *Var) {
  if (Var == NULL || isa<SENDRangeIndex>(Var)) return true;
  if (const SEInductionVariable *IndVar = dyn_cast<SEInductionVariable>(Var)) {
    return IsFixedBoundInductionVariable(IndVar) &&
           GetNestingLevel(IndVar) == 0;
  }
  return false;
}

bool IAI_IDsAndFixedBoundIter(const SEVariable *Var) {
  if (Var == NULL || isa<SENDRangeIndex>(Var)) return true;
  if (const SEInductionVariable *IndVar = dyn_cast<SEInductionVariable>(Var)) {
    return IsFixedBoundInductionVariable(IndVar);
  }
  return false;
}

bool IsFixedBoundInductionVariable(const SEInductionVariable *IndVar) {
  assert(IndVar->getBound());
  return IsAffineIn(IAI_IDsAndFixedBoundIter, IndVar->getBound());
}

unsigned GetNestingLevel(const SEInductionVariable *IndVar) {
  unsigned NestingLevel = 0;
  const SEAffine *Bound = IndVar->getBound();
  assert(Bound);
  for (unsigned Index = 0, NumTerms = Bound->getNumTerms();
       Index != NumTerms; ++Index) {
    const SEVariable *SubVar = Bound->getVariable(Index);
    if (const SEInductionVariable *SubIndVar =
            dyn_cast_or_null<SEInductionVariable>(SubVar)) {
      NestingLevel += GetNestingLevel(SubIndVar);
    }
  }
  return NestingLevel;
}

} // anonymous namespace

MAPElement::MAPElement()
  : Kind(OK_COMPLEX), Addr(NULL), Type(), Width(0), Read(false), Written(false),
    Reusable(false), OneThreadOp(false) {
}

MAPElement::MAPElement(QualType type, uint64_t width, bool r, bool w, bool ru)
  : Kind(OK_COMPLEX), Addr(NULL), Type(type), Width(width), Read(r), Written(w),
    Reusable(ru), OneThreadOp(false) {
  if (Reusable) assert(Written);
  if (Read && Written) assert(Reusable);
}

MAPElement::MAPElement(const SEBufferAddress *addr, QualType type,
                       uint64_t width, bool r, bool w, bool ru)
  : Kind(OK_COMPLEX), Addr(addr), Type(type), Width(width), Read(r), Written(w),
    Reusable(ru), OneThreadOp(false) {
  assert(Addr != NULL);
  const SymbolicExpr *Offset = Addr->getOffset();
  assert(Offset != NULL);
  if (IsConstantExpr(Offset)) {
    Kind = OK_CONSTANT;
  } else if (IsAffineIn(IAI_IDs, Offset)) {
    Kind = OK_AFFINE_IN_IDS;
  } else if (IsAffineIn(IAI_SimpleBoundIter, Offset)) {
    Kind = OK_AFFINE_IN_SIMPLE_BOUND_ITER;
  } else if (IsAffineIn(IAI_FixedBoundIter, Offset)) {
    Kind = OK_AFFINE_IN_FIXED_BOUND_ITER;
  } else if (IsAffineIn(IAI_IDsAndSimpleBoundIter, Offset)) {
    Kind = OK_AFFINE_IN_IDS_AND_SIMPLE_BOUND_ITER;
  } else if (IsAffineIn(IAI_IDsAndFixedBoundIter, Offset)) {
    Kind = OK_AFFINE_IN_IDS_AND_FIXED_BOUND_ITER;
  } else if (isa<SEAffine>(Offset)) {
    Kind = OK_COMPLEX_AFFINE;
  } else {
    Kind = OK_COMPLEX;
  }
  if (Reusable) assert(Written);
  if (Read && Written) assert(Reusable);
}

bool MAPElement::operator==(const MAPElement &RHS) const {
  if (Addr == NULL || RHS.Addr == NULL) {
    return false;
  }
  return (*Addr == *RHS.Addr && Width == RHS.Width && Read == RHS.Read &&
          Written == RHS.Written && Reusable == RHS.Reusable);
}

void MAPElement::print(raw_ostream &OS) const {
  if (Addr == NULL) {
    OS << "<unknown>,<unknown>,";
  } else {
    Addr->getBase()->printName(OS);
    OS << ',';
    Addr->getOffset()->print(OS);
    OS << ',';
  }
  switch (Kind) {
    case OK_CONSTANT: OS << "constant,"; break;
    case OK_AFFINE_IN_IDS: OS << "affine in IDs,"; break;
    case OK_AFFINE_IN_SIMPLE_BOUND_ITER: OS << "affine in simple-bound iter.,"; break;
    case OK_AFFINE_IN_FIXED_BOUND_ITER: OS << "affine in fixed-bound iter.,"; break;
    case OK_AFFINE_IN_IDS_AND_SIMPLE_BOUND_ITER: OS << "affine in IDs and simple-bound iter.,"; break;
    case OK_AFFINE_IN_IDS_AND_FIXED_BOUND_ITER: OS << "affine in IDs and fixed-bound iter.,"; break;
    case OK_COMPLEX_AFFINE: OS << "complex affine,"; break;
    case OK_COMPLEX: OS << "complex,"; break;
    default: llvm_unreachable("impossible case");
  }
  OS << Width << ',';
  if (Read) OS << 'r';
  if (Written) OS << 'w';
  if (Reusable) OS << 'u';
}

MAPInterval::MAPInterval()
  : Kind(MAPElement::OK_COMPLEX), Addr(NULL), Read(false), Written(false) {
}

MAPInterval::MAPInterval(const MAPElement &E)
  : Kind(E.getKind()), Addr(E.getAddress()), Read(E.isRead()),
    Written(E.isWritten()) {
  Members.push_back(MAPIntervalMember(E, 0));
}

bool MAPInterval::isSingleLocation() const {
  if (Members.empty()) return false;
  if (Members.size() == 1) return true;
  QualType AccessTy = Members[0].Element.getType();
  for (SmallVectorImpl<MAPIntervalMember>::const_iterator M = Members.begin() + 1,
                                                          MEnd = Members.end();
       M != MEnd; ++M) {
    if ((*M).Element.getType() != AccessTy || (*M).Distance > 0) {
      return false;
    }
  }
  return true;

}

bool MAPInterval::contains(const MAPElement &E) const {
  for (SmallVectorImpl<MAPIntervalMember>::const_iterator M = Members.begin(),
                                                          MEnd = Members.end();
       M != MEnd; ++M) {
    if ((*M).Element == E) {
      return true;
    }
  }
  return false;
}

uint64_t MAPInterval::getDistanceOf(const MAPElement &E) const {
  for (SmallVectorImpl<MAPIntervalMember>::const_iterator M = Members.begin(),
                                                          MEnd = Members.end();
       M != MEnd; ++M) {
    if ((*M).Element == E) {
      return (*M).Distance;
    }
  }
  llvm_unreachable("not a member");
  return 0;
}

bool MAPInterval::addElement(const MAPElement &E,
                             const SymbolicExprContext &SECtx) {
  if (Members.empty()) {
    Kind = E.getKind();
    Addr = E.getAddress();
    Members.push_back(MAPIntervalMember(E, 0));
    Read = E.isRead();
    Written = E.isWritten();
    return true;
  }

  if (Addr == NULL || E.getAddress() == NULL) {
    return false;
  }
  const SEConstant *D = SECtx.CreateAddrMinusAddrConst(E.getAddress(), Addr);
  if (D == NULL) {
    return false;
  }
  assert(Kind == E.getKind());
  if (D->getValue() >= 0) {
    Members.push_back(MAPIntervalMember(E, D->getValue()));
  } else {
    Addr = E.getAddress();
    for (SmallVectorImpl<MAPIntervalMember>::iterator M = Members.begin(),
                                                      MEnd = Members.end();
         M != MEnd; ++M) {
      (*M).Distance += -D->getValue();
    }
    Members.push_back(MAPIntervalMember(E, 0));
  }
  std::sort(Members.begin(), Members.end());
  Read |= E.isRead();
  Written |= E.isWritten();
  return true;
}

void MAPInterval::print(raw_ostream &OS) const {
  for (unsigned Index = 0, NumMembers = Members.size();
       Index != NumMembers; ++Index) {
    if (Index) OS << '\n';
    OS << "interval " << (uint64_t)this << ',' << Members[Index].Distance << ',';
    Members[Index].Element.print(OS);
  }
}

void MAPBuffer::addElement(const MAPElement &E,
                           const SymbolicExprContext &SECtx) {
  Read |= E.isRead();
  Written |= E.isWritten();
  Reusable |= E.isReusable();
  if (E.getAddress() == NULL || E.getOffset()->isUnknown()) {
    HasUnknown = true;
  } else {
    for (unsigned Index = 0, NumMembers = Members.size();
         Index != NumMembers; ++Index) {
      if (Members[Index].addElement(E, SECtx)) {
        return;
      }
    }
    Members.push_back(MAPInterval(E));
  }
}

void MAPBuffer::print(raw_ostream &OS) const {
  for (unsigned Index = 0, NumMembers = Members.size();
       Index != NumMembers; ++Index) {
    if (Index) OS << '\n';
    Members[Index].print(OS);
  }
  if (HasUnknown) {
    if (!Members.empty()) OS << '\n';
    OS << "<unknown>";
  }
}

MAPACore::MAPACore(ASTContext &C, WCFG *program)
  : ASTCtx(C), Program(program), SECtx(), SA(ASTCtx, SECtx),
    Aliases(program, MemoryAccessTrace::GetGlobalAddressSpaceFilter(C)) {
  assert(Program->isSSA());
  InitializeVariables();
  CalculateVariables();
}

void MAPACore::InitializeVariables() {
  for (WCFG::ssa_var_iterator V = Program->ssa_var_begin(),
                              VEnd = Program->ssa_var_end();
       V != VEnd; ++V) {
    IndexedVarDecl *Var = *V;
    const SymbolicExpr *Init = SECtx.Unknown;
    if (Var->isParameter()) {
      QualType VarTy = Var->getType();
      if (VarTy->isPointerType() &&
          VarTy->getPointeeType().getAddressSpace() != LangAS::opencl_local) {
        Init = new (SECtx) SEBufferAddress(Var, SECtx.Zero);
      } else if (VarTy->isOpenCLSpecificType()) {
        // Init = Unknown;
      } else {
        Init = new (SECtx) SEParameter(Var);
      }
    }
    SA.InitVariable(Var, Init);
  }
}

void MAPACore::CalculateVariables() {
  bool Updated;
  do {
    Updated = false;
    for (WCFG::ssa_var_iterator V = Program->ssa_var_begin(),
                                VEnd = Program->ssa_var_end();
         V != VEnd; ++V) {
      if (WMuFunction *DefinedMu = (*V)->getDefinedMu()) {
        const SymbolicExpr *P = SA.GetValueOf(DefinedMu);
        Updated |= SA.UpdateVariable(*V, P);
      } else if (WStmt *DefinedStmt = (*V)->getDefinedStmt()) {
        const SymbolicExpr *P = SA.GetAssignedValueOf(DefinedStmt);
        Updated |= SA.UpdateVariable(*V, P);
      } else if (WStmt *CompoundDefinedStmt = (*V)->getCompoundDefinedStmt()) {
        WSubVarDecl *SubVar = dyn_cast<WSubVarDecl>((*V)->getDecl());
        assert(SubVar != NULL);
        if ((*V)->getType()->isStructureType()) {
          const SymbolicExpr *P = SA.GetFieldAssignedValueOf(
              CompoundDefinedStmt, SubVar->getField().Field);
          Updated |= SA.UpdateVariable(*V, P);
        } else {
          // Do nothing
        }
      }
    }
  } while (Updated);
}

void MAPACore::Analysis() {
  MemoryAccessTrace Trace(Program, MemoryAccessTrace::GetGlobalAddressSpaceFilter(ASTCtx));
  Analysis(Trace);
}

void MAPACore::Analysis(MemoryAccessTrace &Trace) {
  OperationPattern.clear();
  BufferPattern.clear();
  for (WCFG::ssa_var_iterator V = Program->ssa_var_begin(),
                              VEnd = Program->ssa_var_end();
       V != VEnd; ++V) {
    IndexedVarDecl *Var = *V;
    if (Var->isParameter()) {
      QualType VarTy = Var->getType();
      if (VarTy->isPointerType() &&
          VarTy->getPointeeType().getAddressSpace() != LangAS::opencl_local) {
        BufferPattern[Var->getDecl()->getAsParameter()] = MAPBuffer();
      }
    }
  }

  MemoryAccessRelation *ReachR = MemoryAccessRelation::CreateReachability(Trace, Program);
  ReachR->filter(MemoryAccessRelation::WriteRead);
  MemoryAccessRelation *AliasR = Aliases.CreateRelation(Trace);
  ReachR->join(*AliasR);
  delete AliasR;

  for (MemoryAccessTrace::iterator I = Trace.begin(), E = Trace.end();
       I != E; ++I) {
    MemoryAccess *Entry = *I;
    if (Entry->hasChild()) {
      continue;
    }

    WExpr *Access = Entry->getAccessExpr();
    QualType Type = Entry->getAccessType();
    uint64_t Width = ASTCtx.getTypeSize(Type) / ASTCtx.getCharWidth();
    bool Read = Entry->isRead();
    bool Write = Entry->isWrite();
    bool Reusable = ReachR->containsAtFirst(Entry);
    assert(Aliases.isMemoryAccess(Access));

    if (const SEBufferAddress *Addr = dyn_cast<SEBufferAddress>(SA.GetAddressOf(Access))) {
      assert(Aliases.isPointsToParameter(Access));
      MAPElement Element = MAPElement(Addr, Type, Width, Read, Write, Reusable);
      InspectOneThreadOp(Access, &Element);
      OperationPattern[Access] = Element;
      assert(BufferPattern.count(Addr->getBase()));
      BufferPattern[Addr->getBase()].addElement(Element, SECtx);

    } else if (Aliases.isPointsToParameter(Access)) {
      MAPElement Element = MAPElement(Type, Width, Read, Write, Reusable);
      InspectOneThreadOp(Access, &Element);
      OperationPattern[Access] = Element;
      // Invalidate all buffers in the same alias set
      for (BufferPatternMapTy::const_iterator B = BufferPattern.begin(),
                                              BEnd = BufferPattern.end();
           B != BEnd; ++B) {
        if (Aliases.isUsed(B->first) &&
            Aliases.getAlias(B->first) == Aliases.getAlias(Access)) {
          BufferPattern[B->first].addElement(Element, SECtx);
        }
      }

    } else {
      MAPElement Element = MAPElement(Type, Width, Read, Write, Reusable);
      InspectOneThreadOp(Access, &Element);
      OperationPattern[Access] = Element;
    }
  }

  delete ReachR;
}

bool MAPACore::isWorkItemFunction(WExpr* E,
    enum WWorkItemFunction::WorkItemFunctionKind kind) {
  if (WPseudoObjectExpr *POE = dyn_cast<WPseudoObjectExpr>(E)) {
    if (WWorkItemFunction *WIF =
        dyn_cast<WWorkItemFunction>(POE->getResultExpr())) {
      if (WIF->getFunctionKind() == kind) {
        return true;
      }
    }
  } 
  return false;
}

// Expression: blockDim.x * blockIdx.x + threadIdx.x
bool MAPACore::isGlobalIdExpression(WExpr* E) {
  if (WBinaryOperator *OuterBO = dyn_cast<WBinaryOperator>(E)) {
    if (OuterBO->getOpcode() == BO_Add) {
      WExpr *OuterLHS = OuterBO->getLHS()->IgnoreParenCasts();
      WExpr *OuterRHS = OuterBO->getRHS()->IgnoreParenCasts();

      if (isWorkItemFunction(OuterRHS, WWorkItemFunction::WIF_get_local_id)) {
        if (WBinaryOperator *InnerBO = dyn_cast<WBinaryOperator>(OuterLHS)) {
          if (InnerBO->getOpcode() == BO_Mul) {
            if (isWorkItemFunction(InnerBO->getLHS()->IgnoreParenCasts(),
                  WWorkItemFunction::WIF_get_local_size) &&
                isWorkItemFunction(InnerBO->getRHS()->IgnoreParenCasts(),
                  WWorkItemFunction::WIF_get_group_id)) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

void MAPACore::InspectOneThreadOp(WExpr* Access, MAPElement *Elem) {
  for (unsigned Index = 0; Index < Access->getNumExecuteConds(); Index++) {
    WStmt *Cond = Access->getExecuteCond(Index);

    // i = blockDim.x * blockIdx.x + threadIdx.x
    // if (i == 0) ...
    if (WBinaryOperator *BO = dyn_cast<WBinaryOperator>(Cond)) {
      if (BO->getOpcode() == BO_EQ) {
        WExpr *LHS = BO->getLHS()->IgnoreParenCasts();
        WExpr *RHS = BO->getRHS()->IgnoreParenCasts();

        if (WDeclRefExpr *DRE = dyn_cast<WDeclRefExpr>(LHS)) {
          IndexedVarDeclRef UseDecl = DRE->getIndexedUseDecl();
          if (WDeclStmt *DS = dyn_cast<WDeclStmt>(UseDecl->getDefinedStmt())) {
            WExpr *DefExpr = DS->getSingleInit()->IgnoreParenCasts();
            if (DefExpr && isGlobalIdExpression(DefExpr)) {
              Elem->setOneThreadOp();
              break;
            }
          }
        }
        else if (WDeclRefExpr *DRE = dyn_cast<WDeclRefExpr>(RHS)) {
          IndexedVarDeclRef UseDecl = DRE->getIndexedUseDecl();
          if (WDeclStmt *DS = dyn_cast<WDeclStmt>(UseDecl->getDefinedStmt())) {
            WExpr *DefExpr = DS->getSingleInit()->IgnoreParenCasts();
            if (DefExpr && isGlobalIdExpression(DefExpr)) {
              Elem->setOneThreadOp();
              break;
            }
          }
        }
      }
    }
  }
//  E->print(llvm::errs(), PrintingPolicy(LangOptions()));
//  llvm::errs() << "\n";
//  llvm::errs() << "Class=" << E->getStmtClassName() << "\n";
//
//  E->getOriginal()->dumpColor();

//        Cond->print(llvm::errs(), PrintingPolicy(LangOptions()));
//        llvm::errs() << "\n";
//
//        llvm::errs() << "LHS=" << BO->getLHS()->IgnoreParenCasts()->getStmtClassName() << "\n";
//        llvm::errs() << "RHS=" << BO->getRHS()->IgnoreParenCasts()->getStmtClassName() << "\n";
//
//        llvm::errs() << "Cond " << Index << ":\n";
//        Cond->getOriginal()->dumpColor();
//
//        if (WPseudoObjectExpr *POE = dyn_cast<WPseudoObjectExpr>(BO->getLHS())) {
//          llvm::errs() << POE->getResultExpr()->getStmtClassName() << "\n";
//          if (WWorkItemFunction *WIF = dyn_cast<WWorkItemFunction>(POE->getResultExpr())) {
//            if (WIF->getFunctionKind() == 
//          }
//        }
}

void MAPACore::print(raw_ostream &OS, const ASTContext &ASTCtx) const {
  OS << "MAPA:\n";
  for (OperationPatternMapTy::const_iterator I = OperationPattern.begin(),
                                             E = OperationPattern.end();
       I != E; ++I) {
    OS << "  Expression \"";
    I->first->print(OS, ASTCtx.getPrintingPolicy());
    OS << "\": ";
    I->second.print(OS);
    OS << '\n';
  }
  OS << '\n';
  for (BufferPatternMapTy::const_iterator I = BufferPattern.begin(),
                                          E = BufferPattern.end();
       I != E; ++I) {
    OS << "  Buffer \"";
    I->first->printName(OS);
    OS << "\":\n";
    I->second.print(OS);
    OS << "\n\n";
  }
}

} // namespace snu

} // namespace clang
