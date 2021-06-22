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

#include "clang/SnuMAPA/SymbolicAnalysis.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WAST.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace clang {

namespace snu {

// SymbolicExpr

void* SymbolicExpr::operator new(size_t bytes, const SymbolicExprContext &C,
                                 unsigned alignment) {
  return C.Allocate(bytes, alignment);
}

int SymbolicExpr::Compare(const SymbolicExpr &P) const {
  if (Class < P.Class) {
    return -1;
  } else if (Class > P.Class) {
    return 1;
  } else {
    switch (Class) {
#define EX(type) \
    case type##Class: \
      return static_cast<const SE##type*>(this)->Compare(static_cast<const SE##type&>(P));
    SYMBOLIC_EXPRS()
#undef EX
    case UnknownClass:
      return 0;
    default:
      llvm_unreachable("invalid pattern class");
    }
  }
}

const SymbolicExpr *SymbolicExpr::CreateUnknown(const SymbolicExprContext &Ctx) {
  return new (Ctx) SymbolicExpr(UnknownClass);
}

const SEAffine *SEAffine::Create(const SymbolicExprContext &Ctx,
                                 ArrayRef<AffineTerm> terms) {
  if (terms.size() == 0) {
    return Ctx.Zero;
  }
  if (terms.size() == 1) {
    if (terms[0].Var == NULL) {
      return terms[0].Coeff;
    }
    if (const SEConstant *CoeffConst = dyn_cast<SEConstant>(terms[0].Coeff)) {
      if (CoeffConst->getValue() == 1) {
        return terms[0].Var;
      }
    }
  }
  return new (Ctx) SEComplexAffine(Ctx, terms);
}

bool SEInvariant::isZero() const {
  if (const SEConstant *C = dyn_cast<SEConstant>(this)) {
    return C->isZero();
  } else {
    return false;
  }
}

const SEConstant *SEConstant::Create(const SymbolicExprContext &Ctx,
                                     int64_t Value) {
#ifdef MAPA_USE_SYMBOLIC_EXPRESSION_CONSTANT_CACHE
  if (const SEConstant *Cached = Ctx.LookupConstantCache(Value)) {
    return Cached;
  } else {
#endif
    return new (Ctx) SEConstant(Value);
#ifdef MAPA_USE_SYMBOLIC_EXPRESSION_CONSTANT_CACHE
  }
#endif
}

const SEConstant *SEConstant::Create(const SymbolicExprContext &Ctx,
                                     const llvm::APInt &Value) {
  if (Value.getNumWords() == 1) {
    return SEConstant::Create(Ctx, *Value.getRawData());
  } else {
    return NULL;
  }
}

int SEInvariantOperation::Compare(const SEInvariantOperation &P) const {
  if (Opcode < P.Opcode) {
    return -1;
  } else if (Opcode > P.Opcode) {
    return 1;
  } else {
    int RHSCompare = RHS->Compare(*(P.RHS));
    if (RHSCompare != 0) {
      return RHSCompare;
    }
    return LHS->Compare(*(P.LHS));
  }
}

SEVariable::SEVariable(const SymbolicExprContext &Ctx, SymbolicExprClass PC)
  : SEAffine(PC), DummyCoefficient(Ctx.One) {}

int SEInductionVariable::Compare(const SEInductionVariable &P) const {
#ifndef MAPA_AGGRESSIVE_SYMBOLIC_EXPRESSION_COMPARE
  if (Loop < P.Loop) {
    return -1;
  } else if (Loop > P.Loop) {
    return 1;
  } else {
#endif
    int BoundCompare = Bound->Compare(*(P.Bound));
    if (BoundCompare != 0) {
      return BoundCompare;
    }
    int StepCompare = Step->Compare(*(P.Step));
    if (StepCompare != 0) {
      return StepCompare;
    }
    if (Inclusive != P.Inclusive) {
      return (Inclusive ? 1 : -1);
    }
    return 0;
#ifndef MAPA_AGGRESSIVE_SYMBOLIC_EXPRESSION_COMPARE
  }
#endif
}

int SEBufferValue::Compare(const SEBufferValue &P) const {
  int AddressCompare = Address->Compare(*(P.Address));
  if (AddressCompare != 0) {
    return AddressCompare;
  }
  if (Type.getTypePtr() < P.Type.getTypePtr()) {
    return -1;
  } else if (Type.getTypePtr() > P.Type.getTypePtr()) {
    return 1;
  } else {
    return 0;
  }
}

SEComplexAffine::SEComplexAffine(const SymbolicExprContext &Ctx,
                                 ArrayRef<AffineTerm> terms)
  : SEAffine(ComplexAffineClass) {
  NumTerms = terms.size();
  Terms = (AffineTerm*)Ctx.Allocate(sizeof(AffineTerm) * NumTerms);
  std::copy(terms.begin(), terms.end(), Terms);
}

static int CompareVariable(const SEVariable *LHS, const SEVariable *RHS) {
  if (LHS == NULL && RHS == NULL) {
    return 0;
  } else if (LHS == NULL) {
    return -1;
  } else if (RHS == NULL) {
    return 1;
  } else {
    return LHS->Compare(*RHS);
  }
}

int SEComplexAffine::Compare(const SEComplexAffine &P) const {
  unsigned Index = 0;
  unsigned LHSNumTerms = getNumTerms();
  unsigned RHSNumTerms = P.getNumTerms();

  while (Index < LHSNumTerms && Index < RHSNumTerms) {
    const SEVariable *LHSVar = getVariable(Index);
    const SEVariable *RHSVar = P.getVariable(Index);
    int VarCompare = CompareVariable(LHSVar, RHSVar);
    if (VarCompare != 0) {
      return VarCompare;
    } else {
      const SEInvariant *LHSCoeff = getCoefficient(Index);
      const SEInvariant *RHSCoeff = P.getCoefficient(Index);
      int CoeffCompare = LHSCoeff->Compare(*RHSCoeff);
      if (CoeffCompare != 0) {
        return CoeffCompare;
      }
    }
    Index++;
  }

  if (Index == LHSNumTerms && Index == RHSNumTerms) {
    return 0;
  } else if (Index == LHSNumTerms) {
    return -1;
  } else { // Index == RHSNumTerms
    return 1;
  }
}

int SEBufferAddress::Compare(const SEBufferAddress &P) const {
  if (IBase != P.IBase) {
    return getBase()->getName().compare(P.getBase()->getName());
  } else {
    return Offset->Compare(*(P.Offset));
  }
}

// SymbolicExpr Printer

void SymbolicExpr::print(raw_ostream &OS) const {
  switch (Class) {
#define EX(type) \
  case type##Class: \
    static_cast<const SE##type*>(this)->print(OS); \
    break;
  SYMBOLIC_EXPRS()
#undef EX
  case UnknownClass:
    OS << "<unknown>";
    break;
  default:
    llvm_unreachable("invalid pattern class");
  }
}

void SEConstant::print(raw_ostream &OS) const {
  OS << Value;
}

void SEParameter::print(raw_ostream &OS) const {
  getDecl()->printName(OS);
}

void SENDRangeDimension::print(raw_ostream &OS) const {
  switch (Kind) {
    case LOCAL_SIZE_0: OS << "local_size(0)"; break;
    case LOCAL_SIZE_1: OS << "local_size(1)"; break;
    case LOCAL_SIZE_2: OS << "local_size(2)"; break;
    case NUM_GROUPS_0: OS << "num_groups(0)"; break;
    case NUM_GROUPS_1: OS << "num_groups(1)"; break;
    case NUM_GROUPS_2: OS << "num_groups(2)"; break;
  }
}

void SEInvariantOperation::print(raw_ostream &OS) const {
  OS << '(';
  LHS->print(OS);
  OS << ' ' << BinaryOperator::getOpcodeStr(Opcode) << ' ';
  RHS->print(OS);
  OS << ')';
}

void SENDRangeIndex::print(raw_ostream &OS) const {
  switch (Kind) {
    case LOCAL_ID_0: OS << "local_id(0)"; break;
    case LOCAL_ID_1: OS << "local_id(1)"; break;
    case LOCAL_ID_2: OS << "local_id(2)"; break;
    case GROUP_ID_0: OS << "group_id(0)"; break;
    case GROUP_ID_1: OS << "group_id(1)"; break;
    case GROUP_ID_2: OS << "group_id(2)"; break;
  }
}

void SEInductionVariable::print(raw_ostream &OS) const {
  // Example: (loop variable [0 ~ 100) step 2)
  OS << "(loop variable [0 ~ ";
  Bound->print(OS);
  OS << (Inclusive ? ']' : ')') << " step ";
  Step->print(OS);
  OS << ')';
}

void SEComplexAffine::print(raw_ostream &OS) const {
  for (unsigned I = 0, NumTerms = getNumTerms(); I != NumTerms; ++I) {
    if (I) {
      OS << " + ";
    }
    OS << '[';
    const SEInvariant *Coefficient = getCoefficient(I);
    const SEVariable *Variable = getVariable(I);
    Coefficient->print(OS);
    if (Variable) {
      OS << " * ";
      Variable->print(OS);
    }
    OS << ']';
  }
}

void SEBufferAddress::print(raw_ostream &OS) const {
  OS << "(buffer ";
  getBase()->printName(OS);
  OS << " + offset ";
  Offset->print(OS);
  OS << ')';
}

void SEBufferValue::print(raw_ostream &OS) const {
  OS << '*';
  Address->print(OS);
}

// SymbolicExprContext

SymbolicExprContext::SymbolicExprContext() {
  Unknown = SymbolicExpr::CreateUnknown(*this);

  for (int64_t Value = -CONSTANT_CACHE_RANGE;
       Value <= CONSTANT_CACHE_RANGE; ++Value) {
    ConstantCache[Value + CONSTANT_CACHE_RANGE] = new (*this) SEConstant(Value);
  }
  Zero = ConstantCache[0 + CONSTANT_CACHE_RANGE];
  One = ConstantCache[1 + CONSTANT_CACHE_RANGE];
  Two = ConstantCache[2 + CONSTANT_CACHE_RANGE];
  MinusOne = ConstantCache[-1 + CONSTANT_CACHE_RANGE];

  LocalSize[0] = new (*this) SENDRangeDimension(SENDRangeDimension::LOCAL_SIZE_0);
  LocalSize[1] = new (*this) SENDRangeDimension(SENDRangeDimension::LOCAL_SIZE_1);
  LocalSize[2] = new (*this) SENDRangeDimension(SENDRangeDimension::LOCAL_SIZE_2);
  NumGroups[0] = new (*this) SENDRangeDimension(SENDRangeDimension::NUM_GROUPS_0);
  NumGroups[1] = new (*this) SENDRangeDimension(SENDRangeDimension::NUM_GROUPS_1);
  NumGroups[2] = new (*this) SENDRangeDimension(SENDRangeDimension::NUM_GROUPS_2);
  LocalID[0] = new (*this) SENDRangeIndex(*this, SENDRangeIndex::LOCAL_ID_0);
  LocalID[1] = new (*this) SENDRangeIndex(*this, SENDRangeIndex::LOCAL_ID_1);
  LocalID[2] = new (*this) SENDRangeIndex(*this, SENDRangeIndex::LOCAL_ID_2);
  GroupID[0] = new (*this) SENDRangeIndex(*this, SENDRangeIndex::GROUP_ID_0);
  GroupID[1] = new (*this) SENDRangeIndex(*this, SENDRangeIndex::GROUP_ID_1);
  GroupID[2] = new (*this) SENDRangeIndex(*this, SENDRangeIndex::GROUP_ID_2);
}

const SEConstant *SymbolicExprContext::LookupConstantCache(int64_t Value) const {
  if (Value >= -CONSTANT_CACHE_RANGE && Value <= CONSTANT_CACHE_RANGE) {
    return ConstantCache[Value + CONSTANT_CACHE_RANGE];
  } else {
    return NULL;
  }
}

namespace {

bool CompareInvariantPointer(const SEInvariant *A, const SEInvariant *B) {
  assert(A && B);
  return *A < *B;
}

} // anonymous namespace

const SEInvariant *SymbolicExprContext::CreateInvarOpInvar(
    const SEInvariant *LHS, const SEInvariant *RHS,
    BinaryOperatorKind Opcode) const {
  assert(LHS && RHS);
  if (Opcode != BO_Mul && Opcode != BO_Div && Opcode != BO_Rem &&
      Opcode != BO_Add && Opcode != BO_Sub &&
      Opcode != BO_Shl && Opcode != BO_Shr &&
      Opcode != BO_And && Opcode != BO_Xor && Opcode != BO_Or) {
    return NULL;
  }

  if (const SEInvariant *Merged = MergeInvarOpInvar(LHS, RHS, Opcode)) {
    return Merged;
  }

  // a - b => a + (-b)
  if (Opcode == BO_Sub) {
    Opcode = BO_Add;
    RHS = CreateInvarOpInvar(this->MinusOne, RHS, BO_Mul);
    assert(RHS);
  }

  // (a + b) * c => (a * c) + (b * c)
  // a * (b + c) => (a * b) + (a * c)
  if (Opcode == BO_Mul) {
    const SEInvariantOperation *LHSOp = dyn_cast<SEInvariantOperation>(LHS);
    const SEInvariantOperation *RHSOp = dyn_cast<SEInvariantOperation>(RHS);
    if (LHSOp && LHSOp->getOpcode() == BO_Add) {
      Opcode = BO_Add;
      LHS = CreateInvarOpInvar(LHSOp->getLHS(), RHS, BO_Mul);
      RHS = CreateInvarOpInvar(LHSOp->getRHS(), RHS, BO_Mul);
      assert(LHS && RHS);
    } else if (RHSOp && RHSOp->getOpcode() == BO_Add) {
      Opcode = BO_Add;
      RHS = CreateInvarOpInvar(LHS, RHSOp->getRHS(), BO_Mul);
      LHS = CreateInvarOpInvar(LHS, RHSOp->getLHS(), BO_Mul);
      assert(LHS && RHS);
    }
  }

  // Associative and commutative operator
  if (Opcode == BO_Mul || Opcode == BO_Add ||
      Opcode == BO_And || Opcode == BO_Xor || Opcode == BO_Or) {
    // List all operands and rebuild a right-skewed tree:
    // ex) if Opcode is BO_Add, the left-hand-side operand cannot be
    //     an add operation
    SmallVector<const SEInvariant*, 16> Operands;
    while (const SEInvariantOperation *LHSOp = dyn_cast<SEInvariantOperation>(LHS)) {
      if (LHSOp->getOpcode() == Opcode) {
        MergeInvarsOpInvar(Operands, LHSOp->getLHS(), Opcode);
        LHS = LHSOp->getRHS();
      } else {
        break;
      }
    }
    MergeInvarsOpInvar(Operands, LHS, Opcode);
    while (const SEInvariantOperation *RHSOp = dyn_cast<SEInvariantOperation>(RHS)) {
      if (RHSOp->getOpcode() == Opcode) {
        MergeInvarsOpInvar(Operands, RHSOp->getLHS(), Opcode);
        RHS = RHSOp->getRHS();
      } else {
        break;
      }
    }
    MergeInvarsOpInvar(Operands, RHS, Opcode);

    assert(!Operands.empty());
    if (Operands.size() == 1) {
      return Operands[0];
    }
    std::sort(Operands.begin(), Operands.end(), CompareInvariantPointer);
    LHS = Operands[0];
    RHS = Operands[Operands.size() - 1];
    for (unsigned Index = Operands.size() - 2; Index != 0; --Index) {
      RHS = new (*this) SEInvariantOperation(Operands[Index], RHS, Opcode);
    }
  }

  assert(LHS && RHS);
  return new (*this) SEInvariantOperation(LHS, RHS, Opcode);
}

const SEInvariant *SymbolicExprContext::MergeInvarOpInvar(
    const SEInvariant *LHS, const SEInvariant *RHS,
    BinaryOperatorKind Opcode) const {
  assert(LHS && RHS);
  const SEConstant *LHSConst = dyn_cast<SEConstant>(LHS);
  const SEConstant *RHSConst = dyn_cast<SEConstant>(RHS);

  // Constant folding
  if (LHSConst && RHSConst) {
    int64_t LHSValue = LHSConst->getValue();
    int64_t RHSValue = RHSConst->getValue();

    switch (Opcode) {
      case BO_Mul: return SEConstant::Create(*this, LHSValue * RHSValue);
      case BO_Add: return SEConstant::Create(*this, LHSValue + RHSValue);
      case BO_Sub: return SEConstant::Create(*this, LHSValue - RHSValue);
      case BO_Shr: return SEConstant::Create(*this, LHSValue >> RHSValue);
      case BO_And: return SEConstant::Create(*this, LHSValue & RHSValue);
      case BO_Xor: return SEConstant::Create(*this, LHSValue ^ RHSValue);
      case BO_Or:  return SEConstant::Create(*this, LHSValue | RHSValue);

      // Note that the result of the division (/), remainder (%), and left-shift
      // (<<) operators depend on the type of the operands.
      case BO_Div:
        if (LHSValue >= -((int64_t)1 << 31) && LHSValue < ((int64_t)1 << 31) &&
            RHSValue >= -((int64_t)1 << 31) && RHSValue < ((int64_t)1 << 31)) {
          return SEConstant::Create(*this, LHSValue / RHSValue);
        }
        break;

      case BO_Rem:
      case BO_Shl: break;

      default: llvm_unreachable("invalid operator kind");
    }
  }

  // Identity element and absorbing element
  if (LHSConst || RHSConst) {
    if (Opcode == BO_Mul || Opcode == BO_Add ||
        Opcode == BO_And || Opcode == BO_Xor || Opcode == BO_Or) {
      const SEConstant *ConstOperand = (LHSConst ? LHSConst : RHSConst);
      const SEInvariant *OtherOperand = (LHSConst ? RHS : LHS);
      int64_t ConstValue = ConstOperand->getValue();

      switch (Opcode) {
        case BO_Mul:
          if (ConstValue == 0) return ConstOperand; // A * 0 => 0
          if (ConstValue == 1) return OtherOperand; // A * 1 => A
          break;

        case BO_Add:
          if (ConstValue == 0) return OtherOperand; // A + 0 => A
          break;

        case BO_And:
          if (ConstValue == 0) return ConstOperand; // A & 0 => 0
          if (ConstValue == -1) return OtherOperand; // A & -1 => A
          break;

        case BO_Xor:
          if (ConstValue == 0) return ConstOperand; // A ^ 0 => A
          break;

        case BO_Or:
          if (ConstValue == 0) return OtherOperand; // A | 0 => A
          if (ConstValue == -1) return ConstOperand; // A | -1 => -1
          break;

        default: llvm_unreachable("invalid operator kind");
      }

    } else {
      switch (Opcode) {
        case BO_Div:
          if (LHSConst && LHSConst->getValue() == 0) return LHS; // 0 / A => 0
          if (RHSConst && RHSConst->getValue() == 1) return LHS; // A / 1 => A
          break;

        case BO_Rem:
          if (LHSConst && LHSConst->getValue() == 0) return LHS; // 0 % A => 0
          if (RHSConst && RHSConst->getValue() == 1) return this->Zero; // A % 1 => 0
          break;

        case BO_Sub: {
          if (LHSConst && LHSConst->getValue() == 0) { // 0 - A => -A
            return CreateInvarOpInvar(this->MinusOne, RHS, BO_Mul);
          }
          if (RHSConst && RHSConst->getValue() == 0) return LHS; // A - 0 => A
          break;
        }

        case BO_Shl:
        case BO_Shr:
          if (RHSConst && RHSConst->getValue() == 0) return LHS; // A << 0 => A
                                                                 // A >> 0 => A
          break;

        default: llvm_unreachable("invalid operator kind");
      }
    }
  }

  // LHS == RHS
  if ((Opcode == BO_Div || Opcode == BO_Rem || Opcode == BO_Add ||
       Opcode == BO_Sub || Opcode == BO_And || Opcode == BO_Xor ||
       Opcode == BO_Or) &&
      *LHS == *RHS) {
    switch (Opcode) {
      case BO_Div: return this->One;

      case BO_Rem:
      case BO_Sub:
      case BO_Xor: return this->Zero;

      case BO_Add: return CreateInvarOpInvar(this->Two, LHS, BO_Mul);

      case BO_And:
      case BO_Or:  return LHS;

      default: llvm_unreachable("invalid operator kind");
    }
  }

  // Distributive operator
  if (Opcode == BO_Add || Opcode == BO_Sub) {
    const SEInvariantOperation *LHSOp = dyn_cast<SEInvariantOperation>(LHS);
    const SEInvariantOperation *RHSOp = dyn_cast<SEInvariantOperation>(RHS);
    // (a * c) + (b * c) => (a + b) * c => merged(a, b) * c
    if (LHSOp && LHSOp->getOpcode() == BO_Mul &&
        RHSOp && RHSOp->getOpcode() == BO_Mul &&
        *(LHSOp->getRHS()) == *(RHSOp->getRHS())) {
      if (const SEInvariant *Merged = MergeInvarOpInvar(LHSOp->getLHS(), RHSOp->getLHS(), Opcode)) {
        return CreateInvarOpInvar(Merged, LHSOp->getRHS(), BO_Mul);
      }
    }
    // (a * c) + c => (a + 1) * c => merged(a, 1) * c
    if (LHSOp && LHSOp->getOpcode() == BO_Mul && *(LHSOp->getRHS()) == *RHS) {
      if (const SEInvariant *Merged = MergeInvarOpInvar(LHSOp->getLHS(), this->One, Opcode)) {
        return CreateInvarOpInvar(Merged, LHSOp->getRHS(), BO_Mul);
      }
    }
    // c + (b * c) => (1 + b) * c => merged(1, b) * c
    if (RHSOp && RHSOp->getOpcode() == BO_Mul && *LHS == *(RHSOp->getRHS())) {
      if (const SEInvariant *Merged = MergeInvarOpInvar(this->One, RHSOp->getLHS(), Opcode)) {
        return CreateInvarOpInvar(Merged, RHSOp->getRHS(), BO_Mul);
      }
    }
  }

  return NULL;
}

void SymbolicExprContext::MergeInvarsOpInvar(
    SmallVectorImpl<const SEInvariant*> &LHS, const SEInvariant *RHS,
    BinaryOperatorKind Opcode) const {
  for (SmallVectorImpl<const SEInvariant*>::iterator I = LHS.begin(),
                                                     E = LHS.end();
       I != E; ++I) {
    if (const SEInvariant *Merged = MergeInvarOpInvar(*I, RHS, Opcode)) {
      LHS.erase(I);
      MergeInvarsOpInvar(LHS, Merged, Opcode);
      return;
    }
  }
  LHS.push_back(RHS);
}

const SEAffine *SymbolicExprContext::CreateAffineOpAffine(
    const SEAffine *LHS, const SEAffine *RHS, BinaryOperatorKind Opcode) const {
  assert(LHS && RHS);
  const SEInvariant *LHSInvariant = dyn_cast<SEInvariant>(LHS);
  const SEInvariant *RHSInvariant = dyn_cast<SEInvariant>(RHS);

  if (LHSInvariant && RHSInvariant) {
    return CreateInvarOpInvar(LHSInvariant, RHSInvariant, Opcode);
  }
  if (Opcode == BO_Mul) {
    if (LHSInvariant || RHSInvariant) {
      const SEInvariant *InvOperand = (LHSInvariant ? LHSInvariant : RHSInvariant);
      const SEAffine *OtherOperand = (LHSInvariant ? RHS : LHS);
      SmallVector<SEAffine::AffineTerm, 8> NewTerms;
      for (unsigned Index = 0, NumTerms = OtherOperand->getNumTerms();
           Index != NumTerms; ++Index) {
        const SEInvariant *Coeff = OtherOperand->getCoefficient(Index);
        const SEVariable *Var = OtherOperand->getVariable(Index);
        const SEInvariant *NewCoeff = CreateInvarOpInvar(Coeff, InvOperand, BO_Mul);
        if (!NewCoeff->isZero()) {
          NewTerms.push_back(SEAffine::AffineTerm(NewCoeff, Var));
        }
      }
      return SEAffine::Create(*this, NewTerms);
    } else {
      return NULL;
    }
  }
  if (Opcode != BO_Add && Opcode != BO_Sub) {
    return NULL;
  }

  SmallVector<SEAffine::AffineTerm, 8> NewTerms;
  unsigned LHSIndex = 0;
  unsigned RHSIndex = 0;
  unsigned LHSNumTerms = LHS->getNumTerms();
  unsigned RHSNumTerms = RHS->getNumTerms();
  while (LHSIndex < LHSNumTerms || RHSIndex < RHSNumTerms) {
    bool UseLeft = false;
    bool UseRight = false;
    bool Merge = false;
    if (LHSIndex == LHSNumTerms) {
      UseRight = true;
    } else if (RHSIndex == RHSNumTerms) {
      UseLeft = true;
    } else {
      int VarCompare = CompareVariable(LHS->getVariable(LHSIndex),
                                       RHS->getVariable(RHSIndex));
      if (VarCompare == 0) {
        Merge = true;
      } else if (VarCompare == -1) {
        UseLeft = true;
      } else { // VarCompare == 1
        UseRight = true;
      }
    }

    if (UseLeft) {
      const SEInvariant *LHSCoeff = LHS->getCoefficient(LHSIndex);
      const SEVariable *LHSVar = LHS->getVariable(LHSIndex);
      if (!LHSCoeff->isZero()) {
        NewTerms.push_back(SEAffine::AffineTerm(LHSCoeff, LHSVar));
      }
      LHSIndex++;

    } else if (UseRight) {
      const SEInvariant *RHSCoeff = RHS->getCoefficient(RHSIndex);
      const SEVariable *RHSVar = RHS->getVariable(RHSIndex);
      if (Opcode == BO_Sub) {
        RHSCoeff = CreateInvarOpInvar(this->MinusOne, RHSCoeff, BO_Mul);
      }
      if (!RHSCoeff->isZero()) {
        NewTerms.push_back(SEAffine::AffineTerm(RHSCoeff, RHSVar));
      }
      RHSIndex++;

    } else { // Merge
      assert(Merge);
      const SEInvariant *LHSCoeff = LHS->getCoefficient(LHSIndex);
      const SEInvariant *RHSCoeff = RHS->getCoefficient(RHSIndex);
      const SEInvariant *NewCoeff = CreateInvarOpInvar(LHSCoeff, RHSCoeff, Opcode);
      const SEVariable *MergedVar = LHS->getVariable(LHSIndex);
      if (!NewCoeff->isZero()) {
        NewTerms.push_back(SEAffine::AffineTerm(NewCoeff, MergedVar));
      }
      LHSIndex++;
      RHSIndex++;
    }
  }
  return SEAffine::Create(*this, NewTerms);
}

const SEConstant *SymbolicExprContext::CreateAffineMinusAffineConst(
    const SEAffine *LHS, const SEAffine *RHS) const {
  assert(LHS && RHS);

  unsigned LHSIndex = 0;
  unsigned RHSIndex = 0;
  unsigned LHSNumTerms = LHS->getNumTerms();
  unsigned RHSNumTerms = RHS->getNumTerms();
  const SEInvariant *LHSInvariant = NULL;
  const SEInvariant *RHSInvariant = NULL;
  while (LHSIndex < LHSNumTerms && RHSIndex < RHSNumTerms) {
    const SEInvariant *LHSCoeff = LHS->getCoefficient(LHSIndex);
    const SEInvariant *RHSCoeff = RHS->getCoefficient(RHSIndex);
    const SEVariable *LHSVar = LHS->getVariable(LHSIndex);
    const SEVariable *RHSVar = RHS->getVariable(RHSIndex);
    if (LHSVar == NULL) {
      assert(LHSInvariant == NULL);
      LHSInvariant = LHSCoeff;
      LHSIndex++;
    }
    if (RHSVar == NULL) {
      assert(RHSInvariant == NULL);
      RHSInvariant = RHSCoeff;
      RHSIndex++;
    }
    if (LHSVar != NULL && RHSVar != NULL) {
      if (*LHSVar != *RHSVar || *LHSCoeff != *RHSCoeff) {
        return NULL;
      }
      LHSIndex++;
      RHSIndex++;
    }
  }
  if (LHSIndex != LHSNumTerms || RHSIndex != RHSNumTerms) {
    return NULL;
  }

  if (LHSInvariant == NULL && RHSInvariant == NULL) {
    return this->Zero;
  } else if (RHSInvariant == NULL) {
    return dyn_cast<SEConstant>(LHSInvariant);
  } else if (LHSInvariant == NULL) {
    return dyn_cast_or_null<SEConstant>(CreateInvarOpInvar(this->MinusOne, RHSInvariant, BO_Mul));
  } else {
    return dyn_cast_or_null<SEConstant>(CreateInvarOpInvar(LHSInvariant, RHSInvariant, BO_Sub));
  }
}

const SEBufferAddress *SymbolicExprContext::CreateAddrOpAffine(
    const SEBufferAddress *LHS, const SEAffine *RHS,
    BinaryOperatorKind Opcode) const {
  assert(LHS && RHS);
  if (Opcode != BO_Add && Opcode != BO_Sub) {
    return NULL;
  }
  if (const SEAffine *LHSOffset = dyn_cast<SEAffine>(LHS->getOffset())) {
    if (const SEAffine *NewOffset = CreateAffineOpAffine(LHSOffset, RHS, Opcode)) {
      return new (*this) SEBufferAddress(LHS->getIndexedBase(), NewOffset);
    }
  }
  return new (*this) SEBufferAddress(LHS->getIndexedBase(), this->Unknown);
}

const SymbolicExpr *SymbolicExprContext::CreateAddrMinusAddr(
    const SEBufferAddress *LHS, const SEBufferAddress *RHS) const {
  assert(LHS && RHS);
  if (isa<SEAffine>(LHS->getOffset()) && isa<SEAffine>(RHS->getOffset()) &&
      LHS->getIndexedBase() == RHS->getIndexedBase()) {
    const SEAffine *LHSAffine = static_cast<const SEAffine*>(LHS->getOffset());
    const SEAffine *RHSAffine = static_cast<const SEAffine*>(RHS->getOffset());
    return CreateAffineOpAffine(LHSAffine, RHSAffine, BO_Sub);
  }
  return NULL;
}

const SEConstant *SymbolicExprContext::CreateAddrMinusAddrConst(
    const SEBufferAddress *LHS, const SEBufferAddress *RHS) const {
  assert(LHS && RHS);
  if (isa<SEAffine>(LHS->getOffset()) && isa<SEAffine>(RHS->getOffset()) &&
      LHS->getIndexedBase() == RHS->getIndexedBase()) {
    const SEAffine *LHSAffine = static_cast<const SEAffine*>(LHS->getOffset());
    const SEAffine *RHSAffine = static_cast<const SEAffine*>(RHS->getOffset());
    return CreateAffineMinusAffineConst(LHSAffine, RHSAffine);
  }
  return NULL;
}

// SymbolicAnalysis

namespace {

// SymbolicAnalysisImpl

class SymbolicAnalysisImpl : public WStmtVisitor<SymbolicAnalysisImpl,
                                                 const SymbolicExpr*> {
  ASTContext &ASTCtx;
  SymbolicExprContext &SECtx;

  typedef SymbolicAnalysis::VariableValueMapTy VariableValueMapTy;
  VariableValueMapTy &VarValueMap;

public:
  SymbolicAnalysisImpl(ASTContext &C, SymbolicExprContext &SC,
                       VariableValueMapTy &VV)
    : ASTCtx(C), SECtx(SC), VarValueMap(VV) {}

#define STMT(type) \
  const SymbolicExpr *Visit##type(W##type *Node);
  CLANG_WSTMTS()
#undef STMT

#define WSTMT(type) \
  const SymbolicExpr *Visit##type(W##type *Node);
  EXTENDED_WSTMTS()
#undef WSTMT

  const SymbolicExpr *VisitRawUnaryOperator(WExpr *SubExpr,
                                            UnaryOperatorKind Opcode);
  const SymbolicExpr *VisitRawCallAddressExpr(WCallExpr *Node);
  const SymbolicExpr *VisitRawMemberDotExpr(WExpr *Base, FieldDecl *Member);

  const SymbolicExpr *VisitRawCastExpr(WCastExpr *Node);
  const SymbolicExpr *VisitRawBinaryOperator(WExpr *LHS, WExpr *RHS,
                                             BinaryOperatorKind Opcode);
  const SymbolicExpr *VisitRawAddSubInternal(WExpr *LHS, WExpr *RHS,
                                             int64_t LHSStepValue,
                                             int64_t RHSStepValue,
                                             BinaryOperatorKind Opcode);
  const SymbolicExpr *GetPatternPlusConstant(const SymbolicExpr *LHS, int64_t RHS);

  const SEInductionVariable *CreateZeroBasedInductionVariable(WBaseMuFunction *Mu);

  int64_t GetAddStep(QualType Ty, bool Minus = false);
  uint64_t GetFieldOffset(FieldDecl *Field);
  uint64_t GetFieldOffset(WMemberExpr *ME);
};

// Statements

const SymbolicExpr *SymbolicAnalysisImpl::VisitDeclStmt(WDeclStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitNullStmt(WNullStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitCompoundStmt(WCompoundStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitCaseStmt(WCaseStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitDefaultStmt(WDefaultStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitLabelStmt(WLabelStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitAttributedStmt(WAttributedStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitIfStmt(WIfStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitSwitchStmt(WSwitchStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitWhileStmt(WWhileStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitDoStmt(WDoStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitForStmt(WForStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitGotoStmt(WGotoStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitContinueStmt(WContinueStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitBreakStmt(WBreakStmt*) { return NULL; }
const SymbolicExpr *SymbolicAnalysisImpl::VisitReturnStmt(WReturnStmt*) { return NULL; }

const SymbolicExpr *SymbolicAnalysisImpl::VisitPhiFunction(WPhiFunction *Node) {
  const SymbolicExpr *Common = NULL;
  IndexedVarDecl *CommonBuffer = NULL;
  for (unsigned Index = 0, NumArgs = Node->getNumArgs();
       Index != NumArgs; ++Index) {
    IndexedVarDecl *Var = Node->getIndexedArgDecl(Index);
    assert(Var != NULL);
    if (Var->getIndex() == 0) {
      continue;
    }
    const SymbolicExpr *P = VarValueMap[Var];
    IndexedVarDecl *PBuffer = NULL;
    if (const SEBufferAddress *PAddr = dyn_cast<SEBufferAddress>(P)) {
      PBuffer = PAddr->getIndexedBase();
    }
    if (Common == NULL) {
      Common = VarValueMap[Var];
      CommonBuffer = PBuffer;
    } else {
      if (*Common != *P) {
        Common = SECtx.Unknown;
      }
      if (CommonBuffer != PBuffer) {
        CommonBuffer = NULL;
      }
    }
  }
  assert(Common != NULL);
  if (!Common->isUnknown()) {
    return Common;
  } else if (CommonBuffer) {
    return new (SECtx) SEBufferAddress(CommonBuffer, SECtx.Unknown);
  } else {
    return SECtx.Unknown;
  }
}

// Expressions

#define RETURN_VARIABLE_IF_EXIST(var) \
  VariableValueMapTy::const_iterator I = VarValueMap.find(var); \
  if (I != VarValueMap.end()) { \
    return I->second; \
  }
#define RETURN_EXPR_IF_KNOWN(expr) \
  if (const SymbolicExpr *Result = (expr)) { \
    return Result; \
  }

const SymbolicExpr *SymbolicAnalysisImpl::VisitConstantExpr(
    WConstantExpr *Node) {
  return Visit(Node->getSubExpr());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitOpaqueValueExpr(
    WOpaqueValueExpr* Node) {
  return Visit(Node->getSourceExpr());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitDeclRefExpr(WDeclRefExpr* Node) {
  IndexedVarDeclRef SSAVarRef = Node->getIndexedUseDecl();
  if (SSAVarRef.isSingleVar()) {
    RETURN_VARIABLE_IF_EXIST(*SSAVarRef);
  }
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitPredefinedExpr(
    WPredefinedExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitIntegerLiteral(
    WIntegerLiteral *Node) {
  RETURN_EXPR_IF_KNOWN(SEConstant::Create(SECtx, Node->getValue()));
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCharacterLiteral(
    WCharacterLiteral *Node) {
  RETURN_EXPR_IF_KNOWN(SEConstant::Create(SECtx, Node->getValue()));
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitFloatingLiteral(
    WFloatingLiteral *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitStringLiteral(
    WStringLiteral *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitParenExpr(WParenExpr *Node) {
  return Visit(Node->getSubExpr());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitRawUnaryOperator(
    WExpr *SubExpr, UnaryOperatorKind Opcode) {
  switch (Opcode) {
  case UO_PostInc:
  case UO_PostDec:
  case UO_Plus:
    return Visit(SubExpr);

  case UO_PreInc:
  case UO_PreDec: {
    const SymbolicExpr *SubPattern = Visit(SubExpr);
    int64_t Step = GetAddStep(SubExpr->getType(), (Opcode == UO_PreDec));
    return GetPatternPlusConstant(SubPattern, Step);
  }

  case UO_Minus: {
    const SymbolicExpr *SubPattern = Visit(SubExpr);
    if (const SEAffine *SubAffine = dyn_cast<SEAffine>(SubPattern)) {
      RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(SECtx.MinusOne, SubAffine, BO_Mul));
    }
    return SECtx.Unknown;
  }

  case UO_AddrOf: {
    SubExpr = SubExpr->IgnoreParenCasts();
    // &*A == A
    if (WUnaryOperator *UO = dyn_cast<WUnaryOperator>(SubExpr)) {
      if (UO->getOpcode() == UO_Deref) {
        return Visit(UO->getSubExpr());
      } else {
        return SECtx.Unknown;
      }

    // &A[B] == A + B
    } else if (WArraySubscriptExpr *ASE = dyn_cast<WArraySubscriptExpr>(SubExpr)) {
      return VisitRawBinaryOperator(ASE->getLHS(), ASE->getRHS(), BO_Add);

    // &(A.p) == &A + offset(p)
    // &(A->p) == A + offset(p)
    } else if (WMemberExpr *ME = dyn_cast<WMemberExpr>(SubExpr)) {
      const SymbolicExpr *BasePattern;
      if (ME->isArrow()) {
        BasePattern = Visit(ME->getBase());
      } else {
        BasePattern = VisitRawUnaryOperator(ME->getBase(), UO_AddrOf);
      }
      uint64_t FieldOffset = GetFieldOffset(ME);
      return GetPatternPlusConstant(BasePattern, FieldOffset);
      return SECtx.Unknown;

    } else {
      return SECtx.Unknown;
    }
  }

  case UO_Deref:
    llvm_unreachable("UO_Deref cannot be analyzed in a raw form");

  default:
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitUnaryOperator(
    WUnaryOperator *Node) {
  // UO_Deref requires type information
  if (Node->getOpcode() != UO_Deref) {
    return VisitRawUnaryOperator(Node->getSubExpr(), Node->getOpcode());
  } else {
    WExpr *SubExpr = Node->getSubExpr()->IgnoreParenCasts();
    const SymbolicExpr *SubPattern = Visit(SubExpr);
    if (const SEBufferAddress *SubAddr = dyn_cast<SEBufferAddress>(SubPattern)) {
      return new (SECtx) SEBufferValue(SECtx, SubAddr, Node->getType());
    }
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitUnaryExprOrTypeTraitExpr(
    WUnaryExprOrTypeTraitExpr *Node) {
  Expr::EvalResult RetResult;
  if (Node->EvaluateAsInt(RetResult, ASTCtx)) {
    RETURN_EXPR_IF_KNOWN(SEConstant::Create(SECtx, RetResult.Val.getInt()));
  }
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitArraySubscriptExpr(
    WArraySubscriptExpr *Node) {
  // A[B] == A + B
  if (Node->getType()->isArrayType()) {
    return VisitRawBinaryOperator(Node->getLHS(), Node->getRHS(), BO_Add);
  // A[B] == *(A + B)
  } else {
    const SymbolicExpr *SubPattern = VisitRawBinaryOperator(Node->getLHS(), Node->getRHS(), BO_Add);
    if (const SEBufferAddress *SubAddr = dyn_cast<SEBufferAddress>(SubPattern)) {
      return new (SECtx) SEBufferValue(SECtx, SubAddr, Node->getType());
    }
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCallExpr(WCallExpr *Node) {
  if (Node->isAtomic()) {
    if (Node->getNumArgs() < 2) {
      return SECtx.Unknown;
    }
    WExpr *Arg = Node->getArg(0)->IgnoreParenCasts();
    const SymbolicExpr *ArgPattern = Visit(Arg);
    if (const SEBufferAddress *ArgAddr = dyn_cast<SEBufferAddress>(ArgPattern)) {
      return new (SECtx) SEBufferValue(SECtx, ArgAddr, Node->getType());
    }
    return SECtx.Unknown;

  } else {
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitRawCallAddressExpr(
    WCallExpr *Node) {
  if (Node->isAtomic()) {
    if (Node->getNumArgs() < 2) {
      return SECtx.Unknown;
    }
    WExpr *Arg = Node->getArg(0)->IgnoreParenCasts();
    const SymbolicExpr *ArgPattern = Visit(Arg);
    if (const SEBufferAddress *ArgAddr = dyn_cast<SEBufferAddress>(ArgPattern)) {
      return ArgAddr;
    }
    return SECtx.Unknown;

  } else if (Node->isVectorLoad()) {
    if (Node->getNumArgs() != 2) {
      return SECtx.Unknown;
    }
    WExpr *OffsetArg = Node->getArg(0);
    WExpr *PointerArg = Node->getArg(1);
    int64_t Step = GetAddStep(PointerArg->getType()) * Node->getVectorLoadWidth();
    return VisitRawAddSubInternal(PointerArg, OffsetArg, Step, 1, BO_Add);

  } else if (Node->isVectorStore()) {
    if (Node->getNumArgs() != 3) {
      return SECtx.Unknown;
    }
    WExpr *OffsetArg = Node->getArg(1);
    WExpr *PointerArg = Node->getArg(2);
    int64_t Step = GetAddStep(PointerArg->getType()) * Node->getVectorStoreWidth();
    return VisitRawAddSubInternal(PointerArg, OffsetArg, Step, 1, BO_Sub);

  } else {
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitRawMemberDotExpr(
    WExpr *Base, FieldDecl *Member) {
  Base = Base->IgnoreParenCasts();
  if (WDeclRefExpr *BaseDeclRef = dyn_cast<WDeclRefExpr>(Base)) {
    IndexedVarDeclRef SSAVarRef = BaseDeclRef->getIndexedUseDecl();
    if (SSAVarRef.isCompound()) {
      for (unsigned Index = 0, NumSubVars = SSAVarRef.getNumSubVars();
           Index != NumSubVars; ++Index) {
        IndexedVarDecl *SSAVar = SSAVarRef[Index];
        if (WSubVarDecl *Var = dyn_cast<WSubVarDecl>(SSAVar->getDecl())) {
          if (Var->getField().Field == Member) {
            RETURN_VARIABLE_IF_EXIST(SSAVar);
            break;
          }
        }
      }
    }
  }

  const SymbolicExpr *BasePattern = VisitRawUnaryOperator(Base, UO_AddrOf);
  uint64_t FieldOffset = GetFieldOffset(Member);
  const SymbolicExpr *SubPattern = GetPatternPlusConstant(BasePattern, FieldOffset);
  // A.p == &A + offset(p)
  if (Member->getType()->isArrayType()) {
    return SubPattern;
  // A.p == *&(A.p) == *(&A + offset(p))
  } else {
    if (const SEBufferAddress *SubAddr = dyn_cast<SEBufferAddress>(SubPattern)) {
      return new (SECtx) SEBufferValue(SECtx, SubAddr, Member->getType());
    } else {
      return SECtx.Unknown;
    }
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitMemberExpr(WMemberExpr *Node) {
  if (Node->isArrow()) {
    // A->p == A + offset(p)
    if (Node->getType()->isArrayType()) {
      const SymbolicExpr *BasePattern = Visit(Node->getBase());
      uint64_t FieldOffset = GetFieldOffset(Node);
      return GetPatternPlusConstant(BasePattern, FieldOffset);
    // A->p == *&(A->p) == *(A + offset(p))
    } else {
      const SymbolicExpr *SubPattern = VisitRawUnaryOperator(Node, UO_AddrOf);
      if (const SEBufferAddress *SubAddr = dyn_cast<SEBufferAddress>(SubPattern)) {
        return new (SECtx) SEBufferValue(SECtx, SubAddr, Node->getType());
      }
      return SECtx.Unknown;
    }

  } else {
    IndexedVarDeclRef SSAVarRef = Node->getIndexedUseDecl();
    if (SSAVarRef.isSingleVar()) {
      RETURN_VARIABLE_IF_EXIST(*SSAVarRef);
    }
    return VisitRawMemberDotExpr(Node->getBase(), Node->getMemberDecl());
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCompoundLiteralExpr(
    WCompoundLiteralExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitRawCastExpr(WCastExpr *Node) {
  QualType FromType = Node->getSubExpr()->getType();
  QualType ToType = Node->getType();
  if ((FromType->isIntegerType() || FromType->isPointerType() ||
       FromType->isArrayType()) &&
      (ToType->isIntegerType() || ToType->isPointerType() ||
       ToType->isArrayType())) {
    return Visit(Node->getSubExpr());
  }
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitImplicitCastExpr(
    WImplicitCastExpr *Node) {
  return VisitRawCastExpr(Node);
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCStyleCastExpr(
    WCStyleCastExpr *Node) {
  return VisitRawCastExpr(Node);
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitRawBinaryOperator(
    WExpr *LHS, WExpr *RHS, BinaryOperatorKind Opcode) {
  switch (Opcode) {
  case BO_Add:
  case BO_Sub: {
    return VisitRawAddSubInternal(LHS, RHS, GetAddStep(LHS->getType()),
                                  GetAddStep(RHS->getType()), Opcode);
  }

  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Shl:
  case BO_Shr:
  case BO_And:
  case BO_Xor:
  case BO_Or: {
    const SymbolicExpr *LHSPattern = Visit(LHS);
    const SymbolicExpr *RHSPattern = Visit(RHS);
    if (isa<SEAffine>(LHSPattern) && isa<SEAffine>(RHSPattern)) {
      const SEAffine *LHSAffine = static_cast<const SEAffine*>(LHSPattern);
      const SEAffine *RHSAffine = static_cast<const SEAffine*>(RHSPattern);
      RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(LHSAffine, RHSAffine, Opcode));
    }
    return SECtx.Unknown;
  }

  case BO_Assign:
  case BO_Comma:
    return Visit(RHS);

  default:
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitRawAddSubInternal(
    WExpr *LHS, WExpr *RHS, int64_t LHSStepValue, int64_t RHSStepValue,
    BinaryOperatorKind Opcode) {
  assert(Opcode == BO_Add || Opcode == BO_Sub);
  const SymbolicExpr *LHSPattern = Visit(LHS);
  const SymbolicExpr *RHSPattern = Visit(RHS);
  const SEAffine *LHSAffine = dyn_cast<SEAffine>(LHSPattern);
  const SEAffine *RHSAffine = dyn_cast<SEAffine>(RHSPattern);
  const SEBufferAddress *LHSAddr = dyn_cast<SEBufferAddress>(LHSPattern);
  const SEBufferAddress *RHSAddr = dyn_cast<SEBufferAddress>(RHSPattern);

  if (LHSStepValue > 1 && RHSStepValue > 1) {
    assert(Opcode == BO_Sub);
    assert(LHSStepValue == RHSStepValue);
    const SEAffine *SubAffine = NULL;
    if (LHSAffine && RHSAffine) {
      SubAffine = SECtx.CreateAffineOpAffine(LHSAffine, RHSAffine, BO_Sub);
    } else if (LHSAddr && RHSAddr) {
      const SymbolicExpr *Sub = SECtx.CreateAddrMinusAddr(LHSAddr, RHSAddr);
      SubAffine = dyn_cast_or_null<SEAffine>(Sub);
    }
    if (!SubAffine) return SECtx.Unknown;
    const SEConstant *Step = SEConstant::Create(SECtx, LHSStepValue);
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(SubAffine, Step, BO_Div));
    return SECtx.Unknown;
  }

  if ((LHSAffine || LHSAddr) && (RHSAffine || RHSAddr)) {
    if (LHSStepValue > 1 && RHSAffine) {
      const SEConstant *Step = SEConstant::Create(SECtx, LHSStepValue);
      RHSAffine = SECtx.CreateAffineOpAffine(RHSAffine, Step, BO_Mul);
    }
    if (RHSStepValue > 1 && LHSAffine) {
      const SEConstant *Step = SEConstant::Create(SECtx, RHSStepValue);
      LHSAffine = SECtx.CreateAffineOpAffine(LHSAffine, Step, BO_Mul);
    }

    if (LHSAffine && RHSAffine) {
      RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(LHSAffine, RHSAffine, Opcode));
    } else if (LHSAddr && RHSAffine) {
      RETURN_EXPR_IF_KNOWN(SECtx.CreateAddrOpAffine(LHSAddr, RHSAffine, Opcode));
    } else if (RHSAddr && LHSAffine && Opcode == BO_Add) {
      RETURN_EXPR_IF_KNOWN(SECtx.CreateAddrOpAffine(RHSAddr, LHSAffine, Opcode));
    }
    return SECtx.Unknown;
  }

  if (LHSAddr) {
    return new (SECtx) SEBufferAddress(LHSAddr->getIndexedBase(), SECtx.Unknown);
  } else if (RHSAddr && Opcode == BO_Add) {
    return new (SECtx) SEBufferAddress(RHSAddr->getIndexedBase(), SECtx.Unknown);
  } else {
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitBinaryOperator(
    WBinaryOperator *Node) {
  Expr::EvalResult RetResult;
  if (Node->EvaluateAsInt(RetResult, ASTCtx)) {
    RETURN_EXPR_IF_KNOWN(SEConstant::Create(SECtx, RetResult.Val.getInt()));
    return SECtx.Unknown;
  }
  return VisitRawBinaryOperator(Node->getLHS(), Node->getRHS(), Node->getOpcode());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCompoundAssignOperator(
    WCompoundAssignOperator *Node) {
  BinaryOperatorKind NewOpcode =
    BinaryOperator::getOpForCompoundAssignment(Node->getOpcode());
  return VisitRawBinaryOperator(Node->getLHS(), Node->getRHS(), NewOpcode);
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitConditionalOperator(
    WConditionalOperator *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitInitListExpr(
    WInitListExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitDesignatedInitExpr(
    WDesignatedInitExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitImplicitValueInitExpr(
    WImplicitValueInitExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitParenListExpr(
    WParenListExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitExtVectorElementExpr(
    WExtVectorElementExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitPseudoObjectExpr(
    WPseudoObjectExpr *Node) {
  return Visit(Node->getResultExpr());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXOperatorCallExpr(
    WCXXOperatorCallExpr *Node) {
  return VisitCallExpr(Node);
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXMemberCallExpr(
    WCXXMemberCallExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXBoolLiteralExpr(
    WCXXBoolLiteralExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXDefaultArgExpr(
    WCXXDefaultArgExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXConstructExpr(
    WCXXConstructExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXFunctionalCastExpr(
    WCXXFunctionalCastExpr *Node) {
  return VisitRawCastExpr(Node);
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitExprWithCleanups(
    WExprWithCleanups *Node) {
  return Visit(Node->getSubExpr());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitMaterializeTemporaryExpr(
    WMaterializeTemporaryExpr *Node) {
  return Visit(Node->getSubExpr());
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitMSPropertyRefExpr(
    WMSPropertyRefExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitCXXThisExpr(
    WCXXThisExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitBlockExpr(WBlockExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitAsTypeExpr(WAsTypeExpr *Node) {
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitWorkItemFunction(
    WWorkItemFunction *Node) {
  switch (Node->getFunctionKind()) {
  case WWorkItemFunction::WIF_get_local_size:
    return SECtx.LocalSize[Node->getArg()];

  case WWorkItemFunction::WIF_get_num_groups:
    return SECtx.NumGroups[Node->getArg()];

  case WWorkItemFunction::WIF_get_global_size: {
    const SENDRangeDimension *LS = SECtx.LocalSize[Node->getArg()];
    const SENDRangeDimension *NG = SECtx.NumGroups[Node->getArg()];
    const SymbolicExpr *Result = SECtx.CreateInvarOpInvar(LS, NG, BO_Mul);
    assert(Result != NULL);
    return Result;
  }

  case WWorkItemFunction::WIF_get_local_id:
    return SECtx.LocalID[Node->getArg()];

  case WWorkItemFunction::WIF_get_group_id:
    return SECtx.GroupID[Node->getArg()];

  case WWorkItemFunction::WIF_get_global_id: {
    const SENDRangeIndex *LID = SECtx.LocalID[Node->getArg()];
    const SENDRangeIndex *GID = SECtx.GroupID[Node->getArg()];
    const SENDRangeDimension *LS = SECtx.LocalSize[Node->getArg()];
    // GID * LS + LID
    const SymbolicExpr *Result = SECtx.CreateAffineOpAffine(
        SECtx.CreateAffineOpAffine(GID, LS, BO_Mul), LID, BO_Add);
    assert(Result != NULL);
    return Result;
  }

  default:
    return SECtx.Unknown;
  }
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitBaseMuFunction(
    WBaseMuFunction *Node) {
  IndexedVarDecl *Init = Node->getInitVar();
  assert(Init != NULL);

  const SymbolicExpr *InitPattern = VarValueMap[Init];
  const SEInductionVariable *IV = CreateZeroBasedInductionVariable(Node);
  if (!IV) return SECtx.Unknown;

  if (const SEAffine *InitAffine = dyn_cast<SEAffine>(InitPattern)) {
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(InitAffine, IV, BO_Add));
  } else if (const SEBufferAddress *InitAddr = dyn_cast<SEBufferAddress>(InitPattern)) {
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAddrOpAffine(InitAddr, IV, BO_Add));
  }
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::VisitAuxiliaryMuFunction(
    WAuxiliaryMuFunction *Node) {
  WBaseMuFunction *Base = Node->getBase();
  IndexedVarDecl *Init = Node->getInitVar();
  WExpr *Step = Node->getStep();
  bool Increment = Node->isIncrement();
  assert(Base != NULL && Init != NULL && Step != NULL);

  const SymbolicExpr *InitPattern = VarValueMap[Init];
  if (InitPattern->isUnknown()) return SECtx.Unknown;
  const SEInductionVariable *IV = CreateZeroBasedInductionVariable(Base);
  if (!IV) return SECtx.Unknown;

  const SymbolicExpr *StepPattern = (Increment ? Visit(Step) : VisitRawUnaryOperator(Step, UO_Minus));
  const SEInvariant *StepInvariant = dyn_cast<SEInvariant>(StepPattern);
  if (!StepInvariant) return SECtx.Unknown;

  int64_t StepWeight = GetAddStep(Init->getType());
  if (StepWeight > 1) {
    StepInvariant = SECtx.CreateInvarOpInvar(StepInvariant, SEConstant::Create(SECtx, StepWeight), BO_Mul);
    assert(StepInvariant);
  }
  // We call MergeInvarOpInvar() instead of CreateInvarOpInvar().
  // (IV*S1)/S2 is not equal to IV*(S1/S2) unless S1/S2 can be merged.
  StepInvariant = SECtx.MergeInvarOpInvar(StepInvariant, IV->getStep(), BO_Div);
  if (!StepInvariant) return SECtx.Unknown;
  const SEAffine *WeightedIV = SECtx.CreateAffineOpAffine(IV, StepInvariant, BO_Mul);
  assert(WeightedIV != NULL);

  if (const SEAffine *InitAffine = dyn_cast<SEAffine>(InitPattern)) {
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(InitAffine, WeightedIV, BO_Add));
  } else if (const SEBufferAddress *InitAddr = dyn_cast<SEBufferAddress>(InitPattern)) {
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAddrOpAffine(InitAddr, WeightedIV, BO_Add));
  }
  return SECtx.Unknown;
}

const SymbolicExpr *SymbolicAnalysisImpl::GetPatternPlusConstant(
    const SymbolicExpr *LHS, int64_t RHS) {
  assert(LHS);
  if (const SEAffine *LHSAffine = dyn_cast<SEAffine>(LHS)) {
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAffineOpAffine(LHSAffine, SEConstant::Create(SECtx, RHS), BO_Add));
  } else if (const SEBufferAddress *LHSAddr = dyn_cast<SEBufferAddress>(LHS)) {
    RETURN_EXPR_IF_KNOWN(SECtx.CreateAddrOpAffine(LHSAddr, SEConstant::Create(SECtx, RHS), BO_Add));
  }
  return SECtx.Unknown;
}

const SEInductionVariable *SymbolicAnalysisImpl::CreateZeroBasedInductionVariable(
    WBaseMuFunction *Mu) {
  WStmt *Loop = Mu->getLoop();
  IndexedVarDecl *Init = Mu->getInitVar();
  WExpr *Step = Mu->getStep();
  bool Increment = Mu->isIncrement();
  WExpr *Bound = Mu->getBound();
  bool Inclusive = Mu->isBoundInclusive();
  assert(Loop != NULL && Init != NULL && Step != NULL && Bound != NULL);

  const SymbolicExpr *InitPattern = VarValueMap[Init];
  if (InitPattern->isUnknown()) return NULL;
  const SymbolicExpr *StepPattern = (Increment ? Visit(Step) : VisitRawUnaryOperator(Step, UO_Minus));
  const SEInvariant *StepInvariant = dyn_cast<SEInvariant>(StepPattern);
  if (!StepInvariant) return NULL;
  const SymbolicExpr *BoundPattern = Visit(Bound);
  if (BoundPattern->isUnknown()) return NULL;

  int64_t StepWeight = GetAddStep(Init->getType());
  if (StepWeight > 1) {
    StepInvariant = SECtx.CreateInvarOpInvar(StepInvariant, SEConstant::Create(SECtx, StepWeight), BO_Mul);
    assert(StepInvariant);
  }

  if (isa<SEAffine>(InitPattern) && isa<SEAffine>(BoundPattern)) {
    const SEAffine *InitAffine = static_cast<const SEAffine*>(InitPattern);
    const SEAffine *BoundAffine = static_cast<const SEAffine*>(BoundPattern);
    const SEAffine *ZeroBasedBound = SECtx.CreateAffineOpAffine(BoundAffine, InitAffine, BO_Sub);
    assert(ZeroBasedBound != NULL);
    return new (SECtx) SEInductionVariable(SECtx, Loop, ZeroBasedBound, StepInvariant, Inclusive);

  } else if (isa<SEBufferAddress>(InitPattern) && isa<SEBufferAddress>(BoundPattern)) {
    const SEBufferAddress *InitAddr = static_cast<const SEBufferAddress*>(InitPattern);
    const SEBufferAddress *BoundAddr = static_cast<const SEBufferAddress*>(BoundPattern);
    const SymbolicExpr *ZeroBasedBound = SECtx.CreateAddrMinusAddr(InitAddr, BoundAddr);
    assert(ZeroBasedBound != NULL);
    if (const SEAffine *ZeroBasedAffine = dyn_cast<SEAffine>(ZeroBasedBound)) {
      return new (SECtx) SEInductionVariable(SECtx, Loop, ZeroBasedAffine, StepInvariant, Inclusive);
    } else {
      return NULL;
    }

  } else {
    return NULL;
  }
}

int64_t SymbolicAnalysisImpl::GetAddStep(QualType Ty, bool Minus) {
  int Step = 1;
  if (Ty->isPointerType()) {
    Step = ASTCtx.getTypeSizeInChars(Ty->getPointeeType()).getQuantity();
  }
  return (Minus ? -Step : Step);
}

uint64_t SymbolicAnalysisImpl::GetFieldOffset(FieldDecl *F) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(F->getParent());
  return Layout.getFieldOffset(F->getFieldIndex()) / ASTCtx.getCharWidth();
}

uint64_t SymbolicAnalysisImpl::GetFieldOffset(WMemberExpr *ME) {
  return GetFieldOffset(ME->getMemberDecl());
}

} // anonymous namespace

void SymbolicAnalysis::InitVariable(IndexedVarDecl *Var, const SymbolicExpr *P) {
  assert(P != NULL);
  VarValueMap[Var] = P;
}

bool SymbolicAnalysis::UpdateVariable(IndexedVarDecl *Var, const SymbolicExpr *P) {
  assert(P != NULL);
  assert(VarValueMap.count(Var));
  const SymbolicExpr *Old = VarValueMap[Var];
  if (*Old != *P) {
    VarValueMap[Var] = P;
    return true;
  } else {
    return false;
  }
}

const SymbolicExpr *SymbolicAnalysis::GetValueOf(WExpr *E) {
  // E
  SymbolicAnalysisImpl Impl(ASTCtx, SECtx, VarValueMap);
  return Impl.Visit(E);
}

const SymbolicExpr *SymbolicAnalysis::GetFieldValueOf(WExpr *E, FieldDecl *F) {
  // E.F
  SymbolicAnalysisImpl Impl(ASTCtx, SECtx, VarValueMap);
  return Impl.VisitRawMemberDotExpr(E, F);
}

const SymbolicExpr *SymbolicAnalysis::GetAddressOf(WExpr *E) {
  // &E
  SymbolicAnalysisImpl Impl(ASTCtx, SECtx, VarValueMap);
  if (WCallExpr *CE = dyn_cast<WCallExpr>(E)) {
    return Impl.VisitRawCallAddressExpr(CE);
  } else {
    return Impl.VisitRawUnaryOperator(E, UO_AddrOf);
  }
}

const SymbolicExpr *SymbolicAnalysis::GetFieldAddressOf(WExpr *E, FieldDecl *F) {
  // &(E.F) == &E + offset(F)
  SymbolicAnalysisImpl Impl(ASTCtx, SECtx, VarValueMap);
  const SymbolicExpr *BasePattern = Impl.VisitRawUnaryOperator(E, UO_AddrOf);
  return Impl.GetPatternPlusConstant(BasePattern, Impl.GetFieldOffset(F));
}

const SymbolicExpr *SymbolicAnalysis::GetAssignedValueOf(WStmt *S) {
  SymbolicAnalysisImpl Impl(ASTCtx, SECtx, VarValueMap);
  if (WDeclStmt *DS = dyn_cast<WDeclStmt>(S)) {
    assert(DS->hasSingleInit());
    return Impl.Visit(DS->getSingleInit());

  } else if (WUnaryOperator *UO = dyn_cast<WUnaryOperator>(S)) {
    assert(UO->isIncrementDecrementOp());
    return Impl.VisitRawUnaryOperator(
        UO->getSubExpr(), (UO->isIncrementOp() ? UO_PreInc : UO_PreDec));

  } else if (WCompoundAssignOperator *CAO =
               dyn_cast<WCompoundAssignOperator>(S)) {
    return Impl.Visit(CAO);

  } else if (WBinaryOperator *BO = dyn_cast<WBinaryOperator>(S)) {
    assert(BO->isAssignmentOp());
    return Impl.Visit(BO->getRHS());

  } else if (WPhiFunction *PF = dyn_cast<WPhiFunction>(S)) {
    return Impl.Visit(PF);

  } else {
    llvm_unreachable("invalid assignment statement");
  }
  return NULL;
}

const SymbolicExpr *SymbolicAnalysis::GetFieldAssignedValueOf(WStmt *S, FieldDecl *F) {
  SymbolicAnalysisImpl Impl(ASTCtx, SECtx, VarValueMap);
  if (WDeclStmt *DS = dyn_cast<WDeclStmt>(S)) {
    assert(DS->hasSingleInit());
    return Impl.VisitRawMemberDotExpr(DS->getSingleInit(), F);

  } else if (isa<WUnaryOperator>(S)) {
    llvm_unreachable("struct cannot be an operand of a unary operator");

  } else if (isa<WCompoundAssignOperator>(S)) {
    llvm_unreachable("struct cannot be an operand of a binary operator");

  } else if (WBinaryOperator *BO = dyn_cast<WBinaryOperator>(S)) {
    assert(BO->isAssignmentOp());
    return Impl.VisitRawMemberDotExpr(BO->getRHS(), F);

  } else if (isa<WPhiFunction>(S)) {
    llvm_unreachable("struct cannot be an operand of a phi function");

  } else {
    llvm_unreachable("invalid assignment statement");
  }
}

} // namespace snu

} // namespace clang
