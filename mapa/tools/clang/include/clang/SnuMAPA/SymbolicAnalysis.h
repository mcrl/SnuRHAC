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

#ifndef LLVM_CLANG_SNU_MAPA_SYMBOLICANALYSIS_H
#define LLVM_CLANG_SNU_MAPA_SYMBOLICANALYSIS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WAST.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

namespace snu {

// Canonical Symbolic Expression:
//   Invariant
//     Constant
//     Parameter
//     NDRangeDimension
//     InvariantOperation: Invariant op Invariant
//       [Note] op: BO_Add, BO_Mul, BO_Div, BO_Rem, BO_Shl, BO_Shr, 
//                  BO_And, BO_Xor, or BO_Or
//   Variable
//     NDRangeIndex
//     InductionVariable
//     BufferValue
//   Affine
//     Invariant
//     Variable
//     ComplexAffine: AffineTerm + AffineTerm + ... + AffineTerm
//       [Note] AffineTerm: Invariant * Variable
//   BufferAddress: Parameter + Affine
//   Unknown

#define SYMBOLIC_EXPRS() \
  EX(Constant) \
  EX(Parameter) \
  EX(NDRangeDimension) \
  EX(InvariantOperation) \
  EX(NDRangeIndex) \
  EX(InductionVariable) \
  EX(BufferValue) \
  EX(ComplexAffine) \
  EX(BufferAddress) \

class SymbolicExprContext;

class SymbolicExpr {
public:
  enum SymbolicExprClass {
#define EX(type) type##Class,
    SYMBOLIC_EXPRS()
#undef EX
    UnknownClass,

    firstAffineClass = ConstantClass,
    lastAffineClass = ComplexAffineClass,
    firstInvariantClass = ConstantClass,
    lastInvariantClass = InvariantOperationClass,
    firstVariableClass = NDRangeIndexClass,
    lastVariableClass = BufferValueClass
  };

protected:
  void* operator new(size_t bytes) throw() {
    llvm_unreachable("SymbolicExprs cannot be allocated with regular 'new'.");
  }
  void operator delete(void*) throw() {
    llvm_unreachable("SymbolicExprs cannot be released with regular 'delete'.");
  }

public:
  void* operator new(size_t bytes, const SymbolicExprContext &C,
                     unsigned alignment = 8);
  void* operator new(size_t bytes, void* mem) throw() {
    return mem;
  }

  void operator delete(void*, const SymbolicExprContext&, unsigned) throw() {}
  void operator delete(void*, void*) throw() {}

private:
  SymbolicExprClass Class;

protected:
  explicit SymbolicExpr(SymbolicExprClass PC) : Class(PC) {}

public:
  SymbolicExprClass getClass() const { return Class; }
  bool isUnknown() const { return Class == UnknownClass; }

  // Defines the total order of symbolic expressions
  int Compare(const SymbolicExpr &P) const;

  bool operator==(const SymbolicExpr &P) const {
    return Compare(P) == 0;
  }
  bool operator!=(const SymbolicExpr &P) const {
    return Compare(P) != 0;
  }
  bool operator<(const SymbolicExpr &P) const {
    return Compare(P) < 0;
  }
  bool operator>(const SymbolicExpr &P) const {
    return Compare(P) > 0;
  }
  bool operator<=(const SymbolicExpr &P) const {
    return Compare(P) <= 0;
  }
  bool operator>=(const SymbolicExpr &P) const {
    return Compare(P) >= 0;
  }

  void print(raw_ostream &OS) const;

  static const SymbolicExpr *CreateUnknown(const SymbolicExprContext &Ctx);
};

class SEInvariant;
class SEVariable;
class SEBufferAddress;

class SEAffine : public SymbolicExpr {
public:
  class AffineTerm {
  public:
    const SEInvariant *Coeff;
    const SEVariable *Var;

    AffineTerm()
      : Coeff(NULL), Var(NULL) {}
    AffineTerm(const SEInvariant *coeff, const SEVariable *var)
      : Coeff(coeff), Var(var) {}
  };

protected:
  explicit SEAffine(SymbolicExprClass PC) : SymbolicExpr(PC) {}

public:
  virtual unsigned getNumTerms() const = 0;
  virtual const SEInvariant *getCoefficient(unsigned Index) const = 0;
  virtual const SEVariable *getVariable(unsigned Index) const = 0;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() >= firstAffineClass &&
           P->getClass() <= lastAffineClass;
  }
  static const SEAffine *Create(const SymbolicExprContext &Ctx,
                                ArrayRef<AffineTerm> terms);
};

class SEInvariant : public SEAffine {
protected:
  explicit SEInvariant(SymbolicExprClass PC) : SEAffine(PC) {}

public:
  virtual unsigned getNumTerms() const { return 1; }
  virtual const SEInvariant *getCoefficient(unsigned Index) const {
    assert(Index == 0);
    return this;
  }
  virtual const SEVariable *getVariable(unsigned Index) const {
    assert(Index == 0);
    return NULL;
  }
  bool isZero() const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() >= firstInvariantClass &&
           P->getClass() <= lastInvariantClass;
  }
};

class SEConstant : public SEInvariant {
  int64_t Value;

  explicit SEConstant(int64_t value)
    : SEInvariant(ConstantClass), Value(value) {}

public:
  int64_t getValue() const { return Value; }
  void setValue(int64_t V) { Value = V; }
  bool isZero() const { return Value == 0; }

  int Compare(const SEConstant &P) const {
    if (Value < P.Value) {
      return -1;
    } else if (Value == P.Value) {
      return 0;
    } else {
      return 1;
    }
  }

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == ConstantClass;
  }
  static const SEConstant *Create(const SymbolicExprContext &Ctx,
                                  int64_t Value);
  static const SEConstant *Create(const SymbolicExprContext &Ctx,
                                  const llvm::APInt &Value);

  friend class SymbolicExprContext;
};

class SEParameter : public SEInvariant {
  IndexedVarDecl *IParam;

public:
  explicit SEParameter(IndexedVarDecl *param)
    : SEInvariant(ParameterClass), IParam(param) {}

  WParmVarDecl *getDecl() const { return IParam->getDecl()->getAsParameter(); }
  IndexedVarDecl *getIndexedDecl() const { return IParam; }

  int Compare(const SEParameter &P) const {
    if (IParam == P.IParam) {
      return 0;
    } else {
      return getDecl()->getName().compare(P.getDecl()->getName());
    }
  }

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == ParameterClass;
  }
};

class SENDRangeDimension : public SEInvariant {
public:
  enum DimensionKind {
    LOCAL_SIZE_0,
    LOCAL_SIZE_1,
    LOCAL_SIZE_2,
    NUM_GROUPS_0,
    NUM_GROUPS_1,
    NUM_GROUPS_2
  };

private:
  DimensionKind Kind;

public:
  explicit SENDRangeDimension(DimensionKind kind)
    : SEInvariant(NDRangeDimensionClass), Kind(kind) {}

  DimensionKind getKind() const { return Kind; }

  int Compare(const SENDRangeDimension &P) const {
    if (Kind < P.Kind) {
      return -1;
    } else if (Kind == P.Kind) {
      return 0;
    } else {
      return 1;
    }
  }

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == NDRangeDimensionClass;
  }
};

class SEInvariantOperation : public SEInvariant {
  const SEInvariant *LHS;
  const SEInvariant *RHS;
  BinaryOperatorKind Opcode;

public:
  SEInvariantOperation(const SEInvariant *lhs, const SEInvariant *rhs,
                       BinaryOperatorKind opcode)
    : SEInvariant(InvariantOperationClass), LHS(lhs), RHS(rhs),
      Opcode(opcode) {}

  const SEInvariant *getLHS() const { return LHS; }
  const SEInvariant *getRHS() const { return RHS; }
  BinaryOperatorKind getOpcode() const { return Opcode; }

  int Compare(const SEInvariantOperation &P) const;

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == InvariantOperationClass;
  }
};

class SEVariable : public SEAffine {
  const SEConstant *DummyCoefficient;

protected:
  SEVariable(const SymbolicExprContext &Ctx, SymbolicExprClass PC);

public:
  virtual unsigned getNumTerms() const { return 1; }
  virtual const SEInvariant *getCoefficient(unsigned Index) const {
    assert(Index == 0);
    return DummyCoefficient;
  }
  virtual const SEVariable *getVariable(unsigned Index) const {
    assert(Index == 0);
    return this;
  }

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() >= firstVariableClass &&
           P->getClass() <= lastVariableClass;
  }
};

class SENDRangeIndex : public SEVariable {
public:
  enum IndexKind {
    LOCAL_ID_0,
    LOCAL_ID_1,
    LOCAL_ID_2,
    GROUP_ID_0,
    GROUP_ID_1,
    GROUP_ID_2
  };

private:
  IndexKind Kind;

public:
  SENDRangeIndex(const SymbolicExprContext &Ctx, IndexKind kind)
    : SEVariable(Ctx, NDRangeIndexClass), Kind(kind) {}

  IndexKind getKind() const { return Kind; }

  int Compare(const SENDRangeIndex &P) const {
    if (Kind < P.Kind) {
      return -1;
    } else if (Kind == P.Kind) {
      return 0;
    } else {
      return 1;
    }
  }

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == NDRangeIndexClass;
  }
};

class SEInductionVariable : public SEVariable {
  WStmt *Loop;
  // 0 -- bound, step
  const SEAffine *Bound;
  const SEInvariant *Step;
  bool Inclusive;

public:
  SEInductionVariable(const SymbolicExprContext &Ctx, WStmt *loop,
                      const SEAffine *bound, const SEInvariant *step,
                      bool inclusive)
    : SEVariable(Ctx, InductionVariableClass), Loop(loop), Bound(bound),
      Step(step), Inclusive(inclusive) {}

  const WStmt *getLoop() const { return Loop; }
  const SEAffine *getBound() const { return Bound; }
  const SEInvariant *getStep() const { return Step; }
  bool isInclusive() const { return Inclusive; }

  int Compare(const SEInductionVariable &P) const;

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == InductionVariableClass;
  }
};

class SEBufferValue: public SEVariable {
  const SEBufferAddress *Address;
  QualType Type;

public:
  SEBufferValue(const SymbolicExprContext &Ctx, const SEBufferAddress *address,
                QualType type)
    : SEVariable(Ctx, BufferValueClass), Address(address), Type(type) {}

  const SEBufferAddress *getAddress() const { return Address; }
  QualType getType() { return Type; }

  int Compare(const SEBufferValue &P) const;

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == BufferValueClass;
  }
};

class SEComplexAffine : public SEAffine {
  AffineTerm *Terms;
  unsigned NumTerms;

  SEComplexAffine(const SymbolicExprContext &Ctx, ArrayRef<AffineTerm> terms);

public:
  virtual unsigned getNumTerms() const { return NumTerms; }
  virtual const SEInvariant *getCoefficient(unsigned Index) const {
    assert(Index < NumTerms && "Term access out of range!");
    return Terms[Index].Coeff;
  }
  virtual const SEVariable *getVariable(unsigned Index) const {
    assert(Index < NumTerms && "Term access out of range!");
    return Terms[Index].Var;
  }

  int Compare(const SEComplexAffine &P) const;

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == ComplexAffineClass;
  }

  friend class SEAffine;
};

class SEBufferAddress: public SymbolicExpr {
  IndexedVarDecl *IBase;
  const SymbolicExpr *Offset;

public:
  SEBufferAddress(IndexedVarDecl *base, const SymbolicExpr *offset)
    : SymbolicExpr(BufferAddressClass), IBase(base), Offset(offset) {}

  WParmVarDecl *getBase() const { return IBase->getDecl()->getAsParameter(); }
  IndexedVarDecl *getIndexedBase() const { return IBase; }
  const SymbolicExpr *getOffset() const { return Offset; }

  void setOffset(const SymbolicExpr *O) { Offset = O; }

  int Compare(const SEBufferAddress &P) const;

  void print(raw_ostream &OS) const;

  static bool classof(const SymbolicExpr *P) {
    return P->getClass() == BufferAddressClass;
  }
};

class SymbolicExprContext {
  mutable llvm::BumpPtrAllocator BumpAlloc;

  enum { CONSTANT_CACHE_RANGE = 16 };
  SEConstant *ConstantCache[CONSTANT_CACHE_RANGE * 2 + 1];

public:
  const SymbolicExpr *Unknown;

  // Frequently used constant values
  const SEConstant *Zero;
  const SEConstant *One;
  const SEConstant *Two;
  const SEConstant *MinusOne;

  // Dimensions and indices
  const SENDRangeDimension *LocalSize[3];
  const SENDRangeDimension *NumGroups[3];
  const SENDRangeIndex *LocalID[3];
  const SENDRangeIndex *GroupID[3];

  SymbolicExprContext();
  ~SymbolicExprContext() {}

  llvm::BumpPtrAllocator &getAllocator() const {
    return BumpAlloc;
  }

  void *Allocate(size_t Size, unsigned Align = 8) const {
    return BumpAlloc.Allocate(Size, Align);
  }
  void Deallocate(void *Ptr) const {}

  const SEConstant *LookupConstantCache(int64_t Value) const;

  const SEInvariant *CreateInvarOpInvar(const SEInvariant *LHS,
                                        const SEInvariant *RHS,
                                        BinaryOperatorKind Opcode) const;
  const SEInvariant *MergeInvarOpInvar(const SEInvariant *LHS,
                                       const SEInvariant *RHS,
                                       BinaryOperatorKind Opcode) const;
  void MergeInvarsOpInvar(SmallVectorImpl<const SEInvariant*> &LHS,
                          const SEInvariant *RHS,
                          BinaryOperatorKind Opcode) const;
  const SEAffine *CreateAffineOpAffine(const SEAffine *LHS, const SEAffine *RHS,
                                       BinaryOperatorKind Opcode) const;
  const SEConstant *CreateAffineMinusAffineConst(const SEAffine *LHS,
                                                 const SEAffine *RHS) const;
  const SEBufferAddress *CreateAddrOpAffine(const SEBufferAddress *LHS,
                                            const SEAffine *RHS,
                                            BinaryOperatorKind Opcode) const;
  const SymbolicExpr *CreateAddrMinusAddr(const SEBufferAddress *LHS,
                                          const SEBufferAddress *RHS) const;
  const SEConstant *CreateAddrMinusAddrConst(const SEBufferAddress *LHS,
                                             const SEBufferAddress *RHS) const;
};

class SymbolicAnalysis {
public:
  typedef llvm::DenseMap<IndexedVarDecl*, const SymbolicExpr*> VariableValueMapTy;

private:
  ASTContext &ASTCtx;
  SymbolicExprContext &SECtx;
  VariableValueMapTy VarValueMap;

public:
  SymbolicAnalysis(ASTContext &C, SymbolicExprContext &SC)
    : ASTCtx(C), SECtx(SC) {}

  void InitVariable(IndexedVarDecl *Var, const SymbolicExpr *P);
  bool UpdateVariable(IndexedVarDecl *Var, const SymbolicExpr *P);

  const SymbolicExpr *GetValueOf(WExpr *E);
  const SymbolicExpr *GetFieldValueOf(WExpr *E, FieldDecl *F);
  const SymbolicExpr *GetAddressOf(WExpr *E);
  const SymbolicExpr *GetFieldAddressOf(WExpr *E, FieldDecl *F);
  const SymbolicExpr *GetAssignedValueOf(WStmt *S);
  const SymbolicExpr *GetFieldAssignedValueOf(WStmt *S, FieldDecl *F);
};

} // namespace snu

} // namespace clang

#endif // LLVM_CLANG_SNU_MAPA_SYMBOLICANALYSIS_H
