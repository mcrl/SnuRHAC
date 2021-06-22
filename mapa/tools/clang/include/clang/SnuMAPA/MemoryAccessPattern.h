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

#ifndef LLVM_CLANG_SNU_MAPA_MEMORYACCESSPATTERN_H
#define LLVM_CLANG_SNU_MAPA_MEMORYACCESSPATTERN_H

#include "clang/AST/ASTContext.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WAST.h"
#include "clang/SnuAST/WCFG.h"
#include "clang/SnuAnalysis/MemoryAccess.h"
#include "clang/SnuAnalysis/PointerAnalysis.h"
#include "clang/SnuMAPA/SymbolicAnalysis.h"
#include "clang/SnuSupport/OrderedDenseADT.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

namespace snu {

class MAPElement {
public:
  enum MAPElementKind {
    OK_CONSTANT,
    OK_AFFINE_IN_IDS,
    OK_AFFINE_IN_SIMPLE_BOUND_ITER,
    OK_AFFINE_IN_FIXED_BOUND_ITER,
    OK_AFFINE_IN_IDS_AND_SIMPLE_BOUND_ITER,
    OK_AFFINE_IN_IDS_AND_FIXED_BOUND_ITER,
    OK_COMPLEX_AFFINE,
    OK_COMPLEX
  };

private:
  MAPElementKind Kind;
  const SEBufferAddress *Addr;
  QualType Type;
  uint64_t Width;
  bool Read;
  bool Written;
  bool Reusable;
  bool OneThreadOp;

public:
  MAPElement();
  MAPElement(/*<unknown>,*/ QualType type, uint64_t width, bool r, bool w, bool ru);
  MAPElement(const SEBufferAddress *addr, QualType type, uint64_t width, bool r,
             bool w, bool ru);

  MAPElementKind getKind() const { return Kind; }
  const SEBufferAddress *getAddress() const { return Addr; }
  WParmVarDecl *getBase() const {
    assert(Addr != NULL);
    return Addr->getBase();
  }
  IndexedVarDecl *getIndexedBase() const {
    assert(Addr != NULL);
    return Addr->getIndexedBase();
  }
  const SymbolicExpr *getOffset() const {
    assert(Addr != NULL);
    return Addr->getOffset();
  }
  QualType getType() const { return Type; }
  uint64_t getWidth() const { return Width; }

  void setOffset(const SymbolicExpr *O) {
    assert(Addr != NULL);
    const_cast<SEBufferAddress*>(Addr)->setOffset(O);
  }
  void setWidth(uint64_t W) { Width = W; }
  void setOneThreadOp() { OneThreadOp = true; }

  bool isRead() const { return Read; }
  bool isWritten() const { return Written; }
  bool isReadOnly() const { return Read && !Written; }
  bool isWriteOnly() const { return !Read && Written; }
  bool isReusable() const { return Reusable; }
  bool isOneThreadOp() const { return OneThreadOp; }

  bool operator==(const MAPElement &RHS) const;

  void print(raw_ostream &OS) const;
};

class MAPInterval {
  struct MAPIntervalMember {
    MAPElement Element;
    uint64_t Distance;

    MAPIntervalMember()
      : Element(), Distance(0) {}
    MAPIntervalMember(const MAPElement &E, uint64_t D)
      : Element(E), Distance(D) {}

    uint64_t getWidth() const { return Element.getWidth(); }

    bool operator==(const MAPIntervalMember &RHS) const {
      return Element == RHS.Element && Distance == RHS.Distance;
    }
    bool operator<(const MAPIntervalMember &RHS) const {
      if (Distance != RHS.Distance) return Distance < RHS.Distance;
      if (getWidth() != RHS.getWidth()) return getWidth() < RHS.getWidth();
      return false;
    }
  };

  MAPElement::MAPElementKind Kind;
  const SEBufferAddress *Addr;
  SmallVector<MAPIntervalMember, 16> Members;
  bool Read;
  bool Written;

public:
  MAPInterval();
  MAPInterval(const MAPElement &E);

  MAPElement::MAPElementKind getKind() const { return Kind; }
  const SEBufferAddress *getAddress() const { return Addr; }
  bool isRead() const { return Read; }
  bool isWritten() const { return Written; }
  bool isReadOnly() const { return Read && !Written; }
  bool isWriteOnly() const { return !Read && Written; }

  typedef SmallVectorImpl<MAPIntervalMember>::const_iterator iterator;
  iterator begin() const { return Members.begin(); }
  iterator end() const { return Members.end(); }
  size_t size() const { return Members.size(); }

  bool isSingleLocation() const;
  bool contains(const MAPElement &E) const;
  uint64_t getDistanceOf(const MAPElement &E) const;

  bool addElement(const MAPElement &E, const SymbolicExprContext &SECtx);

  void print(raw_ostream &OS) const;
};

class MAPBuffer {
  SmallVector<MAPInterval, 16> Members;
  bool Read;
  bool Written;
  bool Reusable;
  bool HasUnknown;

public:
  MAPBuffer()
    : Read(false), Written(false), Reusable(false), HasUnknown(false) {}

  bool isUsed() const { return !Members.empty() || HasUnknown; }
  bool isRead() const { return Read; }
  bool isWritten() const { return Written; }
  bool isReadOnly() const { return Read && !Written; }
  bool isWriteOnly() const { return !Read && Written; }
  bool hasFlowDependence() const { return Read && Written && Reusable; }
  bool hasUnknown() const { return HasUnknown; }

  typedef SmallVectorImpl<MAPInterval>::const_iterator iterator;
  iterator begin() const { return Members.begin(); }
  iterator end() const { return Members.end(); }
  size_t size() const { return Members.size(); }

  void addElement(const MAPElement &E, const SymbolicExprContext &SECtx);

  void print(raw_ostream &OS) const;
};

class MAPACore {
  ASTContext &ASTCtx;
  WCFG *Program;
  SymbolicExprContext SECtx;
  SymbolicAnalysis SA;
  AliasSet Aliases;

  typedef OrderedDenseMap<WExpr*, MAPElement> OperationPatternMapTy;
  typedef OrderedDenseMap<WParmVarDecl*, MAPBuffer> BufferPatternMapTy;
  OperationPatternMapTy OperationPattern;
  BufferPatternMapTy BufferPattern;

public:
  MAPACore(ASTContext &C, WCFG *program);
  void Analysis();
  void Analysis(MemoryAccessTrace &Trace);

  const MAPElement &getPattern(WExpr *Op) const {
    assert(OperationPattern.count(Op));
    return (const_cast<MAPACore*>(this))->OperationPattern[Op];
  }
  const MAPBuffer &getPattern(WParmVarDecl *Buffer) const {
    assert(BufferPattern.count(Buffer));
    return (const_cast<MAPACore*>(this))->BufferPattern[Buffer];
  }

  typedef OperationPatternMapTy::const_iterator op_iterator;
  op_iterator op_begin() const { return OperationPattern.begin(); }
  op_iterator op_end() const { return OperationPattern.end(); }

  typedef BufferPatternMapTy::const_iterator buffer_iterator;
  buffer_iterator buffer_begin() const { return BufferPattern.begin(); }
  buffer_iterator buffer_end() const { return BufferPattern.end(); }

  SymbolicExprContext &getSEContext() { return SECtx; }

  bool isWorkItemFunction(WExpr* E,
      enum WWorkItemFunction::WorkItemFunctionKind kind);
  bool isGlobalIdExpression(WExpr* E);
  void InspectOneThreadOp(WExpr* Access, MAPElement *Elem);

  void print(raw_ostream &OS, const ASTContext &ASTCtx) const;

private:
  void InitializeVariables();
  void CalculateVariables();
  void CalculateBufferAccessPatterns();
};

} // namespace snu

} // namespace clang

#endif // LLVM_CLANG_SNU_MAPA_MEMORYACCESSPATTERN_H
