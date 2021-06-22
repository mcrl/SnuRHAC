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

#include "clang/SnuMAPA/InductionVariable.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WAST.h"
#include "clang/SnuAST/WCFG.h"
#include "clang/SnuAnalysis/Invariance.h"
#include "clang/SnuAnalysis/Loop.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include <set>

namespace clang {

namespace snu {

namespace {

class GlobalMemoryInvariance : public AdditionalInvariance {
public:
  virtual bool handleExpr(WExpr *E) {
    // Currently, we assume all buffer accesses are loop-invariant.
    // This can be correctly determined after memory access pattern analysis.
    // (i.e., values from a read-only buffer are loop-invariant)
    LangAS AddressSpace = E->getType().getAddressSpace();
    if (AddressSpace == LangAS::opencl_global ||
        AddressSpace == LangAS::opencl_constant) {
      return true;
    }
    return false;
  }
};

} // anonymous namespace

void InductionVariableDetector::VisitLoop(Loop &L) {
  GlobalMemoryInvariance AInv;
  L.ComputeInductionVariables(&AInv);

  WBaseMuFunction *BaseIV = NULL;
  for (Loop::ind_var_iterator I = L.ind_var_begin(), E = L.ind_var_end();
       I != E; ++I) {
    if (I->hasBound()) {
      assert(BaseIV == NULL);
      BaseIV = new (ASTCtx) WBaseMuFunction(
          L.getLoopStmt(), I->InitVar, I->getOrAllocStep(ASTCtx, L.getParent()),
          I->isIncrement(), I->Bound, I->isInclusive());
      I->IndVar->setDefinedMu(BaseIV);
    }
  }
  if (BaseIV == NULL) {
    return;
  }
  for (Loop::ind_var_iterator I = L.ind_var_begin(), E = L.ind_var_end();
       I != E; ++I) {
    if (!I->hasBound()) {
      WAuxiliaryMuFunction *AuxIV = new (ASTCtx) WAuxiliaryMuFunction(
          L.getLoopStmt(), I->InitVar, I->getOrAllocStep(ASTCtx, L.getParent()),
          I->isIncrement(), BaseIV);
      I->IndVar->setDefinedMu(AuxIV);
    }
  }
}

} // namespace snu

} // namespace clang
