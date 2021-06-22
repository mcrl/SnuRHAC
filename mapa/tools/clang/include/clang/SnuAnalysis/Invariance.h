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

#ifndef LLVM_CLANG_SNU_ANALYSIS_INVARIANCE_H
#define LLVM_CLANG_SNU_ANALYSIS_INVARIANCE_H

#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WAST.h"
#include "clang/SnuAST/WCFG.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {

namespace snu {

class AdditionalInvariance {
public:
  virtual ~AdditionalInvariance() {}
  virtual bool handleExpr(WExpr *E) = 0;
};

class InvarianceImplBase {
protected:
  WCFG *Program;
  llvm::DenseSet<IndexedVarDecl*> InvariantVars;
  AdditionalInvariance *AInv;

  InvarianceImplBase(WCFG *program, AdditionalInvariance *AI);

  void ComputeInvariantVars();

public:
  bool isInvariantDefinition(WStmt *S);
  bool isInvariant(WExpr *E);
  bool isInvariant(IndexedVarDeclRef IVarRef);
};

class WorkItemInvariance : public InvarianceImplBase {
public:
  explicit WorkItemInvariance(WCFG *program, AdditionalInvariance *AI = NULL);
  ~WorkItemInvariance();
};

class RegionInvariance : public InvarianceImplBase {
public:
  RegionInvariance(WCFG *program, const llvm::DenseSet<WCFGBlock*> &Region,
                   AdditionalInvariance *AI = NULL);

private:
  void ExcludeDefinedVariables(WStmt *S, bool Outermost = false);
  void ExcludeDefinedVariable(IndexedVarDeclRef IVarRef);
};

} // namespace snu

} // namespace clang

#endif // LLVM_CLANG_SNU_ANALYSIS_INVARIANCE_H
