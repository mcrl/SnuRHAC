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

#ifndef LLVM_CLANG_SNU_MAPA_MAPDATABASE_H
#define LLVM_CLANG_SNU_MAPA_MAPDATABASE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/SnuMAPA/MemoryAccessPattern.h"
#include <string>

namespace clang {

namespace snu {

// each MAPRecord holds number of expressions in a kernel
class MAPRecord {
public:
  std::string kernel_name;
  int num_expressions;

  llvm::SmallVector<uint32_t, 8> readonly_buffers;
  llvm::SmallVector<uint32_t, 8> non_readonly_buffers;

  llvm::SmallVector<unsigned int, 8> kernel_arg_index;
  llvm::SmallVector<std::string, 8> gx_coeff;
  llvm::SmallVector<std::string, 8> gy_coeff;
  llvm::SmallVector<std::string, 8> gz_coeff;
  llvm::SmallVector<std::string, 8> lx_coeff;
  llvm::SmallVector<std::string, 8> ly_coeff;
  llvm::SmallVector<std::string, 8> lz_coeff;
  llvm::SmallVector<std::string, 8> iter_bound[2];
  llvm::SmallVector<std::string, 8> iter_step[2];
  llvm::SmallVector<std::string, 8> constant;
  llvm::SmallVector<std::string, 8> is_one_thread;
  llvm::SmallVector<size_t, 8> fetch_size;
};

class MAPDatabase {
public:
  MAPDatabase();

  bool TryMerge(SymbolicExprContext &SECtx,
      MAPElement *ThisElem, MAPElement *TargetElem);
  void store(MAPACore &MAPA, FunctionDecl *FD, ASTContext &Ctx,
      DiagnosticsEngine &Diags);
  void print(llvm::raw_ostream& outs);
  void printTotalStatus(llvm::raw_ostream& outs);

private:
  unsigned int GetParamOrder(FunctionDecl *FD, ParmVarDecl *PVD);
  void PrintInvariant(const SEInvariant *E, FunctionDecl *FD,
      raw_ostream &OS);
  bool PrintLoopBound(const SEAffine *E, MAPElement &Element, FunctionDecl *FD,
      raw_ostream &OS);
  bool HandleSymbolicExpression(MAPElement &Element, MAPRecord &Record,
      FunctionDecl* FD);

  std::vector<MAPRecord> records;
  unsigned int num_total;
  unsigned int num_analyzed;
};

} // namespace snu

} // namespace clang

#endif // LLVM_CLANG_SNU_MAPA_MAPDATABASE_H
