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

#include "clang/SnuMAPA/MAPAFrontend.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/SnuAST/WAST.h"
#include "clang/SnuAST/WCFG.h"
#include "clang/SnuAnalysis/Dominator.h"
#include "clang/SnuMAPA/InductionVariable.h"
#include "clang/SnuMAPA/MemoryAccessPattern.h"
#include "clang/SnuPreprocess/FunctionNormalizer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

llvm::cl::OptionCategory MAPACategory("MAPA");
llvm::cl::opt<std::string> OutputFileName(
    "o", llvm::cl::cat(MAPACategory), llvm::cl::desc("Output file name"));

using namespace clang;

namespace clang {

namespace snu {

class MAPAConsumer : public ASTConsumer {
  const std::string &InFile;
  DiagnosticsEngine &Diags;
  const LangOptions &LangOpts;
  const CodeGenOptions &CodeGenOpts;
  const MAPAOptions &MAPAOpts;
  MAPDatabase &MAPDB;

public:
  MAPAConsumer(const std::string &infile,
               DiagnosticsEngine &diags,
               const LangOptions &langopts,
               const CodeGenOptions &codegenopts,
               const MAPAOptions &mapaopts,
               MAPDatabase &mapdb) :
    InFile(infile),
    Diags(diags),
    LangOpts(langopts),
    CodeGenOpts(codegenopts),
    MAPAOpts(mapaopts),
    MAPDB(mapdb) {}

  void Initialize(ASTContext &Ctx);
  void HandleTranslationUnit(ASTContext &Ctx);

  void AnalyzeKernel(FunctionDecl *FD, ASTContext &Ctx);
};

void MAPAConsumer::Initialize(ASTContext &Ctx) {
}

void MAPAConsumer::HandleTranslationUnit(ASTContext &Ctx) {
  if (Diags.hasErrorOccurred())
    return;

  TranslationUnitDecl *TU = Ctx.getTranslationUnitDecl();
  {
    FunctionNormalizer Normalizer(Ctx);
    Normalizer.EnableInlineAll();
    Normalizer.EnableKernelOnly();
    Normalizer.NormalizeAll(TU);
  }

  for (DeclContext::decl_iterator D = TU->decls_begin(), DEnd = TU->decls_end();
       D != DEnd; ++D) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*D)) {
      if (FD->hasBody() &&
          FD->doesThisDeclarationHaveABody() &&
          (FD->hasAttr<OpenCLKernelAttr>() || FD->hasAttr<CUDAGlobalAttr>())) {
        AnalyzeKernel(FD, Ctx);
      }
    }
  }
}

void MAPAConsumer::AnalyzeKernel(FunctionDecl *FD, ASTContext &Ctx) {
  assert(FD->hasBody());
  assert(FD->hasAttr<OpenCLKernelAttr>() || FD->hasAttr<CUDAGlobalAttr>());
//  FD->dumpColor();

  Stmt *Body = FD->getBody();
  CFG::BuildOptions BO;
  std::unique_ptr<CFG> cfg = CFG::buildCFG(NULL, FD->getBody(), &Ctx, BO);
//  cfg->print(llvm::outs(), LangOptions(), true);

  WDeclContext DeclCtx;
  WStmt *WBody = WStmt::WrapClangAST(
      Ctx, DeclCtx, llvm::SmallVector<WExpr*, 4>(), Body);
  WCFG *Wcfg = WCFG::WrapClangCFG(DeclCtx, WBody, cfg.get());
  Wcfg->MakeSSAForm();
//  Wcfg->print(llvm::outs(), LangOptions());

  InductionVariableDetector IVD(Ctx);
  IVD.Visit(Wcfg);

  MAPACore MAPA(Ctx, Wcfg);
  MAPA.Analysis();
//  MAPA.print(llvm::outs(), Ctx);
  MAPDB.store(MAPA, FD, Ctx, Diags);
}

std::unique_ptr<ASTConsumer>
MAPAAction::CreateASTConsumer(CompilerInstance &CI,
                                           StringRef InFile) {
  return std::make_unique<MAPAConsumer>(InFile, CI.getDiagnostics(),
      CI.getLangOpts(), CI.getCodeGenOpts(), MAPAOpts, MAPDB);
}

void MAPAAction::EndSourceFileAction() {
  MAPDB.printTotalStatus(llvm::outs());

  if (strcmp(OutputFileName.c_str(), "") == 0) {
    MAPDB.print(llvm::outs());
  }
  else {
    // print to file
    std::error_code EC;
    llvm::raw_fd_ostream outCode(OutputFileName.c_str(), EC, llvm::sys::fs::F_Text);
    
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      exit(-1);
    }
    outCode.SetUnbuffered();

    MAPDB.print(outCode);
  }
}

bool ExecuteMAPAInvocation(CompilerInstance *Clang, const MAPAOptions &Opts) {
  if (Clang->getFrontendOpts().ShowHelp ||
      Clang->getFrontendOpts().ShowVersion) {
    llvm::outs() << "MAPA: Memory Access Pattern Analyzer\n";
    return true;
  }
  if (Clang->getDiagnostics().hasErrorOccurred())
    return false;
  std::unique_ptr<FrontendAction> Act(new MAPAAction(Opts));
  if (!Act)
    return false;
  bool Success = Clang->ExecuteAction(*Act);
  if (Clang->getFrontendOpts().DisableFree)
    llvm::BuryPointer(std::move(Act));
  return Success;
}

} // namespace snu

} // namespace clang
