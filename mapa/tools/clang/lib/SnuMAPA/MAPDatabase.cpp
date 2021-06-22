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

#include "clang/SnuMAPA/MAPDatabase.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include <cstdlib>
#include <string>

using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm::opt;

namespace clang {

namespace snu {

MAPDatabase::MAPDatabase() {
  records.clear();
  num_total = 0;
  num_analyzed = 0;
}

unsigned int MAPDatabase::GetParamOrder(FunctionDecl *FD, ParmVarDecl *PVD) {
  unsigned int order = 0;
  for (FunctionDecl::param_const_iterator CI = FD->param_begin(),
      CE = FD->param_end(); CI != CE; ++CI) {
    if (DeclarationName::compare(PVD->getDeclName(), (*CI)->getDeclName()) == 0)
      return order;
    order += 1;
  }
  assert(0);
}

void MAPDatabase::PrintInvariant(
    const SEInvariant *E, FunctionDecl *FD, raw_ostream &OS) {
  if (const SEInvariantOperation *SE = dyn_cast<SEInvariantOperation>(E)) {
    OS << '(';
    PrintInvariant(SE->getLHS(), FD, OS);
    OS << ' ' << BinaryOperator::getOpcodeStr(SE->getOpcode()) << ' ';
    PrintInvariant(SE->getRHS(), FD, OS);
    OS << ')';
  }

  else if (const SENDRangeDimension *SE = dyn_cast<SENDRangeDimension>(E)) {
    switch(SE->getKind()) {
    case SENDRangeDimension::LOCAL_SIZE_0: OS << "blockDim.x"; break;
    case SENDRangeDimension::LOCAL_SIZE_1: OS << "blockDim.y"; break;
    case SENDRangeDimension::LOCAL_SIZE_2: OS << "blockDim.z"; break;
    case SENDRangeDimension::NUM_GROUPS_0: OS << "gridDim.x"; break;
    case SENDRangeDimension::NUM_GROUPS_1: OS << "gridDim.y"; break;
    case SENDRangeDimension::NUM_GROUPS_2: OS << "gridDim.z"; break;
    }
  }

  else if (const SEParameter *SE = dyn_cast<SEParameter>(E)) {
    const WParmVarDecl *D = SE->getDecl();
    OS << "(*(" << D->getType().getAsString()
      << "*)args[" << GetParamOrder(FD, D->getOriginal()) << "])";
  }

  else if (const SEConstant *SE = dyn_cast<SEConstant>(E)) {
    OS << SE->getValue();
  }
}

bool MAPDatabase::PrintLoopBound(const SEAffine *E, MAPElement &Element,
    FunctionDecl *FD, raw_ostream &OS) {
  bool printError = false;

  if (SEInvariant::classof(E)) {
    PrintInvariant(static_cast<const SEInvariant*>(E), FD, OS);
  }
  else if (SEVariable::classof(E)) {
    //TODO
    assert(0);
  }
  else if (const SEComplexAffine *CA = dyn_cast<SEComplexAffine>(E)) {
    for (unsigned int i = 0, num_terms = CA->getNumTerms(); i != num_terms; ++i) {
      const SEInvariant *coeff = CA->getCoefficient(i);
      const SEVariable *var = CA->getVariable(i);

      if (var == NULL) {
        PrintInvariant(coeff, FD, OS);
      }
      else {
        // TODO
        printError = true;
        ;
      }
    }
  }

  if (printError == true) {
    // currently, invariant expression is supported only
    // need to think about variable expression more
    if (!SEInvariant::classof(E)) {
      llvm::errs() << "[WARNING] Buffer \""
        << Element.getBase()->getName() << "\" "
        << "in kernel \"" << FD->getName() << "\" "
        << "has induction variable which bound is non-invariant!\n";
      Element.print(llvm::errs());
      llvm::errs() << "\n";

      llvm::errs() << "Bound expression: ";
      E->print(llvm::errs());
      llvm::errs() << "\n\n";
    }
  }

  return true;
}

bool MAPDatabase::HandleSymbolicExpression(MAPElement &Element, MAPRecord &Record, FunctionDecl* FD) {
  const SymbolicExpr *OffsetSE = Element.getOffset();
  unsigned fill_mask = 0;
  bool rollback_push = false;

  if (SEInvariant::classof(OffsetSE)) {
    std::string constant_str;
    llvm::raw_string_ostream tmp_sos(constant_str);
    PrintInvariant(static_cast<const SEInvariant*>(OffsetSE), FD, tmp_sos);
    tmp_sos.flush();

    fill_mask |= 0x0100;
    Record.constant.push_back(constant_str);
  }
  else if (const SEComplexAffine *CA = dyn_cast<SEComplexAffine>(OffsetSE)) {
    unsigned int IV_idx = 0;

    for (unsigned int i = 0, num_terms = CA->getNumTerms(); i != num_terms; ++i) {
      const SEInvariant *coeff = CA->getCoefficient(i);
      const SEVariable *var = CA->getVariable(i);

      std::string coeff_str;
      llvm::raw_string_ostream coeff_ostream(coeff_str);
      PrintInvariant(coeff, FD, coeff_ostream);
      coeff_ostream.flush();

      if (var == NULL) {
        fill_mask |= 0x0100;
        Record.constant.push_back(coeff_str);
      }
      
      else if (const SENDRangeIndex *NDI = dyn_cast<SENDRangeIndex>(var)) {
        switch (NDI->getKind()) {
        case SENDRangeIndex::LOCAL_ID_0:
          fill_mask |= 0x0008;
          Record.lx_coeff.push_back(coeff_str); break;
        case SENDRangeIndex::LOCAL_ID_1:
          fill_mask |= 0x0010;
          Record.ly_coeff.push_back(coeff_str); break;
        case SENDRangeIndex::LOCAL_ID_2:
          fill_mask |= 0x0020;
          Record.lz_coeff.push_back(coeff_str); break;
        case SENDRangeIndex::GROUP_ID_0:
          fill_mask |= 0x0001;
          Record.gx_coeff.push_back(coeff_str); break;
        case SENDRangeIndex::GROUP_ID_1:
          fill_mask |= 0x0002;
          Record.gy_coeff.push_back(coeff_str); break;
        case SENDRangeIndex::GROUP_ID_2:
          fill_mask |= 0x0004;
          Record.gz_coeff.push_back(coeff_str); break;
        }
      }

      else if (const SEInductionVariable *IV = dyn_cast<SEInductionVariable>(var)) {
        // handle bound of induction varaible
        const SEAffine *Bound = IV->getBound();
        std::string bound_str;
        llvm::raw_string_ostream bound_ostream(bound_str);

        // TODO: is it safe method ?
        if (IV->isInclusive())
          bound_ostream << "1 + ";
        bound_ostream << "(" << coeff_str << ") * (";

        if (!PrintLoopBound(Bound, Element, FD, bound_ostream)) {
          rollback_push = true;
          break;
        }

        bound_ostream << ")";
        bound_ostream.flush();
        Record.iter_bound[IV_idx].push_back(bound_str);

        // handle step of induction variable
        std::string step_str;
        llvm::raw_string_ostream step_ostream(step_str);
        step_ostream << "(" << coeff_str << ") * (";
        PrintInvariant(IV->getStep(), FD, step_ostream);
        step_ostream << ")";
        step_ostream.flush();
        Record.iter_step[IV_idx].push_back(step_str);

        if (IV_idx == 0) {
          fill_mask |= 0x0040;
        }
        else if (IV_idx == 1) {
          fill_mask |= 0x0080;
        }
        else {
          assert(0 && "Maximum induction variable supported is 2");
        }

        IV_idx += 1;
      }

      // class SEBufferValue
      else {
        llvm::errs() << "Buffer \"" << Element.getBase()->getName() << "\" "
          << "has \"BufferValue\" symbolic expression which is currently unsupported!\n";
        rollback_push = true;
        break;
      }
    }
  }
  else {
    llvm::errs() << "Buffer \"" << Element.getBase()->getName() << "\" "
      << "has unsupported symbolic expression!\n";
    return false;
  }

  if (Element.isOneThreadOp()) {
    fill_mask |= 0x0200;
    Record.is_one_thread.push_back("1");
  }

  // error occurred, rollback the pushes to the vectors
  if (rollback_push) {
    if ((fill_mask & 0x0001) != 0)
      Record.gx_coeff.pop_back();
    if ((fill_mask & 0x0002) != 0)
      Record.gy_coeff.pop_back();
    if ((fill_mask & 0x0004) != 0)
      Record.gz_coeff.pop_back();
    if ((fill_mask & 0x0008) != 0)
      Record.lx_coeff.pop_back();
    if ((fill_mask & 0x0010) != 0)
      Record.ly_coeff.pop_back();
    if ((fill_mask & 0x0020) != 0)
      Record.lz_coeff.pop_back();
    if ((fill_mask & 0x0040) != 0) {
      Record.iter_bound[0].pop_back();
      Record.iter_step[0].pop_back();
    }
    if ((fill_mask & 0x0080) != 0) {
      Record.iter_bound[1].pop_back();
      Record.iter_step[1].pop_back();
    }
    if ((fill_mask & 0x0100) != 0) {
      Record.constant.pop_back();
    }
    if ((fill_mask & 0x0200) != 0) {
      Record.is_one_thread.pop_back();
    }
    return false;
  }

  if ((fill_mask & 0x0001) == 0)
    Record.gx_coeff.push_back("0");
  if ((fill_mask & 0x0002) == 0)
    Record.gy_coeff.push_back("0");
  if ((fill_mask & 0x0004) == 0)
    Record.gz_coeff.push_back("0");
  if ((fill_mask & 0x0008) == 0)
    Record.lx_coeff.push_back("0");
  if ((fill_mask & 0x0010) == 0)
    Record.ly_coeff.push_back("0");
  if ((fill_mask & 0x0020) == 0)
    Record.lz_coeff.push_back("0");
  if ((fill_mask & 0x0040) == 0) {
    Record.iter_bound[0].push_back("0");
    Record.iter_step[0].push_back("0");
  }
  if ((fill_mask & 0x0080) == 0) {
    Record.iter_bound[1].push_back("0");
    Record.iter_step[1].push_back("0");
  }
  if ((fill_mask & 0x0100) == 0)
    Record.constant.push_back("0");
  if ((fill_mask & 0x0200) == 0)
    Record.is_one_thread.push_back("0");

  return true;
}

bool MAPDatabase::TryMerge(SymbolicExprContext &SECtx,
    MAPElement *ThisElem, MAPElement *TargetElem) {
  if (ThisElem->getKind() != TargetElem->getKind())
    return false;

  if (ThisElem->getIndexedBase() != TargetElem->getIndexedBase())
    return false;

  SymbolicExpr *ThisOffset = const_cast<SymbolicExpr*>(ThisElem->getOffset());
  SymbolicExpr *TargetOffset = const_cast<SymbolicExpr*>(TargetElem->getOffset());

  if (ThisOffset->getClass() != TargetOffset->getClass())
    return false;

  int64_t ThisBase, ThisEnd;
  int64_t TargetBase, TargetEnd;

  // for constant expression, just check the value
  if (ThisOffset->getClass() == SymbolicExpr::ConstantClass) {
    SEConstant *ThisConst = dyn_cast<SEConstant>(ThisOffset);
    SEConstant *TargetConst = dyn_cast<SEConstant>(TargetOffset);

    ThisBase = ThisConst->getValue();
    ThisEnd = ThisBase + ThisElem->getWidth();

    TargetBase = TargetConst->getValue();
    TargetEnd = TargetBase + TargetElem->getWidth();

    if (!((ThisEnd < TargetBase) || (TargetEnd < ThisBase))) {
      int64_t NewBase = std::min(ThisBase, TargetBase);
      uint64_t NewLength = std::max(ThisEnd, TargetEnd) - NewBase;
      TargetConst->setValue(NewBase);
      TargetElem->setWidth(NewLength);
      return true;
    }
  }

  // for complex affine expression, first term represents offset
  // check if the first term is constant expression and the other terms are both same
  else if (ThisOffset->getClass() == SymbolicExpr::ComplexAffineClass) {
    SEComplexAffine *ThisCA = dyn_cast<SEComplexAffine>(ThisOffset);
    SEComplexAffine *TargetCA = dyn_cast<SEComplexAffine>(TargetOffset);

    unsigned int ThisNumTerms = ThisCA->getNumTerms();
    unsigned int TargetNumTerms = TargetCA->getNumTerms();

    if (ThisNumTerms == TargetNumTerms) {
      SEInvariant *ThisCoeff0 =
        const_cast<SEInvariant*>(ThisCA->getCoefficient(0));
      SEInvariant *TargetCoeff0 =
        const_cast<SEInvariant*>(TargetCA->getCoefficient(0));

      if (ThisCoeff0->getClass() == SymbolicExpr::ConstantClass &&
          TargetCoeff0->getClass() == SymbolicExpr::ConstantClass) {
        // check if other terms are all same
        for (unsigned int I = 1; I != ThisNumTerms; ++I) {
          if (ThisCA->getCoefficient(I)->Compare(*TargetCA->getCoefficient(I)) != 0 ||
              ThisCA->getVariable(I)->Compare(*TargetCA->getVariable(I)) != 0) {
            return false;
          }
        }

        // ok, now get constant value from the first term
        SEConstant *ThisConst = dyn_cast<SEConstant>(ThisCoeff0);
        SEConstant *TargetConst = dyn_cast<SEConstant>(TargetCoeff0);

        ThisBase = ThisConst->getValue();
        ThisEnd = ThisBase + ThisElem->getWidth();

        TargetBase = TargetConst->getValue();
        TargetEnd = TargetBase + TargetElem->getWidth();

        if (!((ThisEnd < TargetBase) || (TargetEnd < ThisBase))) {
          int64_t NewBase = std::min(ThisBase, TargetBase);
          uint64_t NewLength = std::max(ThisEnd, TargetEnd) - NewBase;
          dyn_cast<SEConstant>(TargetCoeff0)->setValue(NewBase);
          TargetElem->setWidth(NewLength);
          return true;
        }
      }
    }

    // when this element has zero offset
    else if (ThisNumTerms == TargetNumTerms - 1) {
      SEInvariant *TargetCoeff0 =
        const_cast<SEInvariant*>(TargetCA->getCoefficient(0));

      if (TargetCoeff0->getClass() == SymbolicExpr::ConstantClass) {
        // check if other terms are all same
        for (unsigned int I = 1; I != TargetNumTerms; ++I) {
          if (TargetCA->getCoefficient(I)->Compare(*ThisCA->getCoefficient(I-1)) != 0 ||
              TargetCA->getVariable(I)->Compare(*ThisCA->getVariable(I-1)) != 0) {
            return false;
          }
        }

        // ok, now get constant value from the first term
        SEConstant *TargetConst = dyn_cast<SEConstant>(TargetCoeff0);

        ThisBase = 0;
        ThisEnd = ThisBase + ThisElem->getWidth();

        TargetBase = TargetConst->getValue();
        TargetEnd = TargetBase + TargetElem->getWidth();

        if (!((ThisEnd < TargetBase) || (TargetEnd < ThisBase))) {
          int64_t NewBase = std::min(ThisBase, TargetBase);
          uint64_t NewLength = std::max(ThisEnd, TargetEnd) - NewBase;
          dyn_cast<SEConstant>(TargetCoeff0)->setValue(NewBase);
          TargetElem->setWidth(NewLength);
          return true;
        }
      }
    }

    // when target element has zero offset
    else if (ThisNumTerms == TargetNumTerms + 1) {
      SEInvariant *ThisCoeff0 =
        const_cast<SEInvariant*>(ThisCA->getCoefficient(0));

      if (ThisCoeff0->getClass() == SymbolicExpr::ConstantClass) {
        // check if other terms are all same
        for (unsigned int I = 1; I != ThisNumTerms; ++I) {
          if (ThisCA->getCoefficient(I)->Compare(*TargetCA->getCoefficient(I-1)) != 0 ||
              ThisCA->getVariable(I)->Compare(*TargetCA->getVariable(I-1)) != 0) {
            return false;
          }
        }

        // ok, now get constant value from the first term
        SEConstant *ThisConst = dyn_cast<SEConstant>(ThisCoeff0);

        ThisBase = ThisConst->getValue();
        ThisEnd = ThisBase + ThisElem->getWidth();

        TargetBase = 0;
        TargetEnd = TargetBase + TargetElem->getWidth();

        if (!((ThisEnd < TargetBase) || (TargetEnd < ThisBase))) {
          int64_t NewBase = std::min(ThisBase, TargetBase);
          uint64_t NewLength = std::max(ThisEnd, TargetEnd) - NewBase;

          SmallVector<SEAffine::AffineTerm, 8> NewTerms;
          NewTerms.push_back(
              SEAffine::AffineTerm(SEConstant::Create(SECtx, NewBase), NULL));
          for (unsigned int I = 0; I != TargetNumTerms; ++I) {
            const SEInvariant *NewCoeff = TargetCA->getCoefficient(I);
            const SEVariable *NewVar = TargetCA->getVariable(I);
            NewTerms.push_back(SEAffine::AffineTerm(NewCoeff, NewVar));
          }
          TargetElem->setOffset(SEAffine::Create(SECtx, NewTerms));
          TargetElem->setWidth(NewLength);
          return true;
        }
      }
    }

    else {
      return false;
    }
  }

  return false;
}

void MAPDatabase::store(MAPACore& MAPA, FunctionDecl *FD, ASTContext &Ctx,
    DiagnosticsEngine &Diags) {
  MAPRecord record;
  record.num_expressions = 0;
  unsigned int total_expressions = 0;
  unsigned int analyzed_expressions = 0;

  // save the mangled kernel name
  {
    std::unique_ptr<ItaniumMangleContext> MC(
        ItaniumMangleContext::create(Ctx, Diags));
    llvm::raw_string_ostream tmp_sos(record.kernel_name);
    MC->mangleName(FD, tmp_sos);
    tmp_sos.flush();
  }

  for (MAPACore::op_iterator OI = MAPA.op_begin(),
      OE = MAPA.op_end(); OI != OE; ++OI) {
    MAPElement Element = OI->second;
    if (Element.getAddress())
      total_expressions += 1;
  }

  for (MAPACore::buffer_iterator BI = MAPA.buffer_begin(),
      BE = MAPA.buffer_end(); BI != BE; ++BI) {
    WParmVarDecl* PVD = BI->first;
    MAPBuffer Buffer = BI->second;

    unsigned int arg_index = GetParamOrder(FD, PVD->getOriginal());
    if (Buffer.isReadOnly())
      record.readonly_buffers.push_back(arg_index);
    else
      record.non_readonly_buffers.push_back(arg_index);

    unsigned int num_expressions = 0;
    for (MAPBuffer::iterator BI = Buffer.begin(), BE = Buffer.end();
        BI != BE; ++BI) {
      MAPInterval Interval = *BI;
      for (MAPInterval::iterator II = Interval.begin(), IE = Interval.end();
          II != IE; ++II) {
        num_expressions++;
      }
    }

    unsigned int element_index = 0;
    unsigned int *same_or_merged =
      (unsigned int*)malloc(num_expressions * sizeof(unsigned int));
    memset(same_or_merged, 0, num_expressions * sizeof(unsigned int));

    for (MAPBuffer::iterator BI = Buffer.begin(), BE = Buffer.end();
        BI != BE; ++BI, ++element_index) {
      MAPInterval *Interval = const_cast<MAPInterval*>(&(*BI));

      for (MAPInterval::iterator II = Interval->begin(), IE = Interval->end();
          II != IE; ++II, ++element_index) {
        MAPElement *Element = const_cast<MAPElement*>(&II->Element);
        unsigned int inspect_index = element_index + 1;

        // traverse all later expressions and check if there exists same or
        // mergeable expression regardless of access type
        for (MAPBuffer::iterator CI = BI, CE = BE;
            CI != CE; ++CI, ++inspect_index) {
          MAPInterval::iterator DI = (CI == BI) ? II + 1 : (*CI).begin();
          MAPInterval::iterator DE = (CI == BI) ? IE     : (*CI).end();

          for (; DI != DE; ++DI, ++inspect_index) {
            MAPElement *InspElement = const_cast<MAPElement*>(&(*DI).Element);

            bool merged = false;
            bool same_exists = (Element->getWidth() == InspElement->getWidth() &&
                Element->getOffset()->Compare(*InspElement->getOffset()) == 0);
            if (same_exists == false)
              merged = TryMerge(MAPA.getSEContext(), Element, InspElement);

            // if exists, do not handle this symbolic expression
            if (same_exists || merged) {
              same_or_merged[inspect_index] += (same_or_merged[element_index] + 1);
              goto loop_end;
            }
          }
        }

        if (Element->getKind() >= MAPElement::OK_COMPLEX_AFFINE) {
          llvm::errs() << "[WARNING] Buffer \""
            << Element->getBase()->getName() << "\" "
            << "in kernel \"" << FD->getName() << "\" "
            << "has complex affine pattern which is unsupported!\n";
          continue;
        }

        if (!HandleSymbolicExpression(*Element, record, FD))
          continue;

//        llvm::errs() << "Final expression " << record.num_expressions << ": ";
//        Element->print(llvm::errs());
//        llvm::errs() << "\n";

        record.num_expressions += 1;
        record.kernel_arg_index.push_back(arg_index);
        record.fetch_size.push_back(Element->getWidth());

        analyzed_expressions += (same_or_merged[element_index] + 1);
loop_end:
        ;
      }
    }
  }

  records.push_back(record);

  llvm::outs() << "Kernel \"" << FD->getName() << "\" "
    << analyzed_expressions << "/" << total_expressions
    << " expressions analyzed\n";

  num_total += total_expressions;
  num_analyzed += analyzed_expressions;
}

void MAPDatabase::printTotalStatus(llvm::raw_ostream& outs) {
  llvm::outs() << "Total " << num_analyzed << "/" << num_total
    << " expressions analyzed\n";
}

void MAPDatabase::print(llvm::raw_ostream& outs) {
//  for (std::vector<MAPRecord>::iterator I = records.begin(),
//      E = records.end(); I != E; ++I) {
//    MAPRecord record = *I;
//    llvm::outs() << "Stored kernel name: " << record.kernel_name << "\n";
//    llvm::outs() << "Number of readonly buffers=" << record.readonly_buffers.size()
//      << ", non_readonly buffers=" << record.non_readonly_buffers.size() << "\n";
//
//    unsigned int num_expressions = record.kernel_arg_index.size();
//    for (unsigned int i = 0; i < num_expressions; ++i) {
//      llvm::outs() << ">>> Exp" << i << ". idx=\"" << record.kernel_arg_index[i] << "\", "
//        << "gx=\"" << record.gx_coeff[i] << "\", "
//        << "gy=\"" << record.gy_coeff[i] << "\", "
//        << "gz=\"" << record.gz_coeff[i] << "\", "
//        << "lx=\"" << record.lx_coeff[i] << "\", "
//        << "ly=\"" << record.ly_coeff[i] << "\", "
//        << "lz=\"" << record.lz_coeff[i] << "\", "
//        << "i0_bound=\"" << record.iter_bound[0][i] << "\", "
//        << "i0_step=\"" << record.iter_step[0][i] << "\", "
//        << "i1_bound=\"" << record.iter_bound[1][i] << "\", "
//        << "i1_step=\"" << record.iter_step[1][i] << "\", "
//        << "const=\"" << record.constant[i] << "\", "
//        << "fetch_size=\"" << record.fetch_size[i] << "\"\n";
//    }
//  }

  outs << "#include <stdint.h>\n"
       << "#include <stdio.h>\n"
       << "#include <stdlib.h>\n"
       << "#include <string.h>\n"
       << "#include <assert.h>\n"
       << "typedef struct dim3 {\n"
       << "  unsigned int x, y, z;\n"
       << "} dim3;\n\n";
  
  unsigned int num_kernels = records.size();

  outs << "extern \"C\" {\n\n";

  {
    outs << "int32_t MAPA_get_kernel_id(const char* func_name) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";

      outs << "if (strcmp(func_name, \"" << record.kernel_name << "\") == 0)\n"
           << "    return " << i << ";\n";
    }
    outs << "  if (strcmp(func_name, \"_GET_NUM_KERNELS_\") == 0)\n"
         << "      return " << num_kernels << ";\n";
    outs << "  return -1;\n"
         << "}\n\n";
  }

  {
    outs << "uint32_t MAPA_get_num_readonly_buffers(int32_t kernel_id) {\n"
         << "  switch(kernel_id) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];
      outs << "  case " << i << ": return " << record.readonly_buffers.size() << ";\n";
    }
    outs << "  }\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "uint32_t MAPA_get_num_non_readonly_buffers(int32_t kernel_id) {\n"
         << "  switch(kernel_id) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];
      outs << "  case " << i << ": return " << record.non_readonly_buffers.size() << ";\n";
    }
    outs << "  }\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "void MAPA_get_readonly_buffers(int32_t kernel_id, uint32_t *v) {\n"
         << "  switch(kernel_id) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];
      outs << "  case " << i << ": ";

      for (unsigned int ji = 0, je = record.readonly_buffers.size();
          ji != je; ++ji) {
        outs << "v[" << ji << "] = " << record.readonly_buffers[ji] << "; ";
      }
      outs << "break;\n";
    }
    outs << "  }\n"
         << "}\n\n";
  }

  {
    outs << "void MAPA_get_non_readonly_buffers(int32_t kernel_id, uint32_t *v) {\n"
         << "  switch(kernel_id) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];
      outs << "  case " << i << ": ";

      for (unsigned int ji = 0, je = record.non_readonly_buffers.size();
          ji != je; ++ji) {
        outs << "v[" << ji << "] = " << record.non_readonly_buffers[ji] << "; ";
      }
      outs << "break;\n";
    }
    outs << "  }\n"
         << "}\n\n";
  }

  {
    outs << "uint32_t MAPA_get_num_expressions(int32_t kernel_id) {\n"
         << "  switch(kernel_id) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];
      outs << "  case " << i << ": return " << record.kernel_arg_index.size() << ";\n";
    }
    outs << "  }\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "size_t MAPA_get_kernel_arg_index(int32_t kernel_id, uint32_t expr_index) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.kernel_arg_index.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return " << record.kernel_arg_index[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "bool MAPA_is_readonly_buffer(int32_t kernel_id, uint32_t expr_index) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";

      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.kernel_arg_index.size();
          ji != je; ++ji) {
        bool contains = false;
        unsigned int kernel_arg_index = record.kernel_arg_index[ji];
        for (unsigned int ki = 0, ke = record.readonly_buffers.size();
            ki != ke; ++ki) {
          if (record.readonly_buffers[ki] == kernel_arg_index) {
            contains = true;
            break;
          }
        }
        outs << "    case " << ji << ": return " << contains << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "bool MAPA_is_one_thread_expression(int32_t kernel_id, uint32_t expr_index) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";

      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.kernel_arg_index.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return " << record.is_one_thread[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_gx_coeff(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.gx_coeff.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.gx_coeff[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_gy_coeff(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.gy_coeff.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.gy_coeff[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_gz_coeff(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.gz_coeff.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.gz_coeff[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_lx_coeff(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.lx_coeff.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.lx_coeff[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_ly_coeff(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.ly_coeff.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.ly_coeff[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_lz_coeff(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.lz_coeff.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.lz_coeff[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_i0_bound(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.iter_bound[0].size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.iter_bound[0][ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_i0_step(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.iter_step[0].size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.iter_step[0][ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_i1_bound(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.iter_bound[1].size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.iter_bound[1][ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_i1_step(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.iter_step[1].size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.iter_step[1][ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_const(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.constant.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.constant[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  {
    outs << "int64_t MAPA_get_fetch_size(int32_t kernel_id, uint32_t expr_index,\n"
         << "    void** args, dim3 gridDim, dim3 blockDim) {\n";
    for (unsigned int i = 0; i < num_kernels; ++i) {
      MAPRecord record = records[i];

      if (i == 0)
        outs << "  ";
      else
        outs << "  else ";
      
      outs << "if (kernel_id == " << i << ") {\n";
      outs << "    switch (expr_index) {\n";

      for (unsigned int ji = 0, je = record.fetch_size.size();
          ji != je; ++ji) {
        outs << "    case " << ji << ": return (int64_t)" << record.fetch_size[ji] << ";\n";
      }
      outs << "    }\n"
           << "  }\n";
    }
    outs << "  assert(0);\n"
         << "  return 0;\n"
         << "}\n\n";
  }

  outs << "}\n";
}

} // namespace snu

} // namespace clang
