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

#include "clang/SnuAnalysis/Dominator.h"
#include "clang/Basic/LLVM.h"
#include "clang/SnuAST/WCFG.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

namespace clang {

namespace snu {

void WCFGDominanceFrontier::recalculate(WCFGDominatorTree &DT) {
  Frontiers.clear();
  calculateInternal(DT, DT.getRootNode());
}

void WCFGDominanceFrontier::calculateInternal(WCFGDominatorTree &DT,
                                              const WCFGDomTreeNode *Node) {
  WCFGBlock *Block = Node->getBlock();
  DomSetTy DF; 

  for (WCFGBlock::succ_iterator S = Block->real_succ_begin(),
                                SEnd = Block->real_succ_end();
       S != SEnd; ++S) {
    if (DT.getNode(*S)->getIDom() != Node) {
      DF.insert(*S);
    }
  }

  for (WCFGDomTreeNode::const_iterator C = Node->begin(), CEnd = Node->end();
       C != CEnd; ++C) {
    WCFGBlock *ChildBlock = (*C)->getBlock();
    if (!Frontiers.count(ChildBlock)) {
      calculateInternal(DT, *C);

      DomSetTy &ChildDF = Frontiers[ChildBlock];
      for (DomSetTy::const_iterator F = ChildDF.begin(), FEnd = ChildDF.end();
           F != FEnd; ++F) {
        if (!DT.properlyDominates(Node, DT.getNode(*F))) {
          DF.insert(*F);
        }
      }
    }
  }

  Frontiers[Block] = DF;
}

} // namespace snu

} // namespace clang
