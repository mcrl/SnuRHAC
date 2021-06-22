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

#ifndef __RHAC_FATBIN_HANDLER_H__
#define __RHAC_FATBIN_HANDLER_H__

#include "rhac.h"
#include "utils.h"
#include "config.h"

#include <string>
#include <vector>
#include <map>
#include <fstream>

#define PTX_FILE 0
#define CUBIN_FILE 1

#define RHAC_PARAM_PTX \
  "\
.param .u32 _rhac_block_x_lower, \n\
.param .u32 _rhac_block_x_upper, \n\
.param .u32 _rhac_block_y_lower, \n\
.param .u32 _rhac_block_y_upper, \n\
.param .u32 _rhac_block_z_lower, \n\
.param .u32 _rhac_block_z_upper  \n\
"

#define RHAC_CLIPPING_PTX \
  "\
.reg .pred %rhac_reg_p<12>;\n\
.reg .b32 %rhac_reg_r<10>;\n\
\n\
ld.param.u32 %rhac_reg_r1, [_rhac_block_x_lower];\n\
ld.param.u32 %rhac_reg_r2, [_rhac_block_x_upper];\n\
ld.param.u32 %rhac_reg_r3, [_rhac_block_y_lower];\n\
ld.param.u32 %rhac_reg_r4, [_rhac_block_y_upper];\n\
ld.param.u32 %rhac_reg_r5, [_rhac_block_z_lower];\n\
ld.param.u32 %rhac_reg_r6, [_rhac_block_z_upper];\n\
\n\
mov.u32 %rhac_reg_r7, %ctaid.x;\n\
setp.lt.u32	%rhac_reg_p1, %rhac_reg_r7, %rhac_reg_r1;\n\
setp.gt.u32	%rhac_reg_p2, %rhac_reg_r7, %rhac_reg_r2;\n\
or.pred %rhac_reg_p3, %rhac_reg_p1, %rhac_reg_p2;\n\
mov.u32 %rhac_reg_r8, %ctaid.y;\n\
setp.lt.u32	%rhac_reg_p4, %rhac_reg_r8, %rhac_reg_r3;\n\
or.pred %rhac_reg_p5, %rhac_reg_p3, %rhac_reg_p4;\n\
setp.gt.u32	%rhac_reg_p6, %rhac_reg_r8, %rhac_reg_r4;\n\
or.pred %rhac_reg_p7, %rhac_reg_p5, %rhac_reg_p6;\n\
mov.u32 %rhac_reg_r9, %ctaid.z;\n\
setp.lt.u32	%rhac_reg_p8, %rhac_reg_r9, %rhac_reg_r5;\n\
or.pred %rhac_reg_p9, %rhac_reg_p7, %rhac_reg_p8;\n\
setp.gt.u32	%rhac_reg_p10, %rhac_reg_r9, %rhac_reg_r6;\n\
or.pred %rhac_reg_p11, %rhac_reg_p9, %rhac_reg_p10;\n\
\n\
@!%rhac_reg_p11 bra rhac_BB_next;\n\
ret;\n\
\n\
rhac_BB_next:\n\
"

typedef struct FatbinVar_t {
  int file_idx_;
  std::string var_name_;
  size_t type_bitwidth_;
  size_t array_width_;
} FatbinVar;

typedef struct FatbinFunc_t {
  int file_idx_;
  std::string type_; // "STT_FUNC"
  std::string scope_;
  std::string attribute_;
  std::string func_name_;
  std::string wrapped_func_name_;

  std::string ptx_prefix_;
  std::string ptx_params_;
  std::string ptx_body_;
  std::map<int, std::pair<int, int>> args_; // <ordinal, <offsets, size>>
  int static_smem_size_;
} FatbinFunc;

typedef struct FatbinFile_t {
  int file_idx_;
  int has_global_atomics_;
  std::string ptx_filename_;
  std::string cubin_filename_;
  std::string wrapped_ptx_filename_;
  std::string wrapped_fatbin_filename_;
  std::vector<FatbinFunc*> funcs_;
  std::vector<FatbinVar*> vars_;

  std::string ptx_header_;
} FatbinFile;

class FatbinHandler {

    // functions for global usage 
  public:
    static FatbinHandler* GetFatbinHandler();

    void CreateWrapper();
    const char* GetBinaryName();
    char* ReadBinary(const char* filename);
    void ExecuteShellCommand(const char* shell_command);
    void ExecuteShellCommand(const char* shell_command, std::string& dump_str);

    void GetPTXFileList(std::vector<std::string>& ret);

    void ExtractPTXs(std::vector<FatbinFile*> fatbin_files);
    void ExtractPTX(FatbinFile* fatbin_file);

    void CreateCubinFromPTXs(std::vector<FatbinFile*> fatbin_file);
    void CreateCubinFromPTX(FatbinFile*);

    void DeleteCudaFiles(std::vector<std::string> ptx_files);
    void DeleteCudaFile(std::string ptx_file);

    void GetKernelInfos(std::vector<FatbinFile*> fatbin_files);
    void GetKernelInfo(FatbinFile* fatbin_file);

    void WriteKernelInfos(std::vector<FatbinFile*> fatbin_files);
    std::string WriteKernelInfo(FatbinFile* fatbin_file);
    void WriteVarInfos(std::vector<FatbinFile*> fatbin_files);
    std::string WriteVarInfo(FatbinFile* fatbin_file);

    void GetSymbolMetaDatas(std::vector<FatbinFile*> fatbin_fils);
    void GetSymbolMetaData(FatbinFile* fatbin_file);

    void PrintFatbinFunc(FatbinFunc *fatbin_func);
    void PrintFatbinVar(FatbinVar *fatbin_var);

    void ModifyPTXFiles(std::vector<FatbinFile*> fatbin_files);
    void ModifyPTXFile(FatbinFile *fatbin_file);

    void ReadCubinFile(FatbinFile *fatbin_file);

    std::string GetFirstWord(std::string line);
    std::string GetSymbolNameInLine(std::string line);
    std::string ClearSymbolName(std::string dirt);
    int CountCharInString(std::string str, const char c);
    bool StartsWith(std::string str, std::string word);
    void ReplaceWith(std::string& str, std::string word1, std::string word2);
    void FindAndReplace(std::string& str, std::string word1, std::string word2);
    void ResolveVarName(std::string str, FatbinVar* fatbin_var);

    FatbinVar* GetFatbinVarObj(const int f_idx, const std::string name);
    FatbinFunc* GetFatbinFuncObj(const int f_idx, const std::string name);

    void CreateWrappedFatbins(std::vector<FatbinFile*> fatbin_files);
    void CreateWrappedFatbin(FatbinFile *fatbin_file);

    int GetNumFatbins() { return num_fatbins_; }
    void SetNumFatbins(int number) { num_fatbins_ = number; }

    void RegisterFatbinary(void **cuda_fatbin_handle);
    void RegisterFunction(void **cuda_fatbin_handle,
        const char *host_func, char *device_func);
    std::pair<int, char*> LookupFunction(const char *host_func);
    void RegisterVar(void **cuda_fatbin_handle,
        char *host_var, char *device_address);
    std::pair<int, char*> LookupVar(char *host_var);
    void RegisterTexture(void **cuda_fatbin_handle,
        const struct textureReference *host_var, const void *device_address);
    std::pair<int, char*> LookupTexture(const struct textureReference *host_var);
      
  public:
    static FatbinHandler* singletone_;
    static mutex_t mutex_;

  private:
    FatbinHandler();
    ~FatbinHandler();

    std::vector<FatbinFile*> fatbin_files_;
    std::vector<void**> cuda_fatbin_handles_;
    std::map<const void*, std::pair<int, char*>> cuda_func_map_;
    std::map<void*, std::pair<int, char*>> cuda_var_map_;
    std::map<const struct textureReference*, std::pair<int, char*>> cuda_texture_map_;
    int num_fatbins_;
};

#endif // __RHAC_FATBIN_HANDLER_H__
