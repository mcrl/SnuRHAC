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

#include "fatbin_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>

char* binFile = NULL;

// alloc highest priority
__attribute__((constructor(101)))
void PreLoadBinaryFileName(int argc, char **argv, char **env) {
  binFile = (char *)malloc(sizeof(char)*(strlen(*argv)+1));
  strcpy(binFile, *argv);
}

FatbinHandler* FatbinHandler::singletone_ = NULL;
mutex_t FatbinHandler::mutex_;

FatbinHandler* FatbinHandler::GetFatbinHandler() {
  mutex_.lock();
  if (singletone_ == NULL)
    singletone_ = new FatbinHandler();
  mutex_.unlock();

  return singletone_;
}

FatbinHandler::FatbinHandler() {
}

FatbinHandler::~FatbinHandler() {
  // TODO : Delete fatbin_files_
}

void FatbinHandler::CreateWrapper() {
  std::vector<std::string> ptx_list;
  std::vector<std::string> cubin_list;
  std::vector<std::string>::iterator i;
  std::map<std::string, std::string> ptx_cubin_map;
  int file_idx;

  // get the list of ptx files in the binary
  GetPTXFileList(ptx_list);
  RHAC_LOG("PTX num files : %lu", ptx_list.size());
  for (i = ptx_list.begin(); i != ptx_list.end(); i++) {
    RHAC_LOG("PTX file : %s", (*i).c_str());
  }

  // Create fatbin files objects 
  for (i = ptx_list.begin(), file_idx = 1; 
       i != ptx_list.end(); 
       i++, file_idx++) {
    FatbinFile *new_fatbin_file;
    new_fatbin_file = new FatbinFile;
    new_fatbin_file->file_idx_ = file_idx;
    new_fatbin_file->has_global_atomics_ = 0;
    new_fatbin_file->ptx_filename_ = (*i);
    fatbin_files_.push_back(new_fatbin_file);
  }

  SetNumFatbins(fatbin_files_.size());

  // Extract ptx files
  ExtractPTXs(fatbin_files_);

  // Create cubin files from ptx file
  CreateCubinFromPTXs(fatbin_files_);

  // Get Symbol Meta data
  GetSymbolMetaDatas(fatbin_files_);

  // Modify PTX files
  ModifyPTXFiles(fatbin_files_);

  // get kernel info from cubins generated in the previous step
  GetKernelInfos(fatbin_files_);

  // write kernel info file 
  WriteKernelInfos(fatbin_files_);

  WriteVarInfos(fatbin_files_);

  // write wrapper code to extracted ptx files
  //WriteWrappedPTXFiles(fatbin_files_);

  CreateWrappedFatbins(fatbin_files_);

  // Delete Created Files
  // FIXME
  //DeleteCudaFiles(ptx_list);
  //DeleteCudaFiles(cubin_list);
  
  std::vector<std::string> wrapped_ptx_list;
  std::vector<FatbinFile*>::iterator fi;
  for (fi = fatbin_files_.begin();
       fi != fatbin_files_.end();
       fi ++)
  {
    wrapped_ptx_list.push_back((*fi)->wrapped_ptx_filename_);
  }
  //DeleteCudaFiles(wrapped_ptx_list);
      
}

const char* FatbinHandler::GetBinaryName() {
  assert (binFile != NULL);
  return (const char *)binFile;
}

char* FatbinHandler::ReadBinary(const char *filename) {
  return NULL;
}

void FatbinHandler::ExecuteShellCommand(const char* shell_command, std::string& dump_str) {
  FILE *fp;
  char dump_buf[1024];

  // TODO : This causes MPI warning - fork call
  fp = popen(shell_command, "r");
  if (fp == NULL)
    RHAC_LOG("Error command with %s", shell_command);
  assert(fp != NULL);

  dump_str.clear();

  // wait until the command is finished
  while (fgets(dump_buf, 1024, fp) != NULL)
    dump_str.append(dump_buf);

  pclose(fp);
}

void FatbinHandler::ExecuteShellCommand(const char *shell_command) {
  std::string t_dump_str;

  ExecuteShellCommand(shell_command, t_dump_str);
}

void FatbinHandler::GetPTXFileList(std::vector<std::string>& ret) {
  char input_command[1024];
  std::string dump_str;
  int local_stage = 0;
  int ptx_cnt = 1;

  const char* binary = GetBinaryName();


  sprintf(input_command, "cuobjdump -lptx %s\n", binary);

  ExecuteShellCommand(input_command, dump_str);

  std::istringstream iss(dump_str);

  std::string word;
  while (iss >> word) {
    switch (local_stage) {
      case 0: // "PTX"
        if (word.compare("PTX") != 0
            && word.compare("\0") != 0) {
          std::cerr << "error word : " << word << std::endl;
        }
        assert( (word.compare("PTX") == 0) 
             || (word.compare("\0") == 0) );
        local_stage = 1;
        break;
      case 1: // "FILE"
        assert(word.compare("file") == 0);
        local_stage = 2;
        break;
      case 2: // file number
        char ptx_num_buf[16];
        sprintf(ptx_num_buf, "%d", ptx_cnt);
       
        assert(word.compare(std::string(ptx_num_buf) + ":") == 0);
        ptx_cnt++;
        local_stage = 3;
        break;
      case 3: // PTX file name 
        ret.push_back(word);
        local_stage = 0;
        break;
      default:
        assert(0);
        break;
    }
  };

  return;
}

void FatbinHandler::ExtractPTXs(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator fi;
 
  RHAC_LOG("Extract PTX Files ....");

  for (fi = fatbin_files.begin();
       fi != fatbin_files.end();
       fi++) {
    ExtractPTX(*fi);
  }

  RHAC_LOG("PTX files are extracted");
}

void FatbinHandler::ExtractPTX(FatbinFile* fatbin_file) {
  char input_command[1024];
  std::string filename = fatbin_file->ptx_filename_;

  RHAC_LOG("\tExtract PTX File %s ... ", filename.c_str());

  sprintf(input_command, "cuobjdump -xptx %s %s\n", 
      filename.c_str(), GetBinaryName());

  ExecuteShellCommand(input_command);

  RHAC_LOG("\tDone");
}

void FatbinHandler::CreateCubinFromPTXs(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator fi;

  RHAC_LOG("Create Cubin files from PTX files ... ");

  for (fi = fatbin_files.begin(); 
       fi != fatbin_files.end();
       fi++)
  {
    CreateCubinFromPTX(*fi);
  }

  RHAC_LOG("Done");
}

void FatbinHandler::CreateCubinFromPTX(FatbinFile* fatbin_file) {
  char input_command[1024];
  std::string cubin_filename;

  RHAC_LOG("\tCreate cubin from %s", fatbin_file->ptx_filename_.c_str());

  cubin_filename = fatbin_file->ptx_filename_ + ".to.cubin";
  fatbin_file->cubin_filename_ = cubin_filename;

  sprintf(input_command, "nvcc %s -cubin %s -o %s",
      NVCC_ARCH_OPTION,
      fatbin_file->ptx_filename_.c_str(),
      fatbin_file->cubin_filename_.c_str());

  ExecuteShellCommand(input_command);

  RHAC_LOG("\tDone");
}

void FatbinHandler::DeleteCudaFiles(std::vector<std::string> filenames) {
  std::vector<std::string>::iterator i;

  RHAC_LOG("Delete Files ...");

  for (i = filenames.begin(); i != filenames.end(); i++)
    DeleteCudaFile(*i);

  RHAC_LOG("PTX files are deleted");

}

void FatbinHandler::DeleteCudaFile(std::string filename) {
  char input_command[1024];

  RHAC_LOG("  Delete file %s ... ", filename.c_str());

  sprintf(input_command, "rm %s\n", filename.c_str());

  ExecuteShellCommand(input_command);

  RHAC_LOG("  Done");
}

void FatbinHandler::GetKernelInfos(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator fi;

  RHAC_LOG("Get Kernel Informations ... ");

  for (fi = fatbin_files.begin(); fi != fatbin_files.end(); fi++) {
    GetKernelInfo(*fi);
  }

  RHAC_LOG("Done");
}

void FatbinHandler::GetKernelInfo(FatbinFile* fatbin_file) {
  RHAC_LOG("Get Kernel Information for %d ...", fatbin_file->file_idx_);
  RHAC_LOG("\tcubin file name : %s", fatbin_file->cubin_filename_.c_str());

  // Read Cubin Files and Get Argument Info
  ReadCubinFile(fatbin_file);

  // Print for purpose of checking
  std::vector<FatbinFunc*>::iterator fi;
  std::vector<FatbinVar*>::iterator vi;
  for (fi = fatbin_file->funcs_.begin();
      fi != fatbin_file->funcs_.end();
      fi ++) {
    PrintFatbinFunc(*fi);
  }
  for (vi = fatbin_file->vars_.begin();
      vi != fatbin_file->vars_.end();
      vi++) {
    PrintFatbinVar(*vi);
  }

  RHAC_LOG("\tDone");
}

void FatbinHandler::WriteKernelInfos(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator fi; 
  std::string infos;
  std::ofstream kernel_info_fstream;

  
  infos.clear();

  for (fi = fatbin_files.begin(); fi != fatbin_files.end(); fi++) {
    infos.append(WriteKernelInfo(*fi));
  }

  RHAC_LOG(" Kernel Infos : %s", infos.c_str());

  kernel_info_fstream.open(KERNEL_INFO_FILE_NAME, std::ofstream::out);
  kernel_info_fstream << infos;
  kernel_info_fstream.close();
}

std::string FatbinHandler::WriteKernelInfo(FatbinFile* fatbin_file) {
  std::string ret;
  ret.clear();

  int f_idx = fatbin_file->file_idx_;

  std::vector<FatbinFunc*> func_list;
  std::vector<FatbinFunc*>::iterator func;

  func_list = fatbin_file->funcs_;

  for (func = func_list.begin();
       func != func_list.end();
       func++) {
    // Format : file_idx func_name has_global_atomics num_args arg_size_total
    // arg_size0 arg_size1 ...

    ret.append(std::to_string(f_idx) + " ");

    ret.append((*func)->func_name_ + " ");

    ret.append(std::to_string(fatbin_file->has_global_atomics_) + " ");

    int num_args = (*func)->args_.size();
    ret.append(std::to_string(num_args) + " ");

    std::map<int, std::pair<int, int>>::iterator mi;

    for (mi = (*func)->args_.begin();
         mi != (*func)->args_.end();
         mi++) 
    {
      int arg_size;
      arg_size = (mi->second).second; // arg size
      ret.append(std::to_string(arg_size) + " ");
    }

    ret.append("\n");
  }

  return ret;
}

void FatbinHandler::WriteVarInfos(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator fi;
  std::string infos;
  std::ofstream var_info_fstream;

  infos.clear();

  for (fi = fatbin_files.begin(); fi != fatbin_files.end(); fi++) {
    infos.append(WriteVarInfo(*fi));
  }

  var_info_fstream.open(VAR_INFO_FILE_NAME, std::ofstream::out);
  var_info_fstream << infos;
  var_info_fstream.close();
}

std::string FatbinHandler::WriteVarInfo(FatbinFile* fatbin_file) {
  std::string ret;
  ret.clear();

  int f_idx = fatbin_file->file_idx_;

  std::vector<FatbinVar*> var_list;
  std::vector<FatbinVar*>::iterator var;

  var_list = fatbin_file->vars_;

  for (var = var_list.begin();
       var != var_list.end();
       var++)
  {
    // Format : file_idx var_name type_bitwidth array_width

    ret.append(std::to_string(f_idx) + " ");
    ret.append((*var)->var_name_ + " ");
    ret.append(std::to_string((*var)->type_bitwidth_) + " ");
    ret.append(std::to_string((*var)->array_width_) + " ");
    ret.append("\n");
  }

  return ret;
}

void FatbinHandler::GetSymbolMetaDatas(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator fi;

  RHAC_LOG("GetSymbolMetaDatas Start ... ");

  for (fi = fatbin_files.begin();
       fi != fatbin_files.end();
       fi++)
  {
    GetSymbolMetaData(*fi);
  }

  RHAC_LOG("GetSymbolMetaDatas Done ... ");
}

void FatbinHandler::GetSymbolMetaData(FatbinFile* fatbin_file) {
  std::string dump_str;
  std::string line;
  char input_command[1024];

  RHAC_LOG("\tGetSymbolMetaData %s", fatbin_file->cubin_filename_.c_str());

  sprintf(input_command, "cuobjdump -symbols %s\n",
      fatbin_file->cubin_filename_.c_str());

  ExecuteShellCommand(input_command, dump_str);

  std::istringstream iss(dump_str);


  while (std::getline(iss, line)) {
    std::istringstream line_iss(line);
    std::string word;
    int local_stage = 0;
    FatbinFunc* new_fatbin_func;

    while (line_iss >> word) {

      switch (local_stage) {
        case 0:
          local_stage = (word.compare("STT_FUNC") == 0) ? 1 : 0;
          if (local_stage != 0) {
            new_fatbin_func = new FatbinFunc;
            new_fatbin_func->file_idx_ = fatbin_file->file_idx_;
            new_fatbin_func->type_ = word;
          }
          break;
        case 1:
          // FIXME
          assert(word.compare("STB_GLOBAL") == 0 ||
                 word.compare("STB_LOCAL") == 0 ||
                 word.compare("STB_WEAK") == 0);

          local_stage = (word.compare("STB_GLOBAL") == 0 ||
                         word.compare("STB_LOCAL") == 0 ||
                         word.compare("STB_WEAK") == 0) ? 2 : 0;

          if (local_stage != 0) {
            new_fatbin_func->scope_ = word;
          }
          break;
        case 2:
          if (word.compare("STV_DEFAULT") == 0)
            local_stage = (new_fatbin_func->scope_.compare("STB_WEAK") == 0) ? 4 : 3;
          else if (word.compare("STO_ENTRY") == 0)
            local_stage = 4;
          else
            local_stage = 0;
            
          if (local_stage != 0) {
            new_fatbin_func->attribute_ = word;
          }  
          break;
        case 3: // STV_DEFAULT
//          assert(word.compare("U") == 0);
          local_stage = (word.compare("U") == 0) ? 4 : 0;
          break;
        case 4: // FUNC name
          assert(word.length() != 0);
          new_fatbin_func->func_name_ = word;
          fatbin_file->funcs_.push_back(new_fatbin_func);
          local_stage = 0;
          break;
      }

      if (local_stage == 0)
        break;
    }
  }
}

void FatbinHandler::PrintFatbinFunc(FatbinFunc *fatbin_func) {
  std::vector<int>::iterator i;
  // FIXME
  char tmp_str[1024];
  tmp_str[0] = '\0';

  RHAC_LOG("Function name : %s", fatbin_func->func_name_.c_str());
  RHAC_LOG("\tFile index  : %d", fatbin_func->file_idx_);
  RHAC_LOG("\tType        : %s", fatbin_func->type_.c_str());
  RHAC_LOG("\tScope       : %s", fatbin_func->scope_.c_str());
  RHAC_LOG("\tAttr        : %s", fatbin_func->attribute_.c_str());
  RHAC_LOG("\tPrefix      : %s", fatbin_func->ptx_prefix_.c_str());

  tmp_str[0] = '\0';
  std::map<int, std::pair<int, int>>::iterator mi;
  for (mi = fatbin_func->args_.begin();
       mi != fatbin_func->args_.end();
       mi++)
  {
    sprintf(tmp_str + strlen(tmp_str), "0x%x : (Off - 0x%x, Size - 0x%x) ", 
        mi->first,
        (mi->second).first,
        (mi->second).second);
  }
  RHAC_LOG("\tArgs        : %s", tmp_str);


  RHAC_LOG("\tSmem Size   : %d", fatbin_func->static_smem_size_);
  
  //RHAC_LOG("\tBody        : \n%s", fatbin_func->ptx_body_.c_str());
  RHAC_LOG("\tBody        : %s", "commented out");
}

void FatbinHandler::PrintFatbinVar(FatbinVar *fatbin_var) {
#if 0
  RHAC_LOG("Variable name : %s", fatbin_var->var_name_.c_str());
  RHAC_LOG("\tFile index : %d", fatbin_var->file_idx_);
  RHAC_LOG("\tType       : %s", fatbin_var->type_.c_str());
  RHAC_LOG("\tScope      : %s", fatbin_var->scope_.c_str());
  RHAC_LOG("\tAttr       : %s", fatbin_var->attribute_.c_str());
  RHAC_LOG("\tPTX        : %s", fatbin_var->ptx_code_.c_str());
#endif
}

void FatbinHandler::ModifyPTXFiles(std::vector<FatbinFile*> fatbin_files) {
  RHAC_LOG("ModifyPTXFiles Start ... ");

  std::vector<FatbinFile*>::iterator fi;

  for (fi = fatbin_files.begin();
       fi != fatbin_files.end();
       fi++)
  {
    ModifyPTXFile(*fi);
  }

  RHAC_LOG("ModifyPTXFiles Done ... ");
}

void FatbinHandler::ModifyPTXFile(FatbinFile *fatbin_file) {
  std::ifstream ifs;
  std::ofstream ofs;
  std::string wrapped_filename;
  std::string line;

  // new wrapped file name 
  wrapped_filename.clear();
  wrapped_filename.append("wrapped.");
  wrapped_filename.append(fatbin_file->ptx_filename_);

  fatbin_file->wrapped_ptx_filename_ = wrapped_filename;

  RHAC_LOG("\tInput ptx file name : %s", 
      fatbin_file->ptx_filename_.c_str());
  RHAC_LOG("\tInput ptx file name : %s", 
      fatbin_file->wrapped_ptx_filename_.c_str());

  ifs.open(fatbin_file->ptx_filename_.c_str());
  ofs.open(fatbin_file->wrapped_ptx_filename_.c_str());

  while (std::getline(ifs, line)) {
    std::istringstream line_iss(line);
    std::string word;
    std::string head_token;

    line_iss >> head_token;

    if (head_token.compare(".visible") == 0 ||
        head_token.compare(".entry") == 0) 
    {
      // Get function name 
      std::string func_name;
      func_name = GetSymbolNameInLine(line);
      RHAC_LOG("FXXX : %s", func_name.c_str());

      ofs << line;

      // Find num parenthesis 
      int opened_parenthesis = 0;
      int closed_parenthesis = 0;
      bool param_exist = false;
      opened_parenthesis = CountCharInString(line, '(');
      closed_parenthesis = CountCharInString(line, ')');
      RHAC_LOG("FXXX : %d", opened_parenthesis);

      // Handle Param
      while (opened_parenthesis > closed_parenthesis) {
        std::getline(ifs, line);
        opened_parenthesis += CountCharInString(line, '(');
        closed_parenthesis += CountCharInString(line, ')');

        if (!param_exist) {
          if (GetFirstWord(line).compare(".param") == 0)
            param_exist = true;
        }

        if (opened_parenthesis > closed_parenthesis) {
          ofs << std::endl;
          ofs << line;
        }
      }

      assert(line.compare(")") == 0);

      if (param_exist) {
        ofs << "," << std::endl;
      }

      // insert global variables as kernel arguments
      unsigned int num_global_vars = fatbin_file->vars_.size();
      for (unsigned int I = 0, E = num_global_vars; I != E; ++I) {
        ofs <<  ".param .u64 _rhac_global_var_" << I << ", \n";
      }

      // insert rhac params 
      ofs << RHAC_PARAM_PTX;
      ofs << line << std::endl;

      // insert rhac clipping code
      int opened_brace = 0;
      int closed_brace = 0;

      std::getline(ifs, line);
      opened_brace += CountCharInString(line, '{');
      closed_brace += CountCharInString(line, '}');

      while (line.compare("{") != 0) {
        ofs << line << std::endl;
        std::getline(ifs, line);
      }

      ofs << line << std::endl;
      ofs << RHAC_CLIPPING_PTX;

      // add PTX code for loading global variables
      if (num_global_vars) {
        ofs << ".reg .b64 %rhac_var_x_r<" << num_global_vars + 1 << ">;\n";
        ofs << ".reg .b64 %rhac_var_y_r<" << num_global_vars + 1 << ">;\n";
      }

      for (unsigned int I = 0, E = num_global_vars; I != E; ++I) {
        ofs << "ld.param.u64 %rhac_var_x_r" << I+1 << ", [_rhac_global_var_" << I << "];\n";
      }

      for (unsigned int I = 0, E = num_global_vars; I != E; ++I) {
        ofs << "cvta.to.global.u64 %rhac_var_y_r" << I+1 << ", %rhac_var_x_r" << I+1 << ";\n";
      }

      // inspect function body
      while (opened_brace > closed_brace) {
        std::getline(ifs, line);
        opened_brace += CountCharInString(line, '{');
        closed_brace += CountCharInString(line, '}');

        // change GPU-local global atomics to system-wide global atomics
        if (StartsWith(line, "atom.global.") && !StartsWith(line, "atom.global.sys.")) {
          ReplaceWith(line, "atom.global.", "atom.global.sys.");
          fatbin_file->has_global_atomics_ = 1;
        }

        // change address of global variables
        for (unsigned int I = 0, E = num_global_vars; I != E; ++I) {
          FindAndReplace(line, fatbin_file->vars_[I]->var_name_,
              "%rhac_var_y_r" + std::to_string(I + 1));
        }

        ofs << line << std::endl;
      }
    }
    else if (head_token.compare(".global") == 0) {
      bool bypass_inspection = false;
      bool bypass_nexttoken = false;
      std::string word;

      while (line_iss >> word) {
        if (word.compare(".texref") == 0 ||
            word.compare(".surfref") == 0 ||
            StartsWith(word, "$str")) {
          bypass_inspection = true;
          break;
        }
      }

      if (bypass_inspection) {
        ofs << line << std::endl;
      }
      else {
        FatbinVar* new_fatbin_var;
        new_fatbin_var = new FatbinVar;
        new_fatbin_var->file_idx_ = fatbin_file->file_idx_;
        fatbin_file->vars_.push_back(new_fatbin_var);

        std::istringstream line_iss2(line);
        while (line_iss2 >> word) {
          if (bypass_nexttoken) {
            bypass_nexttoken = false;
            continue;
          }

          if (StartsWith(word, ".align") == true) {
            bypass_nexttoken = true;
          }
          else if (StartsWith(word, "=") == true) {
            break;
          }
          else if (StartsWith(word, ".") == true) {
            if (StartsWith(word, ".u") == true ||
                StartsWith(word, ".b") == true ||
                StartsWith(word, ".f") == true) {
              new_fatbin_var->type_bitwidth_ = std::stoi(word.substr(2, word.size() - 2));
            }
          }
          else {
            ResolveVarName(word, new_fatbin_var);
          }
        }
      }
    }
    else {
      ofs << line << std::endl; 
    }
  }

  ifs.close();
  ofs.close();
}

void FatbinHandler::ReadCubinFile(FatbinFile *fatbin_file) {
  char input_command[1024];
  std::string dump_str;

  sprintf(input_command, 
          "cuobjdump -elf %s",
          fatbin_file->cubin_filename_.c_str());

  ExecuteShellCommand(input_command, dump_str);

  std::istringstream iss(dump_str);
  std::string line, word;
  int local_stage = 0;
  FatbinFunc* func_info = NULL;
  std::string last_func;
  std::vector<FatbinFunc*> func_list = fatbin_file->funcs_;
  int ordinal = 0;
  int tmp_arg_offset, tmp_arg_size;

  while (iss >> word) {

      if (word.compare(0, 9, ".nv.info.") == 0) {
        // match function in func_list
        bool matched = false;
        std::vector<FatbinFunc*>::iterator fi;
        for (fi = func_list.begin(); fi != func_list.end(); fi++) {
          if (word.compare(9, word.size()-9, (*fi)->func_name_) == 0) {
            func_info = (*fi);
            matched = true;  // FIXME
            break;
          }
        }

        assert(fi != func_list.end());

        local_stage = matched ? 1 : 0;
      }
      else {
        switch (local_stage) {
        case 1:
          local_stage = (word.compare(0, 3, "<0x") == 0) ? 2 : 0;
          break;
        case 2:
          local_stage = (word.compare("EIATTR_KPARAM_INFO") == 0) ? 3 : local_stage;
          break;
        case 3:
          local_stage = (word.compare("Ordinal") == 0) ? 4 : local_stage;
          break;
        case 4:
          local_stage = 5;
          break;
        case 5:
          ordinal = strtol(word.c_str(), NULL, 16);
          local_stage = 6;
          break;
        case 6:
          local_stage = (word.compare("Offset") == 0) ? 7 : local_stage;
          break;
        case 7:
          local_stage = 8;
          break;
        case 8:
          assert(func_info != NULL);
          tmp_arg_offset = strtol(word.c_str(), NULL, 16);
          local_stage = 9;
          break;
        case 9:
          local_stage = (word.compare("Size") == 0) ? 10 : local_stage;
          break;
        case 10:
          local_stage = 11;
          break;
        case 11:
          assert(func_info != NULL);
          tmp_arg_size = strtol(word.c_str(), NULL, 16);
          std::pair<int, int> tmp_arg;
          tmp_arg = std::make_pair(tmp_arg_offset, tmp_arg_size);
          func_info->args_.insert(std::make_pair(ordinal, tmp_arg));
          local_stage = 2;
          break;
        }
      }
    } 
}

std::string FatbinHandler::GetFirstWord(std::string line)
{
  std::istringstream iss(line);
  std::string word;
  iss >> word;
  return word;
}

std::string FatbinHandler::GetSymbolNameInLine(std::string line) {
  std::istringstream iss(line);
  std::string word;
  std::string ret;

  while (iss >> word) {
    if (word.compare(0, 6, ".align") == 0) {
      iss >> word;
    }
    else if (word.compare(0, 1, ".") != 0) {
      ret = ClearSymbolName(word);
    }
  };

  return ret;
}

std::string FatbinHandler::ClearSymbolName(std::string dirt) {
  std::string delms = "[;(:";
  std::istringstream del_stream(delms);
  std::string ret;
  char delm;
  ret = dirt;
  while (del_stream >> delm) {
    std::istringstream iss(ret);
    getline(iss, ret, delm);
  };

  return ret;
}

int FatbinHandler::CountCharInString(std::string str, const char c) {
  int ret;
  ret = std::count(str.begin(), str.end(), c);
  return ret;
}

bool FatbinHandler::StartsWith(std::string str, std::string word) {
  return (str.compare(0, word.size(), word) == 0);
}

void FatbinHandler::ReplaceWith(std::string& str,
    std::string word1, std::string word2) {
  str.replace(0, word1.size(), word2);
}

void FatbinHandler::FindAndReplace(std::string &str,
    std::string word1, std::string word2) {
  std::size_t found = str.find(word1);
  if (found != std::string::npos) {
    str.replace(found, word1.size(), word2);
  }
}

void FatbinHandler::ResolveVarName(std::string str, FatbinVar* fatbin_var) {
  std::size_t found_open = str.find('[');
  std::size_t found_close = str.find(']');

  if (found_open == std::string::npos ||
      found_close == std::string::npos) {
    fatbin_var->var_name_ = ClearSymbolName(str);
    fatbin_var->array_width_ = 1;
  }
  else {
    fatbin_var->var_name_ = str.substr(0, found_open);

    std::string number = str.substr(found_open + 1, found_close - found_open - 1);
    fatbin_var->array_width_ = std::stoi(number);
  }
}

FatbinVar* FatbinHandler::GetFatbinVarObj(const int f_idx, const std::string name) {
  std::vector<FatbinFile*>::iterator fi;
  std::vector<FatbinVar*>::iterator vi;

  // find fatbin with idx 
  for (fi = fatbin_files_.begin(); fi != fatbin_files_.end(); fi++) {
    if ((*fi)->file_idx_ == f_idx)
      break;
  }

  assert(fi != fatbin_files_.end());

  // find fatbin var obj
  std::vector<FatbinVar*> t_vars = (*fi)->vars_;
  for (vi = t_vars.begin(); vi != t_vars.end(); vi++) {
    if (name.compare((*vi)->var_name_) == 0)
      break;
  }

  assert(vi != t_vars.end());

  return (*vi);
}

FatbinFunc* FatbinHandler::GetFatbinFuncObj(int f_idx, std::string name) {
  std::vector<FatbinFile*>::iterator fi;
  std::vector<FatbinFunc*>::iterator funci;

  // fatin fatbin obj with idx
  for (fi = fatbin_files_.begin(); fi != fatbin_files_.end(); fi++) {
    if ((*fi)->file_idx_ == f_idx)
      break;
  }

  assert(fi != fatbin_files_.end());

  // find fatbin func obj
  std::vector<FatbinFunc*> t_funcs = (*fi)->funcs_;
  for (funci = t_funcs.begin(); funci != t_funcs.end(); funci++) {
    if (name.compare((*funci)->func_name_) == 0)
      break;
  }

  assert(funci != t_funcs.end());

  return (*funci);
}

void FatbinHandler::CreateWrappedFatbins(std::vector<FatbinFile*> fatbin_files) {
  std::vector<FatbinFile*>::iterator FI;

  RHAC_LOG("Create Wrapped Fatbin files from Wrapped PTX files ... ");

  for (std::vector<FatbinFile*>::iterator FI = fatbin_files.begin(),
      FE = fatbin_files.end(); FI != FE; ++FI) {
    CreateWrappedFatbin(*FI);
  }

  RHAC_LOG("Done");
}

void FatbinHandler::CreateWrappedFatbin(FatbinFile* fatbin_file) {
  char input_command[1024];
  std::string fatbin_filename;

  RHAC_LOG("\tCreate fatbin from %s", fatbin_file->ptx_filename_.c_str());

//  fatbin_filename =
//    fatbin_file->wrapped_ptx_filename_.substr(0,
//        fatbin_file->wrapped_ptx_filename_.length() - 4) +
//    ".fatbin";
  fatbin_filename = "kernels." + std::to_string(fatbin_file->file_idx_) + 
    ".fatbin";
  fatbin_file->wrapped_fatbin_filename_ = fatbin_filename;

  sprintf(input_command, "nvcc %s -fatbin %s -o %s",
      NVCC_ARCH_OPTION,
      fatbin_file->wrapped_ptx_filename_.c_str(),
      fatbin_file->wrapped_fatbin_filename_.c_str());

  ExecuteShellCommand(input_command);
}

void FatbinHandler::RegisterFatbinary(void **cuda_fatbin_handle) {
  cuda_fatbin_handles_.push_back(cuda_fatbin_handle);
}

void FatbinHandler::RegisterFunction(void **cuda_fatbin_handle,
    const char *host_func, char *device_func) {
  int cubin_index = -1;

  // find index of fatbin using handle
  for (unsigned int I = 0, E = cuda_fatbin_handles_.size();
      I != E; ++I) {
    if (cuda_fatbin_handles_[I] == cuda_fatbin_handle) {
      cubin_index = I + 1;
      break;
    }
  }

  if (cubin_index < 0) {
    fprintf(stderr, "Failed to register CUDA function \"%s\"\n", device_func);
    return;
  }

  // store information to mapping table
  // key: host_func(identifier)
  // value: index(ptx file number), device_func(function name)
  cuda_func_map_[host_func] = std::make_pair(cubin_index, device_func);
}

std::pair<int, char*> FatbinHandler::LookupFunction(const char *host_func) {
  return cuda_func_map_[host_func];
}

void FatbinHandler::RegisterVar(void **cuda_fatbin_handle,
    char *host_var, char *device_address) {
  int cubin_index = -1;

  // find index of fatbin using handle
  for (unsigned int I = 0, E = cuda_fatbin_handles_.size();
      I != E; ++I) {
    if (cuda_fatbin_handles_[I] == cuda_fatbin_handle) {
      cubin_index = I + 1;
      break;
    }
  }

  if (cubin_index < 0) {
    fprintf(stderr, "Failed to register CUDA var \"%s\"\n", device_address);
    return;
  }

  // store information to mapping table
  // key: host_var(identifier)
  // value: index(ptx file number), device_address(variable name)
  cuda_var_map_[host_var] = std::make_pair(cubin_index, device_address);
}

std::pair<int, char*> FatbinHandler::LookupVar(char *host_var) {
  return cuda_var_map_[host_var];
}

void FatbinHandler::RegisterTexture(void **cuda_fatbin_handle,
    const struct textureReference *host_var, const void *device_address) {
  int cubin_index = -1;

  // find index of fatbin using handle
  for (unsigned int I = 0, E = cuda_fatbin_handles_.size();
      I != E; ++I) {
    if (cuda_fatbin_handles_[I] == cuda_fatbin_handle) {
      cubin_index = I + 1;
      break;
    }
  }

  if (cubin_index < 0) {
    fprintf(stderr, "Failed to register CUDA texture \"%s\"\n",
        (char*)device_address);
    return;
  }

  // store information to mapping table
  // key: host_var(identifier)
  // value: index(ptx file number), device_address(variable name)
  cuda_texture_map_[host_var] =
    std::make_pair(cubin_index, (char*)device_address);
}

std::pair<int, char*> FatbinHandler::LookupTexture(
    const struct textureReference *host_var) {
  return cuda_texture_map_[host_var];
}
