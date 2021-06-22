add_clang_library(clangSnuMAPA
  InductionVariable.cpp
  MAPAFrontend.cpp
  MAPAOptions.cpp
  MAPDatabase.cpp
  MemoryAccessPattern.cpp
  SymbolicAnalysis.cpp
  )

add_dependencies(clangSnuMAPA
  ClangDriverOptions
  ClangStmtNodes
  )

target_link_libraries(clangSnuMAPA
  clangAST
  clangAnalysis
  clangBasic
  clangFrontend
  clangSnuPreprocess
  clangSnuAST
  clangSnuAnalysis
  )

option(MAPA_USE_SYMBOLIC_EXPRESSION_CONSTANT_CACHE "Use cache for constant symbolic expression" ON)
option(MAPA_AGGRESSIVE_SYMBOLIC_EXPRESSION_COMPARE "Aggressively compare symbolic expressions" OFF)

if (MAPA_USE_SYMBOLIC_EXPRESSION_CONSTANT_CACHE)
  add_definitions(-DMAPA_USE_SYMBOLIC_EXPRESSION_CONSTANT_CACHE)
endif()
if (MAPA_AGGRESSIVE_SYMBOLIC_EXPRESSION_COMPARE)
  add_definitions(-DMAPA_AGGRESSIVE_SYMBOLIC_EXPRESSION_COMPARE)
endif()
