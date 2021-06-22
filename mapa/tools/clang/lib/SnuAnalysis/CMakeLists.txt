add_clang_library(clangSnuAnalysis
  Dominator.cpp
  Invariance.cpp
  LiveVariables.cpp
  Loop.cpp
  MemoryAccess.cpp
  PointerAnalysis.cpp
  )

add_dependencies(clangSnuAnalysis
  ClangStmtNodes
  )

target_link_libraries(clangSnuAnalysis
  clangAST
  clangAnalysis
  clangBasic
  clangSnuAST
  )

option(MAPA_AGGRESSIVE_LOOP_ANALYSIS "Aggressively dectect induction variable in MAPA" OFF)

if (MAPA_AGGRESSIVE_LOOP_ANALYSIS)
  add_definitions(-DMAPA_AGGRESSIVE_LOOP_ANALYSIS)
endif()
