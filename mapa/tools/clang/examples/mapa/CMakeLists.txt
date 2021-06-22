set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  MCJIT
  Support
  native
  )

add_clang_executable(mapa
  main.cpp
  )

add_dependencies(mapa
  clang-headers
  )

target_link_libraries(mapa
  PUBLIC
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangRewrite
  clangTooling
  clangSnuMAPA
  )
