#include "slang_wrapper.h"
#include "slang/slang-com-ptr.h"
#include "slang/slang.h"
#include <array>
#include <stdio.h>
#include <vector>

struct SlangWrapper {
  Slang::ComPtr<slang::IGlobalSession> globalSession;
  Slang::ComPtr<slang::ISession> session;
};

extern "C" SlangWrapper *slang_wrapper_create(void) {
  SlangWrapper *wrapper = new SlangWrapper();

  if (SLANG_FAILED(
          slang::createGlobalSession(wrapper->globalSession.writeRef()))) {
    delete wrapper;
    return nullptr;
  }

  slang::TargetDesc slangTarget{
      .format{SLANG_SPIRV},
      .profile{wrapper->globalSession->findProfile("spirv_1_4")},
  };

  slang::CompilerOptionEntry slangOptions{
      slang::CompilerOptionName::EmitSpirvDirectly,
      slang::CompilerOptionValueKind::Int,
      1,
  };

  slang::SessionDesc slangSessionDesc{
      .targets{&slangTarget},
      .targetCount{SlangInt(1)},
      .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
      .compilerOptionEntries{&slangOptions},
      .compilerOptionEntryCount{1},
  };

  if (SLANG_FAILED(wrapper->globalSession->createSession(
          slangSessionDesc, wrapper->session.writeRef()))) {
    delete wrapper;
    return nullptr;
  }

  return wrapper;
}

extern "C" void slang_wrapper_destroy(SlangWrapper *wrapper) {
  if (wrapper) {
    delete wrapper;
  }
}

extern "C" SpirvBuffer *slang_shader_compile(SlangWrapper *wrapper,
                                             const char *name,
                                             const char *file_path) {
  Slang::ComPtr<slang::IModule> slangModule{
      wrapper->session->loadModuleFromSource(name, file_path, nullptr,
                                             nullptr)};
  if (slangModule == nullptr) {
    printf("failed to load slang module: %s %s\n", name, file_path);
    exit(-1);
  };
  assert(slangModule != nullptr);
  printf("loaded slang module: %s %s\n", name, file_path);

  Slang::ComPtr<ISlangBlob> spirv;
  slangModule->getTargetCode(0, spirv.writeRef());
  assert(spirv != nullptr);
  ISlangBlob *spirv_ptr = spirv.detach();
  return new SpirvBuffer{
      .spirv_ptr = spirv_ptr,
      .code_ptr = (uint32_t *)spirv_ptr->getBufferPointer(),
      .code_size = spirv_ptr->getBufferSize(),
  };
}

extern "C" void slang_shader_free(SpirvBuffer *spirv) {
  if (spirv) {
    if (spirv->spirv_ptr) {
      ISlangBlob *spirv_blob = (ISlangBlob *)(spirv->spirv_ptr);
      spirv_blob->Release();
    }
    delete spirv;
  }
}
