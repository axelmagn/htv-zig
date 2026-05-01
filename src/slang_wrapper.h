#ifndef SLANG_WRAPPER_H
#define SLANG_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SlangWrapper SlangWrapper;

typedef struct SpirvBuffer {
  void *spirv_ptr;
  uint32_t *code_ptr;
  size_t code_size;
} SpirvBuffer;

SlangWrapper *slang_wrapper_create(void);
void slang_wrapper_destroy(SlangWrapper *wrapper);
SpirvBuffer *slang_shader_compile(SlangWrapper *wrapper, const char *name,
                                  const char *file_path);
void slang_shader_free(SpirvBuffer *spirv);
#ifdef __cplusplus
}
#endif

#endif // SLANG_WRAPPER_H
