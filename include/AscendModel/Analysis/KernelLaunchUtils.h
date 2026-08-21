//===- KernelLaunchUtils.h - Kernel launch helper utilities -----*- C++ -*-===//
//
// Shared helpers for constructing KernelLaunchContext from pass-specific body
// models.  Each pass still owns its own body/mode feature extraction, while
// binding parsing and mode normalization live here.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_KERNELLAUNCHUTILS_H
#define ASCENDMODEL_KERNELLAUNCHUTILS_H

#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Utils.h"

#include <initializer_list>

namespace mlir {
namespace ascend {
namespace utils {

inline KernelMode parseKernelModeForLaunch(llvm::StringRef mode) {
  if (mode.equals_insensitive("aiv") || mode.equals_insensitive("vector"))
    return KernelMode::AIV;
  if (mode.equals_insensitive("aic") || mode.equals_insensitive("cube"))
    return KernelMode::AIC;
  if (mode.equals_insensitive("mix"))
    return KernelMode::Mix;
  if (mode.equals_insensitive("simd"))
    return KernelMode::SIMD;
  if (mode.equals_insensitive("simt"))
    return KernelMode::SIMT;
  if (mode.equals_insensitive("simd_simt_mix"))
    return KernelMode::SIMDSIMTMix;
  return KernelMode::Unknown;
}

inline void inferKernelModeFromLaunchFeatures(KernelLaunchContext &ctx) {
  if (ctx.hasVector && ctx.hasCube)
    ctx.mode = KernelMode::Mix;
  else if (ctx.hasCube)
    ctx.mode = KernelMode::AIC;
  else if (ctx.hasVector)
    ctx.mode = KernelMode::AIV;
}

inline int64_t getLaunchBindingOr(
    const Bindings &bindings, std::initializer_list<llvm::StringRef> keys,
    int64_t fallback = -1) {
  for (llvm::StringRef key : keys) {
    if (auto value = bindings.get<int64_t>(key))
      return *value;
  }
  return fallback;
}

inline void applyKernelLaunchBindings(KernelLaunchContext &ctx,
                                      llvm::StringRef bindingsStr) {
  auto bindingsOr = parseBindings(bindingsStr);
  if (!bindingsOr)
    return;

  const auto &bindings = *bindingsOr;
  ctx.blockDim =
      getLaunchBindingOr(bindings, {"block_dim", "blockDim", "block_num",
                                    "blockNum"});
  ctx.usingPrograms = getLaunchBindingOr(
      bindings,
      {"using_programs", "usingPrograms", "num_programs", "numPrograms"});
  ctx.numWaves = getLaunchBindingOr(bindings, {"num_waves", "numWaves"});
  if (auto mode = bindings.get<std::string>("kernel_mode")) {
    KernelMode parsedMode = parseKernelModeForLaunch(*mode);
    if (parsedMode != KernelMode::Unknown)
      ctx.mode = parsedMode;
  }
}

} // namespace utils
} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_KERNELLAUNCHUTILS_H
