//===- PipelineAnalysisPass.cpp - Pipeline scheduling analysis -----------===//
//
// Uses HardwareConfig for configurable hardware parameters.
// Handles dynamic loop bounds via arg-bindings option (supports program_id).
// Generates Perfetto trace with loop unrolling visualization.
// Uses Roofline model for cycle estimation with HW unit overlap.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_PIPELINEANALYSISPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using utils::getScfForTripCount;
using utils::getLoopMultiplier;
using utils::getScfForTripCountWithBindings;
using utils::parseBindings;
using utils::parseLoopTripCounts;

struct TileMixParams {
  bool hasAny = false;
  int64_t vectorLoop = 0;
  int64_t cubeLoop = 0;
};

struct TileMixModelConfig {
  double bufferTargetFraction = 1.0;
  double pressureGranularityFraction = 1.0;
  double handoffTargetFraction = 1.0;
  double intermediateTargetFraction = 1.0;
  double handoffMaxReliefRatio = 0.0;
  double handoffMaxLog2Steps = 1.0;
  double intermediateDtypeBytes = 0.0;
  double intermediatePressurePenaltyRatio = 0.0;
  double intermediatePressureMaxLog2Steps = 1.0;
  double loopGranularityReliefRatio = 0.0;
  double loopGranularityMaxLog2Steps = 1.0;
  double loopMismatchPenaltyRatio = 0.0;
  double syncFrequencyPenaltyRatio = 0.0;
  int64_t syncNeutralSegments = 0;
  double syncPenaltySegments = 1.0;
};

struct TileMixHandoffEstimate {
  int64_t reliefCycles = 0;
  int64_t featureDim = 0;
  int64_t dtypeBytes = 0;
  int64_t tileBytes = 0;
  int64_t targetBytes = 0;
  int64_t neutralBlockN = 0;
  int64_t intermediateTileBytes = 0;
  int64_t intermediateTargetBytes = 0;
  int64_t neutralBlockM = 0;
};

struct TileMixDerivedFeatures {
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t dtypeBytes = 0;
  int64_t handoffTileBytes = 0;
  int64_t handoffFeatureDim = 0;
  int64_t intermediateTileBytes = 0;
  std::string tileShapeSource = "none";
  std::string dtypeSource = "none";
  std::string handoffSource = "none";
  std::string intermediateSource = "none";
};

struct TileMixBalanceStats {
  bool used = false;
  bool valid = false;
  int64_t adjustedCycles = 0;
  int64_t baseCycles = 0;
  int64_t boundaryCycles = 0;
  int64_t balancePenaltyCycles = 0;
  int64_t handoffReliefCycles = 0;
  int64_t workspaceReliefCycles = 0;
  int64_t netDeltaCycles = 0;
  int64_t cubeSegmentCount = 0;
  int64_t vectorSegmentCount = 0;
  int64_t cubeLoopTrip = 0;
  int64_t vectorLoopTrip = 0;
  int64_t cubeLayoutOpCount = 0;
  int64_t vectorLayoutOpCount = 0;
  int64_t cubeWorkspaceBytes = 0;
  int64_t vectorWorkspaceBytes = 0;
  int64_t cubeSubtileBytes = 0;
  int64_t vectorSubtileBytes = 0;
  int64_t cubeTargetBytes = 0;
  int64_t vectorTargetBytes = 0;
  int64_t inferredTileM = 0;
  int64_t inferredTileN = 0;
  int64_t handoffFeatureDim = 0;
  int64_t handoffDtypeBytes = 0;
  int64_t handoffTileBytes = 0;
  int64_t handoffTargetBytes = 0;
  int64_t handoffNeutralBlockN = 0;
  int64_t intermediateTileBytes = 0;
  int64_t intermediateTargetBytes = 0;
  int64_t intermediateNeutralBlockM = 0;
  int64_t intermediatePressurePenaltyCycles = 0;
  int64_t loopGranularityReliefCycles = 0;
  int64_t loopMismatchPenaltyCycles = 0;
  int64_t bufferFitPenaltyCycles = 0;
  int64_t syncFrequencyPenaltyCycles = 0;
  std::string tileShapeSource = "none";
  std::string dtypeSource = "none";
  std::string handoffSource = "none";
  std::string intermediateSource = "none";
};

enum class TileMixPath { None = 0, Cube = 1, Vector = 2 };

bool parsePositiveInt(llvm::StringRef value, int64_t &result) {
  value = value.trim();
  if (value.empty())
    return false;
  int64_t parsed = 0;
  if (value.getAsInteger(10, parsed))
    return false;
  if (parsed <= 0)
    return false;
  result = parsed;
  return true;
}

TileMixParams parseTileMixParams(llvm::StringRef compileParamsStr) {
  TileMixParams params;
  llvm::SmallVector<llvm::StringRef, 8> items;
  compileParamsStr.split(items, ',', -1, false);
  for (llvm::StringRef item : items) {
    item = item.trim();
    if (item.empty())
      continue;
    std::pair<llvm::StringRef, llvm::StringRef> kv = item.split('=');
    llvm::StringRef key = kv.first.trim();
    llvm::StringRef value = kv.second.trim();
    int64_t parsed = 0;
    if (key == "tile_mix_vector_loop" && parsePositiveInt(value, parsed)) {
      params.vectorLoop = parsed;
      params.hasAny = true;
    } else if (key == "tile_mix_cube_loop" && parsePositiveInt(value, parsed)) {
      params.cubeLoop = parsed;
      params.hasAny = true;
    }
  }
  return params;
}

bool isTileMixDagModelEnabled() {
  const char *mode = std::getenv("ASCEND_COSTMODEL_TILE_MIX_MODEL");
  if (!mode)
    return true;
  llvm::StringRef value(mode);
  return !(value.equals_insensitive("0") || value.equals_insensitive("off") ||
           value.equals_insensitive("false") || value.equals_insensitive("none"));
}

int64_t ceilDiv(int64_t value, int64_t divisor) {
  if (value <= 0 || divisor <= 0)
    return 0;
  return (value + divisor - 1) / divisor;
}

int64_t overflowPressureCycles(int64_t bytes, int64_t targetBytes,
                               int64_t affectedCycles,
                               int64_t pressureGranularity) {
  if (bytes <= 0 || targetBytes <= 0 || affectedCycles <= 0)
    return 0;
  double ratio = static_cast<double>(bytes) /
                 static_cast<double>(targetBytes);
  if (ratio <= 1.0)
    return 0;
  double overflowDistance = std::log2(ratio);
  int64_t rawPressure =
      static_cast<int64_t>(std::llround(overflowDistance * affectedCycles));
  int64_t cap = affectedCycles / std::max<int64_t>(pressureGranularity, 1);
  return std::max<int64_t>(0, std::min<int64_t>(rawPressure, cap));
}

TileMixModelConfig getTileMixModelConfig(const HardwareConfig &config) {
  TileMixModelConfig model;
  auto readFraction = [&](llvm::StringRef key, double fallback) {
    double value = config.getCostModelParam(key, fallback);
    if (!std::isfinite(value) || value <= 0.0)
      return fallback;
    return std::min(value, 1.0);
  };
  auto readNonNegative = [&](llvm::StringRef key, double fallback) {
    double value = config.getCostModelParam(key, fallback);
    if (!std::isfinite(value) || value < 0.0)
      return fallback;
    return value;
  };
  auto readPositive = [&](llvm::StringRef key, double fallback) {
    double value = config.getCostModelParam(key, fallback);
    if (!std::isfinite(value) || value <= 0.0)
      return fallback;
    return value;
  };

  model.bufferTargetFraction =
      readFraction("tilemix_buffer_target_fraction", model.bufferTargetFraction);
  model.pressureGranularityFraction = readFraction(
      "tilemix_pressure_granularity_fraction",
      model.pressureGranularityFraction);
  model.handoffTargetFraction = readFraction(
      "tilemix_handoff_local_target_fraction",
      readFraction("tilemix_handoff_ub_target_fraction",
                   model.handoffTargetFraction));
  model.intermediateTargetFraction = readFraction(
      "tilemix_intermediate_ub_target_fraction",
      model.intermediateTargetFraction);
  model.handoffMaxReliefRatio = readNonNegative(
      "tilemix_handoff_max_relief_ratio", model.handoffMaxReliefRatio);
  model.handoffMaxLog2Steps = readPositive(
      "tilemix_handoff_max_log2_steps", model.handoffMaxLog2Steps);
  model.intermediateDtypeBytes = readNonNegative(
      "tilemix_intermediate_dtype_bytes", model.intermediateDtypeBytes);
  model.intermediatePressurePenaltyRatio = readNonNegative(
      "tilemix_intermediate_pressure_penalty_ratio",
      model.intermediatePressurePenaltyRatio);
  model.intermediatePressureMaxLog2Steps = readPositive(
      "tilemix_intermediate_pressure_max_log2_steps",
      model.intermediatePressureMaxLog2Steps);
  model.loopGranularityReliefRatio = readNonNegative(
      "tilemix_loop_granularity_relief_ratio",
      model.loopGranularityReliefRatio);
  model.loopGranularityMaxLog2Steps = readPositive(
      "tilemix_loop_granularity_max_log2_steps",
      model.loopGranularityMaxLog2Steps);
  model.loopMismatchPenaltyRatio = readNonNegative(
      "tilemix_loop_mismatch_penalty_ratio",
      model.loopMismatchPenaltyRatio);
  model.syncFrequencyPenaltyRatio = readNonNegative(
      "tilemix_sync_frequency_penalty_ratio",
      model.syncFrequencyPenaltyRatio);
  model.syncNeutralSegments = std::max<int64_t>(
      0, config.getCostModelIntParam("tilemix_sync_neutral_segments",
                                     model.syncNeutralSegments));
  model.syncPenaltySegments = readPositive(
      "tilemix_sync_penalty_segments", model.syncPenaltySegments);
  return model;
}

int64_t scaledLocalTargetBytes(int64_t localBytes, double fraction) {
  if (localBytes <= 0 || fraction <= 0.0)
    return 1;
  return std::max<int64_t>(
      1, static_cast<int64_t>(
             std::llround(static_cast<double>(localBytes) * fraction)));
}

int64_t minPositiveLocalBytes(int64_t lhs, int64_t rhs) {
  if (lhs <= 0)
    return rhs;
  if (rhs <= 0)
    return lhs;
  return std::min(lhs, rhs);
}

int64_t getElementByteWidth(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (type.isIntOrFloat())
    return (type.getIntOrFloatBitWidth() + 7) / 8;
  if (type.isIndex())
    return 8;
  return 0;
}

int64_t getStaticShapedTypeBytes(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  int64_t elemBytes = getElementByteWidth(shaped.getElementType());
  if (elemBytes <= 0)
    return 0;
  int64_t elements = 1;
  for (int64_t dim : shaped.getShape())
    elements *= dim;
  return elements * elemBytes;
}

void updateTopTwoLoads(int64_t bytes, int64_t dtypeBytes,
                       int64_t &top1Bytes, int64_t &top1DtypeBytes,
                       int64_t &top2Bytes, int64_t &top2DtypeBytes) {
  if (bytes <= 0)
    return;
  if (bytes > top1Bytes) {
    top2Bytes = top1Bytes;
    top2DtypeBytes = top1DtypeBytes;
    top1Bytes = bytes;
    top1DtypeBytes = dtypeBytes;
    return;
  }
  if (bytes > top2Bytes) {
    top2Bytes = bytes;
    top2DtypeBytes = dtypeBytes;
  }
}

TileMixDerivedFeatures inferTileMixDerivedFeatures(
    const PipelineScheduler &scheduler) {
  TileMixDerivedFeatures features;
  int64_t topVectorLoadBytes = 0;
  int64_t topVectorLoadDtypeBytes = 0;
  int64_t secondVectorLoadBytes = 0;
  int64_t secondVectorLoadDtypeBytes = 0;
  int64_t maxBoundaryBytes = 0;
  int64_t maxBoundaryDtypeBytes = 0;
  int64_t maxMatmulResultBytes = 0;

  for (const auto &op : scheduler.getAllOps()) {
    if (!op.mlirOp)
      continue;

    if (auto matmulOp = dyn_cast<MatmulOp>(op.mlirOp)) {
      if (matmulOp.getM() > 0 && matmulOp.getN() > 0) {
        features.tileM = std::max<int64_t>(features.tileM, matmulOp.getM());
        features.tileN = std::max<int64_t>(features.tileN, matmulOp.getN());
        features.tileShapeSource = "matmul_attrs";
      }
      int64_t lhsDtypeBytes = getElementByteWidth(matmulOp.getLhs().getType());
      if (lhsDtypeBytes > 0 && features.dtypeBytes <= 0) {
        features.dtypeBytes = lhsDtypeBytes;
        features.dtypeSource = "matmul_lhs_type";
      }
      int64_t resultBytes =
          getStaticShapedTypeBytes(matmulOp->getResult(0).getType());
      if (resultBytes > maxMatmulResultBytes) {
        maxMatmulResultBytes = resultBytes;
        features.intermediateTileBytes = resultBytes;
        features.intermediateSource = "matmul_result_type";
      }
      continue;
    }

    if (auto loadOp = dyn_cast<VectorLoadOp>(op.mlirOp)) {
      int64_t bytes = std::max<int64_t>(loadOp.getTransferBytes(), 0);
      int64_t dtypeBytes =
          getElementByteWidth(loadOp->getResult(0).getType());
      Operation *sourceDef = loadOp.getSource().getDefiningOp();
      if (sourceDef && isa<MatmulOp>(sourceDef)) {
        if (bytes > maxBoundaryBytes) {
          maxBoundaryBytes = bytes;
          maxBoundaryDtypeBytes = dtypeBytes;
        }
      } else {
        updateTopTwoLoads(bytes, dtypeBytes, topVectorLoadBytes,
                          topVectorLoadDtypeBytes, secondVectorLoadBytes,
                          secondVectorLoadDtypeBytes);
      }
      continue;
    }

    if (auto storeOp = dyn_cast<CubeStoreOp>(op.mlirOp)) {
      int64_t bytes = std::max<int64_t>(storeOp.getTransferBytes(), 0);
      int64_t dtypeBytes = getElementByteWidth(storeOp.getData().getType());
      if (bytes > maxBoundaryBytes) {
        maxBoundaryBytes = bytes;
        maxBoundaryDtypeBytes = dtypeBytes;
      }
    }
  }

  if (topVectorLoadBytes > 0) {
    features.handoffTileBytes = topVectorLoadBytes + secondVectorLoadBytes;
    features.handoffSource =
        secondVectorLoadBytes > 0 ? "vector_load_top2" : "vector_load_top1";
    if (topVectorLoadDtypeBytes > 0) {
      features.dtypeBytes = topVectorLoadDtypeBytes;
      features.dtypeSource = "vector_load_type";
    } else if (secondVectorLoadDtypeBytes > 0 && features.dtypeBytes <= 0) {
      features.dtypeBytes = secondVectorLoadDtypeBytes;
      features.dtypeSource = "vector_load_type";
    }
  } else if (maxBoundaryBytes > 0) {
    features.handoffTileBytes = maxBoundaryBytes;
    features.handoffSource = "cube_vector_boundary";
    if (maxBoundaryDtypeBytes > 0 && features.dtypeBytes <= 0) {
      features.dtypeBytes = maxBoundaryDtypeBytes;
      features.dtypeSource = "boundary_type";
    }
  }

  if (features.intermediateTileBytes <= 0 && maxBoundaryBytes > 0) {
    features.intermediateTileBytes = maxBoundaryBytes;
    features.intermediateSource = "cube_vector_boundary";
  }

  if (features.handoffTileBytes > 0 && features.tileN > 0 &&
      features.dtypeBytes > 0) {
    int64_t bytesPerTileN = features.tileN * features.dtypeBytes;
    if (bytesPerTileN > 0)
      features.handoffFeatureDim =
          std::max<int64_t>(1, features.handoffTileBytes / bytesPerTileN);
  }

  return features;
}

TileMixHandoffEstimate estimateTileMixHandoffRelief(
    int64_t baseCycles, const TileMixDerivedFeatures &features,
    const TileMixModelConfig &model, int64_t l0cBytes, int64_t ubBytes) {
  TileMixHandoffEstimate estimate;
  estimate.featureDim = features.handoffFeatureDim;
  estimate.dtypeBytes = features.dtypeBytes;
  estimate.tileBytes = features.handoffTileBytes;
  estimate.intermediateTileBytes = features.intermediateTileBytes;
  int64_t localBytes = minPositiveLocalBytes(l0cBytes, ubBytes);
  estimate.targetBytes =
      scaledLocalTargetBytes(localBytes, model.handoffTargetFraction);
  estimate.intermediateTargetBytes =
      scaledLocalTargetBytes(ubBytes, model.intermediateTargetFraction);
  if (estimate.featureDim > 0 && estimate.dtypeBytes > 0) {
    int64_t bytesPerBlockN = estimate.featureDim * estimate.dtypeBytes;
    if (bytesPerBlockN > 0)
      estimate.neutralBlockN =
          std::max<int64_t>(1, estimate.targetBytes / bytesPerBlockN);
  }
  int64_t intermediateDtypeBytes = static_cast<int64_t>(
      std::llround(model.intermediateDtypeBytes > 0.0
                       ? model.intermediateDtypeBytes
                       : static_cast<double>(estimate.dtypeBytes)));
  if (features.tileN > 0 && intermediateDtypeBytes > 0) {
    int64_t bytesPerBlockM = features.tileN * intermediateDtypeBytes;
    if (bytesPerBlockM > 0)
      estimate.neutralBlockM = std::max<int64_t>(
          1, estimate.intermediateTargetBytes / bytesPerBlockM);
  }
  if (estimate.tileBytes <= 0 || baseCycles <= 0 ||
      model.handoffMaxReliefRatio <= 0.0)
    return estimate;
  if (estimate.tileBytes <= estimate.targetBytes)
    return estimate;

  double footprintRatio = static_cast<double>(estimate.tileBytes) /
                          static_cast<double>(estimate.targetBytes);
  double steps = std::log2(std::max(footprintRatio, 1.0));
  double bounded = std::min(std::max(steps / model.handoffMaxLog2Steps, 0.0),
                            1.0);
  estimate.reliefCycles = static_cast<int64_t>(std::llround(
      static_cast<double>(baseCycles) * model.handoffMaxReliefRatio *
      bounded));
  return estimate;
}

double clampUnit(double value) {
  if (!std::isfinite(value))
    return 0.0;
  return std::min(std::max(value, 0.0), 1.0);
}

int64_t tileMixIntermediatePressurePenaltyCycles(
    int64_t baseCycles, const TileMixHandoffEstimate &handoff,
    const TileMixModelConfig &model) {
  if (baseCycles <= 0 || handoff.intermediateTileBytes <= 0 ||
      handoff.intermediateTargetBytes <= 0 ||
      model.intermediatePressurePenaltyRatio <= 0.0)
    return 0;
  double ratio = static_cast<double>(handoff.intermediateTileBytes) /
                 static_cast<double>(handoff.intermediateTargetBytes);
  if (ratio <= 1.0)
    return 0;
  double bounded =
      clampUnit(std::log2(ratio) / model.intermediatePressureMaxLog2Steps);
  return static_cast<int64_t>(std::llround(
      static_cast<double>(baseCycles) *
      model.intermediatePressurePenaltyRatio * bounded));
}

int64_t tileMixLoopGranularityReliefCycles(
    int64_t baseCycles, int64_t cubeSegments, int64_t vectorSegments,
    const TileMixHandoffEstimate &handoff, const TileMixModelConfig &model) {
  if (baseCycles <= 0 || cubeSegments <= 1 || vectorSegments <= 1 ||
      handoff.tileBytes <= 0 || handoff.targetBytes <= 0 ||
      model.loopGranularityReliefRatio <= 0.0)
    return 0;

  double handoffFillRatio = clampUnit(static_cast<double>(handoff.tileBytes) /
                                      static_cast<double>(handoff.targetBytes));
  double window = std::sqrt(static_cast<double>(cubeSegments) *
                            static_cast<double>(vectorSegments));
  double bounded =
      clampUnit(std::log2(std::max(window, 1.0)) /
                model.loopGranularityMaxLog2Steps);
  double balance =
      std::sqrt(static_cast<double>(std::min(cubeSegments, vectorSegments)) /
                static_cast<double>(std::max(cubeSegments, vectorSegments)));
  return static_cast<int64_t>(std::llround(
      static_cast<double>(baseCycles) * model.loopGranularityReliefRatio *
      handoffFillRatio * bounded * balance));
}

int64_t tileMixLoopMismatchPenaltyCycles(int64_t baseCycles,
                                         int64_t cubeSegments,
                                         int64_t vectorSegments,
                                         const TileMixModelConfig &model) {
  if (baseCycles <= 0 || cubeSegments <= 0 || vectorSegments <= 0 ||
      model.loopMismatchPenaltyRatio <= 0.0)
    return 0;
  double mismatch =
      std::abs(std::log2(static_cast<double>(cubeSegments) /
                         static_cast<double>(vectorSegments)));
  if (mismatch <= 0.0)
    return 0;
  double bounded = mismatch / (1.0 + mismatch);
  return static_cast<int64_t>(std::llround(
      static_cast<double>(baseCycles) * model.loopMismatchPenaltyRatio *
      bounded));
}

int64_t tileMixSyncFrequencyPenaltyCycles(int64_t baseCycles,
                                          int64_t cubeSegments,
                                          int64_t vectorSegments,
                                          const TileMixModelConfig &model) {
  if (baseCycles <= 0)
    return 0;
  int64_t extraSegments =
      std::max<int64_t>(0, cubeSegments + vectorSegments -
                               model.syncNeutralSegments);
  if (extraSegments <= 0)
    return 0;
  return static_cast<int64_t>(std::llround(
      static_cast<double>(baseCycles) * model.syncFrequencyPenaltyRatio *
      static_cast<double>(extraSegments) / model.syncPenaltySegments));
}

TileMixPath getTileMixPath(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:
    case HWUnit::CubeMTE2:
    case HWUnit::FixPipe:
      return TileMixPath::Cube;
    case HWUnit::Vector:
    case HWUnit::VecMTE2:
    case HWUnit::MTE3:
      return TileMixPath::Vector;
    default:
      return TileMixPath::None;
  }
}

bool isTileMixLayoutOp(HWUnit unit) {
  return unit == HWUnit::CubeMTE2 || unit == HWUnit::FixPipe ||
         unit == HWUnit::VecMTE2 || unit == HWUnit::MTE3;
}

TileMixBalanceStats estimateTileMixLoopTilingBalance(
    const TileMixParams &params, const PipelineScheduler &scheduler,
    const HardwareConfig &config, int64_t cubePathCycles,
    int64_t vectorPathCycles, int64_t cubeTransferCycles,
    int64_t vectorTransferCycles, int64_t baseCycles) {
  TileMixBalanceStats stats;
  stats.baseCycles = baseCycles;
  stats.adjustedCycles = baseCycles;
  if (!params.hasAny || !isTileMixDagModelEnabled())
    return stats;

  for (const auto &op : scheduler.getAllOps()) {
    TileMixPath path = getTileMixPath(op.hwUnit);
    if (path == TileMixPath::None)
      continue;
    int64_t loopMultiplier = std::max<int64_t>(op.loopMultiplier, 1);
    if (path == TileMixPath::Cube) {
      stats.cubeLoopTrip = std::max(stats.cubeLoopTrip, loopMultiplier);
      if (isTileMixLayoutOp(op.hwUnit))
        ++stats.cubeLayoutOpCount;
      if (op.mlirOp) {
        if (auto storeOp = dyn_cast<CubeStoreOp>(op.mlirOp))
          stats.cubeWorkspaceBytes +=
              std::max<int64_t>(storeOp.getTransferBytes(), 0);
      }
    } else {
      stats.vectorLoopTrip = std::max(stats.vectorLoopTrip, loopMultiplier);
      if (isTileMixLayoutOp(op.hwUnit))
        ++stats.vectorLayoutOpCount;
      if (op.mlirOp) {
        if (auto loadOp = dyn_cast<VectorLoadOp>(op.mlirOp))
          stats.vectorWorkspaceBytes +=
              std::max<int64_t>(loadOp.getTransferBytes(), 0);
        if (auto storeOp = dyn_cast<VectorStoreOp>(op.mlirOp))
          stats.vectorWorkspaceBytes +=
              std::max<int64_t>(storeOp.getTransferBytes(), 0);
      }
    }
  }

  auto computeSegments = [&](int64_t pathCycles, int64_t loopTrip,
                             int64_t tileMixLoop) {
    if (pathCycles <= 0)
      return int64_t(0);
    if (tileMixLoop <= 1)
      return int64_t(1);

    // TileCubeVectorLoop uses tile_mix_*_loop as the target loop trip count
    // after splitting.  A larger value means finer CopyOut/producer tiling; 1
    // means no extra tiling.  The previous model interpreted the option in the
    // opposite direction, which made coarse tiling look artificially better.
    if (loopTrip > 0)
      return std::max<int64_t>(1, std::min<int64_t>(tileMixLoop, loopTrip));
    return std::max<int64_t>(1, tileMixLoop);
  };

  stats.cubeSegmentCount = computeSegments(
      cubePathCycles, stats.cubeLoopTrip, params.cubeLoop);
  stats.vectorSegmentCount = computeSegments(
      vectorPathCycles, stats.vectorLoopTrip, params.vectorLoop);

  int64_t l0cBytes = static_cast<int64_t>(config.getMemorySizeBytes("l0c"));
  int64_t ubBytes = static_cast<int64_t>(config.getMemorySizeBytes("ub"));
  if (l0cBytes <= 0)
    l0cBytes = 256 * 1024;
  if (ubBytes <= 0)
    ubBytes = 256 * 1024;
  TileMixModelConfig model = getTileMixModelConfig(config);
  TileMixDerivedFeatures features = inferTileMixDerivedFeatures(scheduler);

  int64_t rawCubeWorkspaceBytes =
      std::max<int64_t>(stats.cubeWorkspaceBytes, 1);
  int64_t rawVectorWorkspaceBytes =
      std::max<int64_t>(stats.vectorWorkspaceBytes, 1);

  // The pass works by slicing CopyOut and its producers so that each sub-tile
  // fits the local buffer regime.  L0C/UB cannot be treated as fully available
  // to one value: CV pipelining, alignment, and co-live producer temporaries
  // all consume part of it.  The available fraction is a chip-level cost-model
  // parameter, so different Ascend generations can tune it without changing
  // the formula.
  stats.cubeTargetBytes =
      scaledLocalTargetBytes(l0cBytes, model.bufferTargetFraction);
  stats.vectorTargetBytes =
      scaledLocalTargetBytes(ubBytes, model.bufferTargetFraction);
  stats.cubeSubtileBytes =
      ceilDiv(rawCubeWorkspaceBytes, std::max<int64_t>(stats.cubeSegmentCount, 1));
  stats.vectorSubtileBytes =
      ceilDiv(rawVectorWorkspaceBytes, std::max<int64_t>(stats.vectorSegmentCount, 1));

  // TileCubeVectorLoop is meant to reduce local-buffer/workspace pressure by
  // slicing CopyOut and its producers.  Model that as a before/after pressure
  // delta instead of an always-positive penalty: only bytes above the effective
  // local-buffer target create pressure, and finer tilemix can reduce it.
  //
  // The adjustment is bounded by transfer-side work and amortized by one
  // vector alignment quantum.  This keeps compile-parameter effects local: they
  // can break ties within a tiling family, but they should not overwhelm the
  // primary roofline signal when TTIR cannot prove a real spill.
  int64_t pressureGranularity = scaledLocalTargetBytes(
      config.getVectorWidthBytes(), model.pressureGranularityFraction);
  int64_t cubeBeforePressure = overflowPressureCycles(
      rawCubeWorkspaceBytes, stats.cubeTargetBytes, cubeTransferCycles,
      pressureGranularity);
  int64_t cubeAfterPressure = overflowPressureCycles(
      stats.cubeSubtileBytes, stats.cubeTargetBytes, cubeTransferCycles,
      pressureGranularity);
  int64_t vectorBeforePressure = overflowPressureCycles(
      rawVectorWorkspaceBytes, stats.vectorTargetBytes, vectorTransferCycles,
      pressureGranularity);
  int64_t vectorAfterPressure = overflowPressureCycles(
      stats.vectorSubtileBytes, stats.vectorTargetBytes, vectorTransferCycles,
      pressureGranularity);

  int64_t cubeDelta = cubeAfterPressure - cubeBeforePressure;
  int64_t vectorDelta = vectorAfterPressure - vectorBeforePressure;
  int64_t adjustedCubePath = std::max<int64_t>(1, cubePathCycles + cubeDelta);
  int64_t adjustedVectorPath =
      std::max<int64_t>(1, vectorPathCycles + vectorDelta);
  int64_t pressureAdjustedCycles =
      std::max<int64_t>(baseCycles, std::max(adjustedCubePath, adjustedVectorPath));
  TileMixHandoffEstimate handoff = estimateTileMixHandoffRelief(
      baseCycles, features, model, l0cBytes, ubBytes);
  int64_t intermediatePressurePenalty =
      tileMixIntermediatePressurePenaltyCycles(baseCycles, handoff, model);
  int64_t loopGranularityRelief = tileMixLoopGranularityReliefCycles(
      baseCycles, stats.cubeSegmentCount, stats.vectorSegmentCount, handoff,
      model);
  int64_t loopMismatchPenalty = tileMixLoopMismatchPenaltyCycles(
      baseCycles, stats.cubeSegmentCount, stats.vectorSegmentCount, model);
  int64_t segmentSyncPenalty = tileMixSyncFrequencyPenaltyCycles(
      baseCycles, stats.cubeSegmentCount, stats.vectorSegmentCount, model);
  if (segmentSyncPenalty > 0 && handoff.tileBytes > 0 &&
      handoff.targetBytes > 0) {
    double handoffFillRatio = static_cast<double>(handoff.tileBytes) /
                              static_cast<double>(handoff.targetBytes);
    double uncoveredSyncRatio =
        1.0 - std::min(std::max(handoffFillRatio, 0.0), 1.0);
    segmentSyncPenalty = static_cast<int64_t>(std::llround(
        static_cast<double>(segmentSyncPenalty) * uncoveredSyncRatio));
  }

  stats.boundaryCycles = 0;
  stats.balancePenaltyCycles = 0;
  stats.inferredTileM = features.tileM;
  stats.inferredTileN = features.tileN;
  stats.handoffReliefCycles = handoff.reliefCycles;
  stats.handoffFeatureDim = handoff.featureDim;
  stats.handoffDtypeBytes = handoff.dtypeBytes;
  stats.handoffTileBytes = handoff.tileBytes;
  stats.handoffTargetBytes = handoff.targetBytes;
  stats.handoffNeutralBlockN = handoff.neutralBlockN;
  stats.intermediateTileBytes = handoff.intermediateTileBytes;
  stats.intermediateTargetBytes = handoff.intermediateTargetBytes;
  stats.intermediateNeutralBlockM = handoff.neutralBlockM;
  stats.intermediatePressurePenaltyCycles = intermediatePressurePenalty;
  stats.loopGranularityReliefCycles = loopGranularityRelief;
  stats.loopMismatchPenaltyCycles = loopMismatchPenalty;
  stats.syncFrequencyPenaltyCycles = segmentSyncPenalty;
  stats.tileShapeSource = features.tileShapeSource;
  stats.dtypeSource = features.dtypeSource;
  stats.handoffSource = features.handoffSource;
  stats.intermediateSource = features.intermediateSource;
  stats.workspaceReliefCycles =
      std::max<int64_t>(0, cubeBeforePressure - cubeAfterPressure) +
      std::max<int64_t>(0, vectorBeforePressure - vectorAfterPressure);
  stats.bufferFitPenaltyCycles = cubeAfterPressure + vectorAfterPressure;
  stats.adjustedCycles = std::max<int64_t>(
      1, pressureAdjustedCycles + stats.syncFrequencyPenaltyCycles -
             stats.handoffReliefCycles + stats.intermediatePressurePenaltyCycles +
             stats.loopMismatchPenaltyCycles - stats.loopGranularityReliefCycles);
  stats.netDeltaCycles = stats.adjustedCycles - baseCycles;
  stats.used = true;
  stats.valid = true;
  return stats;
}

int getTrackId(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:     return 1;
    case HWUnit::CubeMTE2: return 2;
    case HWUnit::FixPipe:  return 3;
    case HWUnit::Vector:   return 4;
    case HWUnit::VecMTE2:  return 5;
    case HWUnit::MTE3:     return 6;
    case HWUnit::Scalar:   return 7;
    default:               return 0;
  }
}

const char* getColorName(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:     return "rail_response";
    case HWUnit::CubeMTE2: return "rail_load";
    case HWUnit::FixPipe:  return "cq_build_passed";
    case HWUnit::Vector:   return "rail_animation";
    case HWUnit::VecMTE2:  return "good";
    case HWUnit::MTE3:     return "bad";
    case HWUnit::Scalar:   return "grey";
    default:               return "generic_work";
  }
}

HWUnit getOpHWUnit(Operation *op) {
  if (isa<MatmulOp>(op)) return HWUnit::Cube;
  if (isa<CubeLoadOp>(op)) return HWUnit::CubeMTE2;
  if (isa<CubeStoreOp>(op)) return HWUnit::FixPipe;
  if (isa<VectorLoadOp>(op)) return HWUnit::VecMTE2;
  if (isa<VectorStoreOp>(op)) return HWUnit::MTE3;
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp,
          ExpOp, LogOp, SqrtOp, RsqrtOp, TanhOp, SigmoidOp,
          NegOp, AbsOp, ReluOp, CastOp,
          ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp,
          BroadcastOp, SelectOp>(op))
    return HWUnit::Vector;
  return HWUnit::Scalar;
}

/// Generate Perfetto trace with loop unrolling.
/// If maxIterations > 0, limits the number of iterations shown in trace.
void generatePerfettoTrace(const PipelineScheduler &scheduler,
                           StringRef filename,
                           int64_t oneIterCycles,
                           int64_t totalCycles,
                           int64_t maxIterations = 100) {
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
    return;
  }
  
  const auto &config = scheduler.getConfig();
  const auto &allOps = scheduler.getAllOps();
  
  // Calculate the maximum loop multiplier to determine iteration count
  int64_t maxLoopMultiplier = 1;
  for (const auto &op : allOps) {
    maxLoopMultiplier = std::max(maxLoopMultiplier, op.loopMultiplier);
  }
  
  // Limit iterations for visualization (avoid huge traces)
  int64_t numIterations = std::min(maxLoopMultiplier, maxIterations);
  bool truncated = (maxLoopMultiplier > maxIterations);
  
  double cycleToUs = 1.0;  // 1 cycle = 1 unit for visualization
  
  file << "{\n  \"traceEvents\": [\n";
  bool first = true;
  
  // Track metadata
  struct TrackInfo { int tid; const char* name; };
  TrackInfo tracks[] = {
    {1, "Cube Core"}, {2, "Cube MTE2 (HBM->L1)"}, {3, "FixPipe (L0C->HBM)"},
    {4, "Vector Core"}, {5, "Vec MTE2 (HBM->UB)"}, {6, "MTE3 (UB->HBM)"}, {7, "Scalar"}
  };
  
  // Write track metadata
  for (const auto &track : tracks) {
    if (!first) file << ",\n";
    first = false;
    file << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": " 
         << track.tid << ", \"args\": {\"name\": \"" << track.name << "\"}}";
  }
  
  for (const auto &track : tracks) {
    file << ",\n    {\"name\": \"thread_sort_index\", \"ph\": \"M\", \"pid\": 1, \"tid\": " 
         << track.tid << ", \"args\": {\"sort_index\": " << track.tid << "}}";
  }
  
  file << ",\n    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, "
       << "\"args\": {\"name\": \"" << config.getName().str() << " Pipeline";
  if (truncated) {
    file << " (showing " << numIterations << "/" << maxLoopMultiplier << " iterations)";
  }
  file << "\"}}";
  
  // Calculate actual total cycles shown in trace
  int64_t traceTotalCycles = 0;
  
  // Generate events for each iteration
  // Key insight: operations with different loopMultipliers execute different numbers of times
  // We need to track per-HW-unit time to model pipeline parallelism across iterations
  
  // Track end time for each hardware unit
  llvm::DenseMap<HWUnit, int64_t> hwUnitEndTime;
  for (int i = 0; i <= static_cast<int>(HWUnit::Scalar); ++i) {
    hwUnitEndTime[static_cast<HWUnit>(i)] = 0;
  }
  
  // For each iteration
  for (int64_t iter = 0; iter < numIterations; ++iter) {
    // Track dependencies within this iteration
    llvm::DenseMap<int64_t, int64_t> opEndTimes;  // opId -> endTime in this iter
    
    for (const auto &op : allOps) {
      // Check if this op executes in this iteration
      // An op with loopMultiplier=N executes N times
      if (iter >= op.loopMultiplier)
        continue;
      
      // Calculate start time considering:
      // 1. Dependencies from previous ops in this iteration
      // 2. Hardware unit availability
      int64_t startTime = hwUnitEndTime[op.hwUnit];
      
      // Check dependencies
      for (int64_t depId : op.dependsOn) {
        auto it = opEndTimes.find(depId);
        if (it != opEndTimes.end()) {
          startTime = std::max(startTime, it->second);
        }
      }
      
      int64_t endTime = startTime + op.duration;
      
      // Update tracking
      hwUnitEndTime[op.hwUnit] = endTime;
      opEndTimes[op.opId] = endTime;
      traceTotalCycles = std::max(traceTotalCycles, endTime);
      
      // Write event
      int tid = getTrackId(op.hwUnit);
      file << ",\n    {\"name\": \"" << op.opName;
      if (op.loopMultiplier > 1) {
        file << "[" << iter << "]";  // Show iteration number
      }
      file << "\", "
           << "\"cat\": \"" << stringifyHWUnit(op.hwUnit).str() << "\", \"ph\": \"X\", "
           << "\"ts\": " << llvm::format("%.3f", startTime * cycleToUs) << ", "
           << "\"dur\": " << llvm::format("%.3f", op.duration * cycleToUs) << ", "
           << "\"pid\": 1, \"tid\": " << tid << ", "
           << "\"cname\": \"" << getColorName(op.hwUnit) << "\", "
           << "\"args\": {"
           << "\"op_id\": " << op.opId << ", "
           << "\"iteration\": " << iter << ", "
           << "\"cycles\": " << op.duration << ", "
           << "\"loop_multiplier\": " << op.loopMultiplier
           << "}}";
    }
  }
  
  // Add markers for total timeline
  for (const auto &track : tracks) {
    file << ",\n    {\"name\": \"\", \"cat\": \"marker\", \"ph\": \"i\", \"s\": \"t\", "
         << "\"ts\": 0, \"pid\": 1, \"tid\": " << track.tid << "}";
    file << ",\n    {\"name\": \"\", \"cat\": \"marker\", \"ph\": \"i\", \"s\": \"t\", "
         << "\"ts\": " << llvm::format("%.3f", traceTotalCycles * cycleToUs) 
         << ", \"pid\": 1, \"tid\": " << track.tid << "}";
  }
  
  // Add iteration markers
  if (numIterations > 1) {
    // Add counter track for iteration progress
    file << ",\n    {\"name\": \"Iterations\", \"ph\": \"C\", \"ts\": 0, \"pid\": 1, "
         << "\"args\": {\"shown\": " << numIterations << ", \"total\": " << maxLoopMultiplier << "}}";
  }
  
  file << "\n  ],\n";
  
  // Metadata
  file << "  \"metadata\": {\n";
  file << "    \"hardware\": \"" << config.getName().str() << "\",\n";
  file << "    \"one_iter_cycles\": " << oneIterCycles << ",\n";
  file << "    \"total_cycles\": " << totalCycles << ",\n";
  file << "    \"trace_cycles\": " << traceTotalCycles << ",\n";
  file << "    \"iterations_shown\": " << numIterations << ",\n";
  file << "    \"iterations_total\": " << maxLoopMultiplier << ",\n";
  file << "    \"clock_freq_ghz\": " << config.getClockFrequencyGHz() << ",\n";
  file << "    \"estimated_time_us\": " << llvm::format("%.3f", config.cyclesToMicroseconds(totalCycles)) << "\n";
  file << "  },\n";
  
  file << "  \"displayTimeUnit\": \"ns\"\n";
  file << "}\n";
  
  file.close();
  
  llvm::outs() << "Perfetto trace: " << filename << "\n";
  if (truncated) {
    llvm::outs() << "  Note: Showing " << numIterations << " of " << maxLoopMultiplier 
                 << " iterations (use full trace for complete view)\n";
  }
  llvm::outs() << "  Trace cycles: " << traceTotalCycles << " (";
  if (truncated) {
    llvm::outs() << "partial, ";
  }
  llvm::outs() << "actual total: " << totalCycles << ")\n";
  llvm::outs() << "  Open with: https://ui.perfetto.dev/\n";
}

struct PipelineAnalysisPass
    : public impl::PipelineAnalysisPassBase<PipelineAnalysisPass> {
  using PipelineAnalysisPassBase::PipelineAnalysisPassBase;
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Load an independent, validated hardware config for this analysis without
    // mutating process-global state (back-port of triton-ascend #337).
    std::string hardwareConfigError;
    auto hardwareConfig =
        loadHardwareConfigForAnalysis(hardwareConfigPath, hardwareConfigError);
    if (!hardwareConfig) {
      emitError(module.getLoc(), hardwareConfigError);
      return signalPassFailure();
    }
    const HardwareConfig &config = *hardwareConfig;

    // Parse bindings
    llvm::DenseMap<unsigned, int64_t> argBindings;
    llvm::StringMap<int64_t> programIdBindings;
    SmallVector<int64_t> loopTripCountOverrides;
    
    if (!argBindingsStr.empty()) {
      std::string parseError;
      if (!parseBindings(argBindingsStr, argBindings, programIdBindings, parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }
    
    if (!loopTripCountsStr.empty()) {
      std::string parseError;
      if (!parseLoopTripCounts(loopTripCountsStr, loopTripCountOverrides, parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }

    TileMixParams tileMixParams = parseTileMixParams(compileParamsStr);
    
    // Collect loops and ensure trip counts are set
    SmallVector<scf::ForOp> allLoops;
    module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });
    
    bool hasError = false;
    for (size_t loopIdx = 0; loopIdx < allLoops.size(); ++loopIdx) {
      scf::ForOp forOp = allLoops[loopIdx];
      
      if (forOp->hasAttr("ascend.trip_count"))
        continue;
      
      int64_t tripCount = 1;
      if (loopIdx < loopTripCountOverrides.size()) {
        tripCount = loopTripCountOverrides[loopIdx];
      } else {
        auto result = getScfForTripCountWithBindings(forOp, argBindings, programIdBindings);
        if (result.isStatic) {
          tripCount = result.staticTripCount;
        } else {
          emitError(forOp.getLoc(), "Loop " + std::to_string(loopIdx) + 
                    " trip count unknown. " + result.errorMsg);
          hasError = true;
          continue;
        }
      }
      
      forOp->setAttr("ascend.trip_count",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64), tripCount));
    }
    
    if (hasError) return signalPassFailure();
    
    // Build scheduler
    PipelineScheduler scheduler(&config);
    llvm::DenseMap<Value, int64_t> valueProducers;
    
    module.walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::YieldOp, scf::IfOp>(op)) return;
      
      auto opIdAttr = op->getAttrOfType<IntegerAttr>("op_id");
      if (!opIdAttr) return;
      
      int64_t opId = opIdAttr.getInt();
      auto cyclesAttr = op->getAttrOfType<IntegerAttr>("estimated_cycles");
      int64_t cycles = cyclesAttr ? cyclesAttr.getInt() : 1;
      
      PipelineOp pipelineOp;
      pipelineOp.opId = opId;
      pipelineOp.hwUnit = getOpHWUnit(op);
      pipelineOp.duration = cycles;
      pipelineOp.mlirOp = op;
      pipelineOp.opName = op->getName().getStringRef().str();
      pipelineOp.loopMultiplier = getLoopMultiplier(op);
      
      for (Value operand : op->getOperands()) {
        auto it = valueProducers.find(operand);
        if (it != valueProducers.end()) {
          pipelineOp.dependsOn.push_back(it->second);
          scheduler.addDependency(it->second, opId);
        }
      }
      
      for (Value result : op->getResults())
        valueProducers[result] = opId;
      
      scheduler.addOperation(pipelineOp);
    });
    
    if (!scheduler.schedule()) {
      emitError(module.getLoc(), "Failed to schedule pipeline");
      return signalPassFailure();
    }
    
    // Calculate cycles using roofline model
    // oneIterCycles from scheduler already considers HW unit parallelism for one iteration
    int64_t oneIterCycles = scheduler.getTotalCycles();
    
    // For total cycles with loops, we need to consider:
    // 1. Each HW unit's total work across all iterations
    // 2. Take max (not sum) since they can overlap
    
    // Collect per-HW-unit cycles
    llvm::DenseMap<HWUnit, int64_t> hwUnitCycles;
    for (const auto &pipelineOp : scheduler.getAllOps()) {
      hwUnitCycles[pipelineOp.hwUnit] += pipelineOp.duration * pipelineOp.loopMultiplier;
    }
    
    // Group by path and apply roofline model.
    // Cube path: max(Cube, CubeMTE2, FixPipe). No cube-side mutex on 910B
    // (tilesim pipe_exclusive_config only pairs AIV MTE2<->MTE3).
    int64_t cubePathCycles = std::max({
      hwUnitCycles[HWUnit::Cube],
      hwUnitCycles[HWUnit::CubeMTE2],
      hwUnitCycles[HWUnit::FixPipe]
    });

    // Vector path. Vector compute overlaps with load/store transfers, but on
    // 910B AIV the MTE2 (load) and MTE3 (store) units share one physical
    // pipeline (tilesim MutexComponents) and must serialize. With the mutex
    // the transfer time is VecMTE2 + MTE3; without it (legacy) they are
    // assumed to overlap (max). This is root cause 3.
    int64_t vecTransfer;
    if (config.areMutexUnits("vec_mte2", "mte3"))
      vecTransfer = hwUnitCycles[HWUnit::VecMTE2] + hwUnitCycles[HWUnit::MTE3];
    else
      vecTransfer = std::max(hwUnitCycles[HWUnit::VecMTE2], hwUnitCycles[HWUnit::MTE3]);
    int64_t vectorPathCycles = std::max(hwUnitCycles[HWUnit::Vector], vecTransfer);

    // Total: max of paths (Cube and Vector paths overlap). If tile mix
    // parameters are present, refine this optimistic roofline with a TTIR-only
    // approximation of the TileCubeVectorLoop implementation: loop tiling
    // changes C/V handoff granularity and workspace/layout pressure, but the
    // model never assumes HIVM-only CopyOut/subview details are directly
    // visible in TTIR.
    int64_t baseRooflineTotalCycles = std::max(cubePathCycles, vectorPathCycles);
    int64_t cubeTransferCycles =
        hwUnitCycles[HWUnit::CubeMTE2] + hwUnitCycles[HWUnit::FixPipe];
    int64_t vectorTransferCycles =
        hwUnitCycles[HWUnit::VecMTE2] + hwUnitCycles[HWUnit::MTE3];
    TileMixBalanceStats tileMixStats = estimateTileMixLoopTilingBalance(
        tileMixParams, scheduler, config, cubePathCycles, vectorPathCycles,
        cubeTransferCycles, vectorTransferCycles, baseRooflineTotalCycles);
    int64_t rooflineTotalCycles = tileMixStats.valid
                                      ? tileMixStats.adjustedCycles
                                      : baseRooflineTotalCycles;
    
    // Also calculate simple sum for comparison
    int64_t simpleSumCycles = 0;
    for (const auto &pipelineOp : scheduler.getAllOps())
      simpleSumCycles += pipelineOp.duration * pipelineOp.loopMultiplier;
    
    module->setAttr("ascend.scheduled_cycles_one_iter",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), oneIterCycles));
    module->setAttr("ascend.roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), rooflineTotalCycles));
    module->setAttr("ascend.base_roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), baseRooflineTotalCycles));
    if (tileMixStats.used) {
      module->setAttr("ascend.tile_mix_schedule_model",
                      StringAttr::get(module.getContext(), "ttir_tilemix_working_set_loop"));
      module->setAttr("ascend.tile_mix_model_valid",
                      BoolAttr::get(module.getContext(), tileMixStats.valid));
      module->setAttr("ascend.tile_mix_adjusted_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.adjustedCycles));
      module->setAttr("ascend.tile_mix_net_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.netDeltaCycles));
      module->setAttr("ascend.tile_mix_boundary_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.boundaryCycles));
      module->setAttr("ascend.tile_mix_balance_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.balancePenaltyCycles));
      module->setAttr("ascend.tile_mix_handoff_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffReliefCycles));
      module->setAttr("ascend.tile_mix_workspace_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.workspaceReliefCycles));
      module->setAttr("ascend.tile_mix_buffer_fit_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.bufferFitPenaltyCycles));
      module->setAttr("ascend.tile_mix_sync_frequency_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.syncFrequencyPenaltyCycles));
      module->setAttr("ascend.tile_mix_cube_segments",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeSegmentCount));
      module->setAttr("ascend.tile_mix_vector_segments",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorSegmentCount));
      module->setAttr("ascend.tile_mix_cube_loop_trip",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeLoopTrip));
      module->setAttr("ascend.tile_mix_vector_loop_trip",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorLoopTrip));
      module->setAttr("ascend.tile_mix_cube_layout_ops",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeLayoutOpCount));
      module->setAttr("ascend.tile_mix_vector_layout_ops",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorLayoutOpCount));
      module->setAttr("ascend.tile_mix_cube_workspace_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeWorkspaceBytes));
      module->setAttr("ascend.tile_mix_vector_workspace_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorWorkspaceBytes));
      module->setAttr("ascend.tile_mix_cube_subtile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeSubtileBytes));
      module->setAttr("ascend.tile_mix_vector_subtile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorSubtileBytes));
      module->setAttr("ascend.tile_mix_cube_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.cubeTargetBytes));
      module->setAttr("ascend.tile_mix_vector_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.vectorTargetBytes));
      if (tileMixStats.inferredTileM > 0) {
        module->setAttr("ascend.tile_mix_block_m",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileM));
        module->setAttr("ascend.tile_mix_inferred_block_m",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileM));
      }
      if (tileMixStats.inferredTileN > 0) {
        module->setAttr("ascend.tile_mix_block_n",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileN));
        module->setAttr("ascend.tile_mix_inferred_block_n",
                        IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.inferredTileN));
      }
      module->setAttr("ascend.tile_mix_tile_shape_source",
                      StringAttr::get(module.getContext(), tileMixStats.tileShapeSource));
      module->setAttr("ascend.tile_mix_dtype_source",
                      StringAttr::get(module.getContext(), tileMixStats.dtypeSource));
      module->setAttr("ascend.tile_mix_handoff_source",
                      StringAttr::get(module.getContext(), tileMixStats.handoffSource));
      module->setAttr("ascend.tile_mix_intermediate_source",
                      StringAttr::get(module.getContext(), tileMixStats.intermediateSource));
      module->setAttr("ascend.tile_mix_handoff_feature_dim",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffFeatureDim));
      module->setAttr("ascend.tile_mix_handoff_dtype_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffDtypeBytes));
      module->setAttr("ascend.tile_mix_handoff_tile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffTileBytes));
      module->setAttr("ascend.tile_mix_handoff_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffTargetBytes));
      module->setAttr("ascend.tile_mix_handoff_neutral_block_n",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffNeutralBlockN));
      module->setAttr("ascend.tile_mix_intermediate_tile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediateTileBytes));
      module->setAttr("ascend.tile_mix_intermediate_target_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediateTargetBytes));
      module->setAttr("ascend.tile_mix_intermediate_neutral_block_m",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediateNeutralBlockM));
      module->setAttr("ascend.tile_mix_intermediate_pressure_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.intermediatePressurePenaltyCycles));
      module->setAttr("ascend.tile_mix_loop_granularity_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.loopGranularityReliefCycles));
      module->setAttr("ascend.tile_mix_loop_mismatch_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.loopMismatchPenaltyCycles));
    }
    if (tileMixParams.vectorLoop > 0) {
      module->setAttr("ascend.tile_mix_vector_loop",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixParams.vectorLoop));
    }
    if (tileMixParams.cubeLoop > 0) {
      module->setAttr("ascend.tile_mix_cube_loop",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixParams.cubeLoop));
    }
    module->setAttr("ascend.scheduled_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), rooflineTotalCycles));
    module->setAttr("ascend.simple_sum_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), simpleSumCycles));
    
    // Print results
    llvm::outs() << "\n=== Pipeline Analysis (" << config.getName() << ") ===\n";
    llvm::outs() << "One iteration cycles (scheduled): " << oneIterCycles << "\n";
    llvm::outs() << "\nPer-HW-Unit cycles (with loops):\n";
    llvm::outs() << "  Cube path:\n";
    llvm::outs() << "    Cube compute: " << hwUnitCycles[HWUnit::Cube] << "\n";
    llvm::outs() << "    CubeMTE2 (load): " << hwUnitCycles[HWUnit::CubeMTE2] << "\n";
    llvm::outs() << "    FixPipe (store): " << hwUnitCycles[HWUnit::FixPipe] << "\n";
    llvm::outs() << "    Path total (max): " << cubePathCycles << "\n";
    llvm::outs() << "  Vector path:\n";
    llvm::outs() << "    Vector compute: " << hwUnitCycles[HWUnit::Vector] << "\n";
    llvm::outs() << "    VecMTE2 (load): " << hwUnitCycles[HWUnit::VecMTE2] << "\n";
    llvm::outs() << "    MTE3 (store): " << hwUnitCycles[HWUnit::MTE3] << "\n";
    llvm::outs() << "    Path total (max): " << vectorPathCycles << "\n";
    llvm::outs() << "\nTotal cycles:\n";
    llvm::outs() << "  Simple sum (no overlap): " << simpleSumCycles 
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(simpleSumCycles)) << " us)\n";
    llvm::outs() << "  Roofline base (with overlap): " << baseRooflineTotalCycles
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(baseRooflineTotalCycles)) << " us)\n";
    if (tileMixStats.used) {
      llvm::outs() << "  Tile mix params: vector_loop=" << tileMixParams.vectorLoop
                   << ", cube_loop=" << tileMixParams.cubeLoop
                   << "\n";
      llvm::outs() << "  Tile mix inferred features: block_m="
                   << tileMixStats.inferredTileM
                   << ", block_n=" << tileMixStats.inferredTileN
                   << ", tile_shape_source="
                   << tileMixStats.tileShapeSource
                   << ", dtype_source=" << tileMixStats.dtypeSource
                   << ", handoff_source=" << tileMixStats.handoffSource
                   << ", intermediate_source="
                   << tileMixStats.intermediateSource << "\n";
      llvm::outs() << "  Tile mix TTIR buffer fit: "
                   << (tileMixStats.valid ? "valid" : "fallback")
                   << ", cube_segments=" << tileMixStats.cubeSegmentCount
                   << ", vector_segments=" << tileMixStats.vectorSegmentCount
                   << ", cube_loop_trip=" << tileMixStats.cubeLoopTrip
                   << ", vector_loop_trip=" << tileMixStats.vectorLoopTrip
                   << "\n";
      llvm::outs() << "  Tile mix layout proxies: cube_layout_ops="
                   << tileMixStats.cubeLayoutOpCount
                   << ", vector_layout_ops="
                   << tileMixStats.vectorLayoutOpCount << "\n";
      llvm::outs() << "  Tile mix buffer fit: cube_workspace_bytes="
                   << tileMixStats.cubeWorkspaceBytes
                   << ", vector_workspace_bytes="
                   << tileMixStats.vectorWorkspaceBytes
                   << ", cube_subtile_bytes="
                   << tileMixStats.cubeSubtileBytes
                   << ", vector_subtile_bytes="
                   << tileMixStats.vectorSubtileBytes
                   << ", cube_target_bytes="
                   << tileMixStats.cubeTargetBytes
                   << ", vector_target_bytes="
                   << tileMixStats.vectorTargetBytes << "\n";
      llvm::outs() << "  Tile mix handoff footprint: feature_dim="
                   << tileMixStats.handoffFeatureDim
                   << ", dtype_bytes="
                   << tileMixStats.handoffDtypeBytes
                   << ", tile_bytes="
                   << tileMixStats.handoffTileBytes
                   << ", target_bytes="
                   << tileMixStats.handoffTargetBytes
                   << ", neutral_block_n="
                   << tileMixStats.handoffNeutralBlockN << "\n";
      llvm::outs() << "  Tile mix intermediate footprint: tile_bytes="
                   << tileMixStats.intermediateTileBytes
                   << ", target_bytes="
                   << tileMixStats.intermediateTargetBytes
                   << ", neutral_block_m="
                   << tileMixStats.intermediateNeutralBlockM << "\n";
      llvm::outs() << "  Tile mix delta cycles: boundary="
                   << tileMixStats.boundaryCycles
                   << ", balance_penalty="
                   << tileMixStats.balancePenaltyCycles
                   << ", handoff_relief="
                   << tileMixStats.handoffReliefCycles
                   << ", loop_granularity_relief="
                   << tileMixStats.loopGranularityReliefCycles
                   << ", workspace_relief="
                   << tileMixStats.workspaceReliefCycles
                   << ", buffer_fit_penalty="
                   << tileMixStats.bufferFitPenaltyCycles
                   << ", intermediate_pressure_penalty="
                   << tileMixStats.intermediatePressurePenaltyCycles
                   << ", loop_mismatch_penalty="
                   << tileMixStats.loopMismatchPenaltyCycles
                    << ", sync_frequency_penalty="
                    << tileMixStats.syncFrequencyPenaltyCycles
                    << ", net_delta=" << tileMixStats.netDeltaCycles << "\n";
    }
    llvm::outs() << "  Roofline model (TTIR buffer-fit tile mix): " << rooflineTotalCycles
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(rooflineTotalCycles)) << " us)\n";
    llvm::outs() << "  Speedup from overlap: " << llvm::format("%.2fx", 
                    static_cast<double>(simpleSumCycles) / std::max(rooflineTotalCycles, 1L)) << "\n";
    
    llvm::outs() << "\n=== Pipeline Timeline (one iteration) ===\n";
    scheduler.printTimeline(llvm::outs());
    llvm::outs() << "\n=== Utilization Report ===\n";
    scheduler.printUtilizationReport(llvm::outs());
    
    // Generate Perfetto trace only when an explicit path is provided.
    // Replaces the old hard-coded "pipeline_trace.json" cwd write.
    if (!perfettoTraceFile.empty()) {
      generatePerfettoTrace(scheduler, perfettoTraceFile,
                            oneIterCycles, rooflineTotalCycles);
    }

    // Emit dependency graph JSON for downstream performance bound model consumers
    // (perfbound/model/serialization.py mandatory/avoidable split)
    // Only writes when an explicit output path is provided — never pollutes cwd.
    if (!dependencyGraphFile.empty()) {
      std::error_code depEC;
      llvm::raw_fd_ostream depFile(dependencyGraphFile, depEC,
                                    llvm::sys::fs::OF_Text);
      if (!depEC) {
        scheduler.emitDependencyGraphJSON(depFile);
        llvm::outs() << "Dependency graph: " << dependencyGraphFile << "\n";
      } else {
        llvm::errs() << "Warning: could not write " << dependencyGraphFile
                     << ": " << depEC.message() << "\n";
      }
    }
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
