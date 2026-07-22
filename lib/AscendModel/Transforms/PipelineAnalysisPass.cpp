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
#include "AscendModel/Analysis/KernelLaunchUtils.h"
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
#include <map>
#include <set>
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
  bool summaryPresent = false;
  bool summaryValid = false;
  bool vectorApplied = false;
  bool cubeApplied = false;
  int64_t vectorSegments = 1;
  int64_t cubeSegments = 1;
  int64_t syncOpsBefore = 0;
  int64_t syncOpsAfter = 0;
  std::string summarySource = "missing";
  std::string vectorSkipReason = "missing_pass_summary";
  std::string cubeSkipReason = "missing_pass_summary";
};

struct WorkspaceMultibufferParams {
  bool present = false;
  int64_t requestedSlots = 1;
};

struct TileMixModelConfig {
  int64_t loopControlCyclesPerSegment = 2;
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

struct TileMixStats {
  bool used = false;
  bool valid = false;
  bool adjustmentApplied = false;
  bool cubeApplied = false;
  bool vectorApplied = false;
  int64_t confidencePercent = 0;
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
  int64_t handoffSubtileBytes = 0;
  int64_t handoffSegmentCount = 0;
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
  int64_t gmDeltaCycles = 0;
  int64_t externalSyncDeltaCycles = 0;
  int64_t bufferDeltaCycles = 0;
  int64_t pipelineDeltaCycles = 0;
  int64_t scalarControlDeltaCycles = 0;
  int64_t syncOpsBefore = 0;
  int64_t syncOpsAfter = 0;
  std::string summarySource = "missing";
  std::string cubeSkipReason = "missing_pass_summary";
  std::string vectorSkipReason = "missing_pass_summary";
  std::string tileShapeSource = "none";
  std::string dtypeSource = "none";
  std::string handoffSource = "none";
  std::string intermediateSource = "none";
};

struct WorkspaceMultibufferStats {
  bool used = false;
  bool valid = false;
  bool adjustmentApplied = false;
  int64_t requestedSlots = 1;
  int64_t referenceSlots = 1;
  int64_t slotDelta = 0;
  int64_t extraSlots = 0;
  int64_t workspaceFamilyCount = 0;
  int64_t cubeToVectorFamilyCount = 0;
  int64_t vectorToCubeFamilyCount = 0;
  int64_t workspaceBytesPerSlot = 0;
  int64_t iterationCount = 0;
  int64_t cubeToVectorIterations = 0;
  int64_t vectorToCubeIterations = 0;
  int64_t cubeProducerTailCycles = 0;
  int64_t vectorProducerTailCycles = 0;
  int64_t syncPairCycles = 0;
  int64_t syncDeltaCycles = 0;
  int64_t blockingCycles = 0;
  int64_t referenceBlockingCycles = 0;
  int64_t producerWaitReliefCycles = 0;
  int64_t referenceQueuePenaltyCycles = 0;
  int64_t overlapReliefCycles = 0;
  int64_t queueDeltaCycles = 0;
  int64_t netDeltaCycles = 0;
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

bool parseNonNegativeInt(llvm::StringRef value, int64_t &result) {
  value = value.trim();
  int64_t parsed = 0;
  if (value.empty() || value.getAsInteger(10, parsed) || parsed < 0)
    return false;
  result = parsed;
  return true;
}

bool parseBool01(llvm::StringRef value, bool &result) {
  int64_t parsed = 0;
  if (!parseNonNegativeInt(value, parsed) || parsed > 1)
    return false;
  result = parsed == 1;
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
    } else if (key == "tile_mix_summary_source" && !value.empty()) {
      params.summarySource = value.str();
      params.summaryPresent = true;
    } else if (key == "tile_mix_summary_valid") {
      params.summaryPresent = true;
      parseBool01(value, params.summaryValid);
    } else if (key == "tile_mix_vector_applied") {
      parseBool01(value, params.vectorApplied);
    } else if (key == "tile_mix_cube_applied") {
      parseBool01(value, params.cubeApplied);
    } else if (key == "tile_mix_vector_segments" &&
               parsePositiveInt(value, parsed)) {
      params.vectorSegments = parsed;
    } else if (key == "tile_mix_cube_segments" &&
               parsePositiveInt(value, parsed)) {
      params.cubeSegments = parsed;
    } else if (key == "tile_mix_sync_ops_before" &&
               parseNonNegativeInt(value, parsed)) {
      params.syncOpsBefore = parsed;
    } else if (key == "tile_mix_sync_ops_after" &&
               parseNonNegativeInt(value, parsed)) {
      params.syncOpsAfter = parsed;
    } else if (key == "tile_mix_vector_skip_reason" && !value.empty()) {
      params.vectorSkipReason = value.str();
    } else if (key == "tile_mix_cube_skip_reason" && !value.empty()) {
      params.cubeSkipReason = value.str();
    }
  }
  return params;
}

WorkspaceMultibufferParams
parseWorkspaceMultibufferParams(llvm::StringRef compileParamsStr) {
  WorkspaceMultibufferParams params;
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
    if (key == "set_workspace_multibuffer" &&
        parsePositiveInt(value, parsed)) {
      params.requestedSlots = parsed;
      params.present = true;
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

bool isWorkspaceMultibufferModelEnabled() {
  const char *mode =
      std::getenv("ASCEND_COSTMODEL_WORKSPACE_MULTIBUFFER_MODEL");
  // Preserve existing Off/On report behavior when the new independent switch
  // is not set. Callers can explicitly control the peer feature with the new
  // variable.
  if (!mode)
    return isTileMixDagModelEnabled();
  llvm::StringRef value(mode);
  return !(value.equals_insensitive("0") || value.equals_insensitive("off") ||
           value.equals_insensitive("false") || value.equals_insensitive("none"));
}

int64_t ceilDiv(int64_t value, int64_t divisor) {
  if (value <= 0 || divisor <= 0)
    return 0;
  return (value + divisor - 1) / divisor;
}

int64_t overflowTransferCycles(int64_t chunkBytes, int64_t targetBytes,
                               int64_t segmentCount, int64_t totalBytes,
                               int64_t transferCycles) {
  if (chunkBytes <= 0 || targetBytes <= 0 || segmentCount <= 0 ||
      totalBytes <= 0 || transferCycles <= 0)
    return 0;
  int64_t overflowPerSegment = std::max<int64_t>(0, chunkBytes - targetBytes);
  if (overflowPerSegment == 0)
    return 0;
  long double overflowBytes = static_cast<long double>(overflowPerSegment) *
                              static_cast<long double>(segmentCount);
  long double cycles = overflowBytes * static_cast<long double>(transferCycles) /
                       static_cast<long double>(totalBytes);
  return std::max<int64_t>(0, static_cast<int64_t>(std::ceil(cycles)));
}

TileMixModelConfig getTileMixModelConfig(const HardwareConfig &config) {
  TileMixModelConfig model;
  model.loopControlCyclesPerSegment = std::max<int64_t>(
      0, config.getCostModelIntParam("tilemix_loop_control_cycles_per_segment",
                                     model.loopControlCyclesPerSegment));
  return model;
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

int64_t tileMixExternalSyncDeltaCycles(const TileMixParams &params,
                                       const HardwareConfig &config) {
  // The pass summary counts concrete sync operations before and after the
  // transformation. Convert that marginal count to cycles; do not scale the
  // whole roofline by an empirical percentage.
  int64_t deltaOps = params.syncOpsAfter - params.syncOpsBefore;
  int64_t cyclesPerOp = std::max<int64_t>(
      0, (config.getSyncOpCycles("set_flag", 1) +
          config.getSyncOpCycles("wait_flag", 2) + 1) /
             2);
  return deltaOps * cyclesPerOp;
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

struct TileMixSideDecision {
  bool known = false;
  bool applied = false;
  int64_t segments = 1;
  std::string skipReason = "unknown";
};

TileMixSideDecision inferTileMixSideFromTTIR(
    TileMixPath side, int64_t requestedSegments, int64_t pathCycles,
    int64_t layoutOps, int64_t workspaceBytes,
    int64_t localBufferBytes, int64_t vectorAlignmentBytes) {
  TileMixSideDecision decision;
  if (requestedSegments <= 1) {
    decision.known = true;
    decision.skipReason = "requested_loop_le_one";
    return decision;
  }
  if (pathCycles <= 0) {
    decision.known = true;
    decision.skipReason = side == TileMixPath::Cube ? "no_cube_path"
                                                    : "no_vector_path";
    return decision;
  }
  if (layoutOps <= 0) {
    decision.known = true;
    decision.skipReason = side == TileMixPath::Cube ? "no_cube_copyout"
                                                    : "no_vector_copyout";
    return decision;
  }
  if (workspaceBytes <= 0 || localBufferBytes <= 0) {
    decision.skipReason = side == TileMixPath::Cube
                              ? "unknown_cube_copyout_bytes"
                              : "unknown_vector_copyout_bytes";
    return decision;
  }

  // tile-mix-*-loop is the target trip count of a new inner sub-tile loop.
  // It splits one existing Cube/Vector loop iteration; it is not capped by
  // the source TTIR loop trip count. In particular, a source loop with trip
  // count one can still be split into target trip count 2/4 when CopyOut and
  // buffer evidence satisfy the pass constraints.
  decision.segments = requestedSegments;

  if (side == TileMixPath::Cube && workspaceBytes < localBufferBytes) {
    // principle.docx: Cube does not tile when the pre-tile chunk already fits
    // the total L0C capacity.
    decision.known = true;
    decision.segments = 1;
    decision.skipReason = "cube_tile_fits_l0c";
    return decision;
  }
  if (side == TileMixPath::Vector &&
      ceilDiv(workspaceBytes, decision.segments) < vectorAlignmentBytes) {
    // principle.docx: Vector does not tile when its post-tile chunk is below
    // one UB/vector alignment quantum.
    decision.known = true;
    decision.segments = 1;
    decision.skipReason = "vector_subtile_below_alignment";
    return decision;
  }

  decision.known = true;
  decision.applied = true;
  decision.skipReason = "none";
  return decision;
}

struct WorkspaceMultibufferEvidence {
  int64_t familyCount = 0;
  int64_t bytesPerSlot = 0;
  int64_t iterationCount = 0;
  int64_t cubeToVectorFamilyCount = 0;
  int64_t vectorToCubeFamilyCount = 0;
  int64_t cubeToVectorIterations = 0;
  int64_t vectorToCubeIterations = 0;
  int64_t cubeProducerTailCycles = 0;
  int64_t vectorProducerTailCycles = 0;
};

WorkspaceMultibufferEvidence inferWorkspaceMultibufferEvidence(
    const PipelineScheduler &scheduler) {
  std::map<int64_t, int64_t> cubeLoadIterations;
  std::map<int64_t, int64_t> cubeStoreIterations;
  std::map<int64_t, int64_t> vectorLoadIterations;
  std::map<int64_t, int64_t> vectorStoreIterations;
  std::map<int64_t, int64_t> cubeStoreEndCycles;
  std::map<int64_t, int64_t> vectorStoreEndCycles;
  int64_t cubePathEndCycle = 0;
  int64_t vectorPathEndCycle = 0;

  auto recordTransfer = [](std::map<int64_t, int64_t> &transfers,
                           int64_t bytes, int64_t iterations) {
    if (bytes <= 0)
      return;
    transfers[bytes] =
        std::max(transfers[bytes], std::max<int64_t>(iterations, 1));
  };

  for (const auto &op : scheduler.getAllOps()) {
    TileMixPath path = getTileMixPath(op.hwUnit);
    if (path == TileMixPath::Cube)
      cubePathEndCycle = std::max(cubePathEndCycle, op.endCycle);
    else if (path == TileMixPath::Vector)
      vectorPathEndCycle = std::max(vectorPathEndCycle, op.endCycle);
    if (!op.mlirOp)
      continue;
    if (auto loadOp = dyn_cast<CubeLoadOp>(op.mlirOp)) {
      recordTransfer(cubeLoadIterations, loadOp.getTransferBytes(),
                     op.loopMultiplier);
    } else if (auto storeOp = dyn_cast<CubeStoreOp>(op.mlirOp)) {
      recordTransfer(cubeStoreIterations, storeOp.getTransferBytes(),
                     op.loopMultiplier);
      cubeStoreEndCycles[storeOp.getTransferBytes()] = std::max(
          cubeStoreEndCycles[storeOp.getTransferBytes()], op.endCycle);
    } else if (auto loadOp = dyn_cast<VectorLoadOp>(op.mlirOp)) {
      recordTransfer(vectorLoadIterations, loadOp.getTransferBytes(),
                     op.loopMultiplier);
    } else if (auto storeOp = dyn_cast<VectorStoreOp>(op.mlirOp)) {
      recordTransfer(vectorStoreIterations, storeOp.getTransferBytes(),
                     op.loopMultiplier);
      vectorStoreEndCycles[storeOp.getTransferBytes()] = std::max(
          vectorStoreEndCycles[storeOp.getTransferBytes()], op.endCycle);
    }
  }

  WorkspaceMultibufferEvidence evidence;
  auto addMatchedFamilies = [&](const std::map<int64_t, int64_t> &producers,
                                const std::map<int64_t, int64_t> &consumers,
                                const std::map<int64_t, int64_t> &producerEnds,
                                int64_t producerPathEnd,
                                int64_t &directionFamilyCount,
                                int64_t &directionIterations,
                                int64_t &producerTailCycles) {
    for (const auto &[bytes, producerIterations] : producers) {
      auto consumer = consumers.find(bytes);
      if (consumer == consumers.end())
        continue;
      ++evidence.familyCount;
      ++directionFamilyCount;
      evidence.bytesPerSlot += bytes;
      int64_t commonIterations =
          std::min(producerIterations, consumer->second);
      if (evidence.iterationCount == 0)
        evidence.iterationCount = commonIterations;
      else
        evidence.iterationCount =
            std::min(evidence.iterationCount, commonIterations);
      if (directionIterations == 0)
        directionIterations = commonIterations;
      else
        directionIterations = std::min(directionIterations, commonIterations);
      auto producerEnd = producerEnds.find(bytes);
      int64_t tail = producerEnd == producerEnds.end()
          ? 0
          : std::max<int64_t>(0, producerPathEnd - producerEnd->second);
      if (directionFamilyCount == 1)
        producerTailCycles = tail;
      else
        producerTailCycles = std::min(producerTailCycles, tail);
    }
  };
  // Direction is part of the family identity: equally sized Cube->Vector and
  // Vector->Cube workspaces have independent version/synchronization state.
  addMatchedFamilies(
      cubeStoreIterations, vectorLoadIterations, cubeStoreEndCycles,
      cubePathEndCycle, evidence.cubeToVectorFamilyCount,
      evidence.cubeToVectorIterations, evidence.cubeProducerTailCycles);
  addMatchedFamilies(
      vectorStoreIterations, cubeLoadIterations, vectorStoreEndCycles,
      vectorPathEndCycle, evidence.vectorToCubeFamilyCount,
      evidence.vectorToCubeIterations, evidence.vectorProducerTailCycles);
  return evidence;
}

struct BoundedBufferSchedule {
  int64_t makespanCycles = 0;
  int64_t producerBlockingCycles = 0;
};

BoundedBufferSchedule scheduleFiniteWorkspaceBuffer(
    int64_t producerPathCycles, int64_t consumerPathCycles,
    int64_t iterationCount, int64_t bufferSlots,
    int64_t producerTailCycles = 0) {
  BoundedBufferSchedule result;
  if (producerPathCycles <= 0 || consumerPathCycles <= 0 ||
      iterationCount <= 0 || bufferSlots <= 0)
    return result;

  // Distribute the integer path cycles over concrete TTIR iterations. This is
  // an exact deterministic finite FIFO schedule for the static information
  // available at TTIR: producer i cannot reuse slot i%B before consumer i-B
  // releases it, and consumer i cannot start before producer i completes.
  auto segmentCycles = [](int64_t total, int64_t count, int64_t index) {
    int64_t base = total / count;
    int64_t remainder = total % count;
    return std::max<int64_t>(1, base + (index < remainder ? 1 : 0));
  };

  producerTailCycles = std::max<int64_t>(
      0, std::min(producerTailCycles, producerPathCycles - 1));
  int64_t producerHandoffCycles = producerPathCycles - producerTailCycles;
  std::vector<int64_t> consumerDone(iterationCount, 0);
  int64_t producerCursor = 0;
  int64_t consumerCursor = 0;
  for (int64_t i = 0; i < iterationCount; ++i) {
    int64_t slotRelease =
        i >= bufferSlots ? consumerDone[i - bufferSlots] : 0;
    int64_t producerStart = std::max(producerCursor, slotRelease);
    result.producerBlockingCycles += producerStart - producerCursor;
    int64_t producerDone =
        producerStart + segmentCycles(producerHandoffCycles, iterationCount, i);
    producerCursor = producerDone;

    int64_t consumerStart = std::max(consumerCursor, producerDone);
    consumerDone[i] =
        consumerStart + segmentCycles(consumerPathCycles, iterationCount, i);
    consumerCursor = consumerDone[i];
  }
  result.makespanCycles =
      std::max(producerCursor + producerTailCycles, consumerCursor);
  return result;
}

WorkspaceMultibufferStats estimateWorkspaceMultibuffer(
    const WorkspaceMultibufferParams &params,
    const PipelineScheduler &scheduler, const HardwareConfig &config,
    int64_t cubePathCycles, int64_t vectorPathCycles) {
  WorkspaceMultibufferStats stats;
  if (!params.present || !isWorkspaceMultibufferModelEnabled())
    return stats;

  stats.used = true;
  stats.requestedSlots = std::max<int64_t>(1, params.requestedSlots);
  WorkspaceMultibufferEvidence evidence =
      inferWorkspaceMultibufferEvidence(scheduler);
  stats.workspaceFamilyCount = evidence.familyCount;
  stats.cubeToVectorFamilyCount = evidence.cubeToVectorFamilyCount;
  stats.vectorToCubeFamilyCount = evidence.vectorToCubeFamilyCount;
  stats.workspaceBytesPerSlot = evidence.bytesPerSlot;
  stats.iterationCount = evidence.iterationCount;
  stats.cubeToVectorIterations = evidence.cubeToVectorIterations;
  stats.vectorToCubeIterations = evidence.vectorToCubeIterations;
  stats.cubeProducerTailCycles = evidence.cubeProducerTailCycles;
  stats.vectorProducerTailCycles = evidence.vectorProducerTailCycles;

  // The roofline base already assumes simultaneous producer/consumer paths.
  // A proven cross-path handoff therefore has a two-slot ping-pong reference
  // depth. With no proven family, one slot is the neutral reference.
  stats.referenceSlots = stats.workspaceFamilyCount > 0 ? 2 : 1;
  stats.slotDelta = stats.requestedSlots - stats.referenceSlots;
  stats.extraSlots = std::max<int64_t>(0, stats.slotDelta);
  stats.syncPairCycles = config.getSyncOpCycles("set_flag", 1) +
                         config.getSyncOpCycles("wait_flag", 2);
  if (stats.workspaceFamilyCount > 0) {
    stats.syncDeltaCycles = stats.extraSlots * stats.workspaceFamilyCount *
                            stats.syncPairCycles;

    auto accumulateDirection = [&](int64_t familyCount, int64_t iterations,
                                   int64_t producerCycles,
                                   int64_t consumerCycles,
                                   int64_t producerTailCycles) {
      if (familyCount <= 0 || iterations <= 0)
        return;
      BoundedBufferSchedule requested = scheduleFiniteWorkspaceBuffer(
          producerCycles, consumerCycles, iterations, stats.requestedSlots,
          producerTailCycles);
      BoundedBufferSchedule reference = scheduleFiniteWorkspaceBuffer(
          producerCycles, consumerCycles, iterations, stats.referenceSlots,
          producerTailCycles);
      BoundedBufferSchedule ideal = scheduleFiniteWorkspaceBuffer(
          producerCycles, consumerCycles, iterations, iterations,
          producerTailCycles);
      stats.blockingCycles =
          std::max(stats.blockingCycles, requested.producerBlockingCycles);
      stats.referenceBlockingCycles = std::max(
          stats.referenceBlockingCycles, reference.producerBlockingCycles);
      stats.queueDeltaCycles = std::max(
          stats.queueDeltaCycles,
          std::max<int64_t>(0, requested.makespanCycles - ideal.makespanCycles));
      stats.referenceQueuePenaltyCycles = std::max(
          stats.referenceQueuePenaltyCycles,
          std::max<int64_t>(0, reference.makespanCycles - ideal.makespanCycles));
    };
    accumulateDirection(stats.cubeToVectorFamilyCount,
                        stats.cubeToVectorIterations, cubePathCycles,
                        vectorPathCycles, stats.cubeProducerTailCycles);
    accumulateDirection(stats.vectorToCubeFamilyCount,
                        stats.vectorToCubeIterations, vectorPathCycles,
                        cubePathCycles, stats.vectorProducerTailCycles);
    stats.overlapReliefCycles = std::max<int64_t>(
        0, stats.referenceQueuePenaltyCycles - stats.queueDeltaCycles);
    stats.producerWaitReliefCycles = std::max<int64_t>(
        0, stats.referenceBlockingCycles - stats.blockingCycles);
  }

  stats.netDeltaCycles = stats.syncDeltaCycles + stats.queueDeltaCycles;
  stats.valid = true;
  stats.adjustmentApplied = stats.netDeltaCycles != 0;
  return stats;
}

TileMixStats estimateTileMix(
    const TileMixParams &params, const PipelineScheduler &scheduler,
    const HardwareConfig &config, int64_t cubePathCycles,
    int64_t vectorPathCycles, int64_t cubeTransferCycles,
    int64_t vectorTransferCycles, int64_t baseCycles) {
  TileMixStats stats;
  stats.baseCycles = baseCycles;
  stats.adjustedCycles = baseCycles;
  if (!params.hasAny || !isTileMixDagModelEnabled())
    return stats;

  stats.used = true;
  stats.summarySource = params.summarySource;
  stats.cubeSkipReason = params.cubeSkipReason;
  stats.vectorSkipReason = params.vectorSkipReason;
  stats.syncOpsBefore = params.syncOpsBefore;
  stats.syncOpsAfter = params.syncOpsAfter;

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

  int64_t l0cBytes = static_cast<int64_t>(config.getMemorySizeBytes("l0c"));
  int64_t ubBytes = static_cast<int64_t>(config.getMemorySizeBytes("ub"));
  if (l0cBytes <= 0)
    l0cBytes = 256 * 1024;
  if (ubBytes <= 0)
    ubBytes = 256 * 1024;
  TileMixModelConfig model = getTileMixModelConfig(config);
  TileMixDerivedFeatures features = inferTileMixDerivedFeatures(scheduler);
  stats.inferredTileM = features.tileM;
  stats.inferredTileN = features.tileN;
  stats.tileShapeSource = features.tileShapeSource;
  stats.dtypeSource = features.dtypeSource;
  stats.handoffSource = features.handoffSource;
  stats.intermediateSource = features.intermediateSource;
  stats.handoffFeatureDim = features.handoffFeatureDim;
  stats.handoffDtypeBytes = features.dtypeBytes;
  stats.handoffTileBytes = features.handoffTileBytes;
  stats.intermediateTileBytes = features.intermediateTileBytes;

  int64_t vectorAlignment = std::max<int64_t>(1, config.getVectorWidthBytes());
  TileMixSideDecision cubeDecision;
  TileMixSideDecision vectorDecision;
  bool usingPassSummary = params.summaryPresent && params.summaryValid;
  if (usingPassSummary) {
    cubeDecision.known = true;
    cubeDecision.applied = params.cubeApplied;
    cubeDecision.segments = params.cubeApplied ? params.cubeSegments : 1;
    cubeDecision.skipReason = params.cubeSkipReason;
    vectorDecision.known = true;
    vectorDecision.applied = params.vectorApplied;
    vectorDecision.segments = params.vectorApplied ? params.vectorSegments : 1;
    vectorDecision.skipReason = params.vectorSkipReason;
    stats.summarySource = params.summarySource;
    stats.confidencePercent = 100;
  } else {
    // P0-1 primary path: infer only the eligibility conditions explicitly
    // stated by principle.docx from TTIR/AscendModel evidence. The optional
    // HIVM pass summary is a higher-confidence override, not a prerequisite.
    cubeDecision = inferTileMixSideFromTTIR(
        TileMixPath::Cube, params.cubeLoop, cubePathCycles,
        stats.cubeLayoutOpCount, stats.cubeWorkspaceBytes, l0cBytes,
        vectorAlignment);
    vectorDecision = inferTileMixSideFromTTIR(
        TileMixPath::Vector, params.vectorLoop, vectorPathCycles,
        stats.vectorLayoutOpCount, stats.vectorWorkspaceBytes, ubBytes,
        vectorAlignment);
    stats.summarySource = params.summaryPresent
        ? "ttir_principle_v2_target_trip_after_invalid_summary"
        : "ttir_principle_v2_target_trip";
    bool hasUnknownRequestedSide =
        (params.cubeLoop > 1 && !cubeDecision.known) ||
        (params.vectorLoop > 1 && !vectorDecision.known);
    stats.confidencePercent = hasUnknownRequestedSide ? 50 : 70;
  }

  // P0-3 validates each side independently. Unknown/contradictory evidence on
  // one side disables only that side; no global heuristic benefit is invented.
  auto validatePassDecision = [](TileMixSideDecision &decision,
                                 int64_t pathCycles, int64_t layoutOps,
                                 int64_t workspaceBytes,
                                 llvm::StringRef side) {
    if (!decision.applied)
      return;
    if (decision.segments <= 1 || pathCycles <= 0 || layoutOps <= 0 ||
        workspaceBytes <= 0) {
      decision.known = false;
      decision.applied = false;
      decision.segments = 1;
      decision.skipReason = "model_missing_" + side.str() + "_evidence";
    }
  };
  validatePassDecision(cubeDecision, cubePathCycles, stats.cubeLayoutOpCount,
                       stats.cubeWorkspaceBytes, "cube");
  validatePassDecision(vectorDecision, vectorPathCycles,
                       stats.vectorLayoutOpCount, stats.vectorWorkspaceBytes,
                       "vector");

  stats.cubeApplied = cubeDecision.applied;
  stats.vectorApplied = vectorDecision.applied;
  stats.cubeSkipReason = cubeDecision.skipReason;
  stats.vectorSkipReason = vectorDecision.skipReason;
  stats.cubeSegmentCount = cubeDecision.applied ? cubeDecision.segments : 1;
  stats.vectorSegmentCount = vectorDecision.applied ? vectorDecision.segments : 1;
  stats.cubeTargetBytes = l0cBytes;
  stats.vectorTargetBytes = ubBytes;
  stats.cubeSubtileBytes =
      ceilDiv(stats.cubeWorkspaceBytes, stats.cubeSegmentCount);
  stats.vectorSubtileBytes =
      ceilDiv(stats.vectorWorkspaceBytes, stats.vectorSegmentCount);
  stats.handoffSegmentCount =
      std::max(stats.cubeSegmentCount, stats.vectorSegmentCount);
  stats.handoffSubtileBytes =
      ceilDiv(stats.handoffTileBytes, stats.handoffSegmentCount);
  stats.handoffTargetBytes = minPositiveLocalBytes(l0cBytes, ubBytes);
  stats.intermediateTargetBytes = ubBytes;

  bool anyKnownRequestedSide =
      (params.cubeLoop > 1 && cubeDecision.known) ||
      (params.vectorLoop > 1 && vectorDecision.known);
  if (!anyKnownRequestedSide)
    return stats;

  bool cubeApplied = cubeDecision.applied;
  bool vectorApplied = vectorDecision.applied;

  // P0-2: every modeled term below is a marginal number of cycles. There is no
  // baseCycles * empirical_ratio relief, no generic loop mismatch penalty, and
  // no reward merely for requesting a larger segment count.
  int64_t cubeBeforePressure = overflowTransferCycles(
      stats.cubeWorkspaceBytes, stats.cubeTargetBytes, 1,
      stats.cubeWorkspaceBytes, cubeTransferCycles);
  int64_t cubeAfterPressure = cubeApplied
      ? overflowTransferCycles(stats.cubeSubtileBytes, stats.cubeTargetBytes,
                               stats.cubeSegmentCount,
                               stats.cubeWorkspaceBytes, cubeTransferCycles)
      : cubeBeforePressure;
  int64_t vectorBeforePressure = overflowTransferCycles(
      stats.vectorWorkspaceBytes, stats.vectorTargetBytes, 1,
      stats.vectorWorkspaceBytes, vectorTransferCycles);
  int64_t vectorAfterPressure = vectorApplied
      ? overflowTransferCycles(stats.vectorSubtileBytes, stats.vectorTargetBytes,
                               stats.vectorSegmentCount,
                               stats.vectorWorkspaceBytes, vectorTransferCycles)
      : vectorBeforePressure;

  int64_t cubeDelta = cubeAfterPressure - cubeBeforePressure;
  int64_t vectorDelta = vectorAfterPressure - vectorBeforePressure;
  int64_t adjustedCubePath = std::max<int64_t>(1, cubePathCycles + cubeDelta);
  int64_t adjustedVectorPath =
      std::max<int64_t>(1, vectorPathCycles + vectorDelta);
  int64_t pressureAdjustedCycles =
      std::max(adjustedCubePath, adjustedVectorPath);

  stats.bufferDeltaCycles = cubeDelta + vectorDelta;
  stats.pipelineDeltaCycles = pressureAdjustedCycles - baseCycles;
  stats.externalSyncDeltaCycles = usingPassSummary
      ? tileMixExternalSyncDeltaCycles(params, config)
      : 0;
  int64_t extraSegments =
      (cubeApplied ? stats.cubeSegmentCount - 1 : 0) +
      (vectorApplied ? stats.vectorSegmentCount - 1 : 0);
  stats.scalarControlDeltaCycles =
      extraSegments * model.loopControlCyclesPerSegment;
  // GM bytes before/after are not yet emitted by the proprietary pass. Unknown
  // terms are zero by fail-closed policy, never inferred as a percentage.
  stats.gmDeltaCycles = 0;
  stats.workspaceReliefCycles = std::max<int64_t>(0, -stats.pipelineDeltaCycles);
  stats.bufferFitPenaltyCycles = cubeAfterPressure + vectorAfterPressure;
  stats.adjustedCycles = std::max<int64_t>(
      1, baseCycles + stats.gmDeltaCycles + stats.externalSyncDeltaCycles +
             stats.pipelineDeltaCycles + stats.scalarControlDeltaCycles);
  stats.netDeltaCycles = stats.adjustedCycles - baseCycles;
  stats.valid = true;
  stats.adjustmentApplied = cubeApplied || vectorApplied;
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

static KernelLaunchContext makeKernelLaunchContext(
    int64_t bodyCycles, const PipelineScheduler &scheduler,
    llvm::StringRef bindingsStr) {
  KernelLaunchContext ctx;
  ctx.bodyCycles = bodyCycles;
  ctx.opCount = scheduler.getAllOps().size();
  for (const PipelineOp &op : scheduler.getAllOps()) {
    ctx.hasVector |= op.hwUnit == HWUnit::Vector || op.hwUnit == HWUnit::VecMTE2;
    ctx.hasCube |= op.hwUnit == HWUnit::Cube || op.hwUnit == HWUnit::CubeMTE2 ||
                   op.hwUnit == HWUnit::FixPipe;
    ctx.hasMTE |= op.hwUnit == HWUnit::VecMTE2 || op.hwUnit == HWUnit::MTE3 ||
                  op.hwUnit == HWUnit::CubeMTE2 ||
                  op.hwUnit == HWUnit::FixPipe;
  }
  utils::inferKernelModeFromLaunchFeatures(ctx);
  utils::applyKernelLaunchBindings(ctx, bindingsStr);
  return ctx;
}

/// Generate Perfetto trace with loop unrolling.
/// If maxIterations > 0, limits the number of iterations shown in trace.
void generatePerfettoTrace(const PipelineScheduler &scheduler,
                           StringRef filename,
                           int64_t oneIterCycles,
                           int64_t bodyCycles,
                           int64_t launchOverheadCycles,
                           int64_t predictedTotalCycles,
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
  file << "    \"total_cycles\": " << bodyCycles << ",\n";
  file << "    \"body_cycles\": " << bodyCycles << ",\n";
  file << "    \"kernel_launch_overhead_cycles\": "
       << launchOverheadCycles << ",\n";
  file << "    \"predicted_total_cycles\": " << predictedTotalCycles << ",\n";
  file << "    \"trace_cycles\": " << traceTotalCycles << ",\n";
  file << "    \"iterations_shown\": " << numIterations << ",\n";
  file << "    \"iterations_total\": " << maxLoopMultiplier << ",\n";
  file << "    \"clock_freq_ghz\": " << config.getClockFrequencyGHz() << ",\n";
  file << "    \"estimated_time_us\": "
       << llvm::format("%.3f", config.cyclesToMicroseconds(bodyCycles))
       << ",\n";
  file << "    \"predicted_total_time_us\": "
       << llvm::format("%.3f",
                       config.cyclesToMicroseconds(predictedTotalCycles))
       << "\n";
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
  llvm::outs() << "actual body: " << bodyCycles
               << ", predicted total: " << predictedTotalCycles << ")\n";
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
    WorkspaceMultibufferParams workspaceMultibufferParams =
        parseWorkspaceMultibufferParams(compileParamsStr);
    
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

    // Total: max of paths (Cube and Vector paths overlap). The primary
    // pre-compilation path uses principle-backed TTIR eligibility. An optional
    // real TileCubeVectorLoop summary overrides that inference for validation.
    int64_t baseRooflineTotalCycles = std::max(cubePathCycles, vectorPathCycles);
    int64_t cubeTransferCycles =
        hwUnitCycles[HWUnit::CubeMTE2] + hwUnitCycles[HWUnit::FixPipe];
    int64_t vectorTransferCycles =
        hwUnitCycles[HWUnit::VecMTE2] + hwUnitCycles[HWUnit::MTE3];
    TileMixStats tileMixStats = estimateTileMix(
        tileMixParams, scheduler, config, cubePathCycles, vectorPathCycles,
        cubeTransferCycles, vectorTransferCycles, baseRooflineTotalCycles);
    WorkspaceMultibufferStats workspaceMultibufferStats =
        estimateWorkspaceMultibuffer(workspaceMultibufferParams, scheduler,
                                     config, cubePathCycles,
                                     vectorPathCycles);
    int64_t tileMixDeltaCycles =
        tileMixStats.valid ? tileMixStats.netDeltaCycles : 0;
    int64_t workspaceMultibufferDeltaCycles =
        workspaceMultibufferStats.valid
            ? workspaceMultibufferStats.netDeltaCycles
            : 0;
    int64_t rooflineTotalCycles = std::max<int64_t>(
        1, baseRooflineTotalCycles + tileMixDeltaCycles +
               workspaceMultibufferDeltaCycles);
    
    // Also calculate simple sum for comparison
    int64_t simpleSumCycles = 0;
    for (const auto &pipelineOp : scheduler.getAllOps())
      simpleSumCycles += pipelineOp.duration * pipelineOp.loopMultiplier;

    KernelLaunchContext launchCtx =
        makeKernelLaunchContext(rooflineTotalCycles, scheduler, argBindingsStr);
    KernelLaunchEstimate launch =
        config.estimateKernelLaunchOverhead(launchCtx);
    int64_t predictedTotalCycles = rooflineTotalCycles + launch.totalCycles;
    
    module->setAttr("ascend.scheduled_cycles_one_iter",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), oneIterCycles));
    module->setAttr("ascend.roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), rooflineTotalCycles));
    module->setAttr("ascend.base_roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), baseRooflineTotalCycles));
    module->setAttr("ascend.kernel_body_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     rooflineTotalCycles));
    module->setAttr("ascend.kernel_launch_overhead_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     launch.totalCycles));
    module->setAttr("ascend.predicted_total_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                     predictedTotalCycles));
    if (tileMixStats.used) {
      module->setAttr("ascend.tile_mix_schedule_model",
                      StringAttr::get(module.getContext(), "ttir_principle_marginal_cycles_v5_target_trip_peer_model"));
      module->setAttr("ascend.tile_mix_model_valid",
                      BoolAttr::get(module.getContext(), tileMixStats.valid));
      module->setAttr("ascend.tile_mix_adjustment_applied",
                      BoolAttr::get(module.getContext(), tileMixStats.adjustmentApplied));
      module->setAttr("ascend.tile_mix_confidence_percent",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.confidencePercent));
      module->setAttr("ascend.tile_mix_summary_source",
                      StringAttr::get(module.getContext(), tileMixStats.summarySource));
      module->setAttr("ascend.tile_mix_cube_applied",
                      BoolAttr::get(module.getContext(), tileMixStats.cubeApplied));
      module->setAttr("ascend.tile_mix_vector_applied",
                      BoolAttr::get(module.getContext(), tileMixStats.vectorApplied));
      module->setAttr("ascend.tile_mix_cube_skip_reason",
                      StringAttr::get(module.getContext(), tileMixStats.cubeSkipReason));
      module->setAttr("ascend.tile_mix_vector_skip_reason",
                      StringAttr::get(module.getContext(), tileMixStats.vectorSkipReason));
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
      module->setAttr("ascend.tile_mix_delta_gm_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.gmDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_external_sync_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.externalSyncDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_buffer_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.bufferDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_pipeline_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.pipelineDeltaCycles));
      module->setAttr("ascend.tile_mix_delta_scalar_control_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.scalarControlDeltaCycles));
    }
    if (workspaceMultibufferStats.used) {
      module->setAttr("ascend.workspace_multibuffer_schedule_model",
                      StringAttr::get(module.getContext(), "ttir_finite_fifo_v1"));
      module->setAttr("ascend.workspace_multibuffer_model_valid",
                      BoolAttr::get(module.getContext(), workspaceMultibufferStats.valid));
      module->setAttr("ascend.workspace_multibuffer_adjustment_applied",
                      BoolAttr::get(module.getContext(), workspaceMultibufferStats.adjustmentApplied));
      module->setAttr("ascend.workspace_multibuffer_slots",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.requestedSlots));
      module->setAttr("ascend.workspace_multibuffer_reference_slots",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.referenceSlots));
      module->setAttr("ascend.workspace_multibuffer_slot_delta",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.slotDelta));
      module->setAttr("ascend.workspace_multibuffer_extra_slots",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.extraSlots));
      module->setAttr("ascend.workspace_multibuffer_family_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.workspaceFamilyCount));
      module->setAttr("ascend.workspace_multibuffer_cube_to_vector_family_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.cubeToVectorFamilyCount));
      module->setAttr("ascend.workspace_multibuffer_vector_to_cube_family_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.vectorToCubeFamilyCount));
      module->setAttr("ascend.workspace_multibuffer_bytes_per_slot",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.workspaceBytesPerSlot));
      module->setAttr("ascend.workspace_multibuffer_iteration_count",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.iterationCount));
      module->setAttr("ascend.workspace_multibuffer_cube_to_vector_iterations",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.cubeToVectorIterations));
      module->setAttr("ascend.workspace_multibuffer_vector_to_cube_iterations",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.vectorToCubeIterations));
      module->setAttr("ascend.workspace_multibuffer_cube_producer_tail_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.cubeProducerTailCycles));
      module->setAttr("ascend.workspace_multibuffer_vector_producer_tail_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.vectorProducerTailCycles));
      module->setAttr("ascend.workspace_multibuffer_sync_pair_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.syncPairCycles));
      module->setAttr("ascend.workspace_multibuffer_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.syncDeltaCycles));
      module->setAttr("ascend.workspace_multibuffer_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.blockingCycles));
      module->setAttr("ascend.workspace_multibuffer_reference_blocking_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.referenceBlockingCycles));
      module->setAttr("ascend.workspace_multibuffer_producer_wait_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.producerWaitReliefCycles));
      module->setAttr("ascend.workspace_multibuffer_reference_queue_penalty_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.referenceQueuePenaltyCycles));
      module->setAttr("ascend.workspace_multibuffer_overlap_relief_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.overlapReliefCycles));
      module->setAttr("ascend.workspace_multibuffer_queue_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.queueDeltaCycles));
      module->setAttr("ascend.workspace_multibuffer_net_delta_cycles",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), workspaceMultibufferStats.netDeltaCycles));
    }
    if (tileMixStats.used) {
      module->setAttr("ascend.tile_mix_sync_ops_before",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.syncOpsBefore));
      module->setAttr("ascend.tile_mix_sync_ops_after",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.syncOpsAfter));
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
      module->setAttr("ascend.tile_mix_handoff_subtile_bytes",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffSubtileBytes));
      module->setAttr("ascend.tile_mix_handoff_segments",
                      IntegerAttr::get(IntegerType::get(module.getContext(), 64), tileMixStats.handoffSegmentCount));
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
      llvm::outs() << "  Tile mix pass summary: source="
                   << tileMixStats.summarySource
                   << ", confidence=" << tileMixStats.confidencePercent
                   << "%"
                   << ", cube_applied="
                   << (tileMixStats.cubeApplied ? "true" : "false")
                   << ", vector_applied="
                   << (tileMixStats.vectorApplied ? "true" : "false")
                   << ", cube_skip_reason=" << tileMixStats.cubeSkipReason
                   << ", vector_skip_reason=" << tileMixStats.vectorSkipReason
                   << "\n";
      llvm::outs() << "  Tile mix eligibility: "
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
                   << ", subtile_bytes="
                   << tileMixStats.handoffSubtileBytes
                   << ", segments="
                   << tileMixStats.handoffSegmentCount
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
    }
    if (workspaceMultibufferStats.used) {
      llvm::outs() << "  Workspace multibuffer params: requested_slots="
                   << workspaceMultibufferParams.requestedSlots << "\n";
      llvm::outs() << "  Workspace multibuffer: slots="
                   << workspaceMultibufferStats.requestedSlots
                   << ", reference_slots="
                   << workspaceMultibufferStats.referenceSlots
                   << ", slot_delta="
                   << workspaceMultibufferStats.slotDelta
                   << ", extra_slots="
                   << workspaceMultibufferStats.extraSlots
                   << ", workspace_families="
                   << workspaceMultibufferStats.workspaceFamilyCount
                   << ", cube_to_vector_families="
                   << workspaceMultibufferStats.cubeToVectorFamilyCount
                   << ", vector_to_cube_families="
                   << workspaceMultibufferStats.vectorToCubeFamilyCount
                   << ", bytes_per_slot="
                   << workspaceMultibufferStats.workspaceBytesPerSlot
                   << ", iterations="
                   << workspaceMultibufferStats.iterationCount
                   << ", cube_to_vector_iterations="
                   << workspaceMultibufferStats.cubeToVectorIterations
                   << ", vector_to_cube_iterations="
                   << workspaceMultibufferStats.vectorToCubeIterations
                   << ", cube_producer_tail_cycles="
                   << workspaceMultibufferStats.cubeProducerTailCycles
                   << ", vector_producer_tail_cycles="
                   << workspaceMultibufferStats.vectorProducerTailCycles
                   << ", sync_pair_cycles="
                   << workspaceMultibufferStats.syncPairCycles
                   << ", sync_delta_cycles="
                   << workspaceMultibufferStats.syncDeltaCycles
                   << ", blocking_cycles="
                   << workspaceMultibufferStats.blockingCycles
                   << ", reference_blocking_cycles="
                   << workspaceMultibufferStats.referenceBlockingCycles
                   << ", producer_wait_relief_cycles="
                   << workspaceMultibufferStats.producerWaitReliefCycles
                   << ", reference_queue_penalty_cycles="
                   << workspaceMultibufferStats.referenceQueuePenaltyCycles
                   << ", overlap_relief_cycles="
                   << workspaceMultibufferStats.overlapReliefCycles
                   << ", queue_delta_cycles="
                   << workspaceMultibufferStats.queueDeltaCycles
                   << ", net_delta_cycles="
                   << workspaceMultibufferStats.netDeltaCycles << "\n";
      llvm::outs() << "  Workspace multibuffer marginal cycles: sync="
                   << workspaceMultibufferStats.syncDeltaCycles
                   << ", queue=" << workspaceMultibufferStats.queueDeltaCycles
                   << ", relief=" << workspaceMultibufferStats.overlapReliefCycles
                   << ", net=" << workspaceMultibufferStats.netDeltaCycles
                   << "\n";
    }
    if (tileMixStats.used) {
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
      llvm::outs() << "  Tile mix marginal cycles: gm="
                   << tileMixStats.gmDeltaCycles
                   << ", external_sync="
                   << tileMixStats.externalSyncDeltaCycles
                   << ", buffer=" << tileMixStats.bufferDeltaCycles
                   << ", pipeline=" << tileMixStats.pipelineDeltaCycles
                   << ", scalar_control="
                   << tileMixStats.scalarControlDeltaCycles << "\n";
    }
    llvm::outs() << "  Combined feature model delta cycles: tile_mix="
                 << tileMixDeltaCycles << ", workspace_multibuffer="
                 << workspaceMultibufferDeltaCycles << "\n";
    llvm::outs() << "  Roofline model (TTIR principle marginal tile mix): " << rooflineTotalCycles
                 << " (" << llvm::format("%.3f", config.cyclesToMicroseconds(rooflineTotalCycles)) << " us)\n";
    llvm::outs() << "  Kernel launch overhead: " << launch.totalCycles
                 << " ("
                 << llvm::format("%.3f",
                                 config.cyclesToMicroseconds(launch.totalCycles))
                 << " us)";
    if (launch.blockDim > 0)
      llvm::outs() << ", block_dim=" << launch.blockDim;
    if (launch.numWaves > 0)
      llvm::outs() << ", waves=" << launch.numWaves;
    llvm::outs() << "\n";
    llvm::outs() << "  Predicted total: " << predictedTotalCycles
                 << " ("
                 << llvm::format(
                        "%.3f",
                        config.cyclesToMicroseconds(predictedTotalCycles))
                 << " us)\n";
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
                            oneIterCycles, rooflineTotalCycles,
                            launch.totalCycles, predictedTotalCycles);
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
