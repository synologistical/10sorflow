/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_

include "mlir/Pass/PassBase.td"

def TritonXLAExtractInsertToTritonPass : Pass<"triton-xla-extract-insert-to-triton", "mlir::ModuleOp"> {
  let summary = "Convert XLA Triton tile, extract and, insert ops to Triton ops.";
  let description = [{
    This pass converts `triton_xla.extract`, `triton_xla.insert` and
    `triton_xla.tile` ops to Triton ops. It also rewrites `func` args to
    `tt.ptr` types and removes function return args.
  }];
  let dependentDialects = [
    "triton::TritonDialect",
  ];
  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto">,
    Option<"tma_enabled_", "tma_enabled", "bool", /*default=*/"false",
           "Flag to enable/disable TMA">,
  ];
  let constructor = "CreateTritonXLAExtractInsertToTritonPass()";
}

def GeneralizeKernelSignaturePass
    : Pass<"generalize-kernel-signature"> {
  let summary = "Rewrite kernels to use generic data pointer arguments.";
  let description = [{
    Rewrite signatures of kernel functions from global pointers to generic
    pointers and cast them to global ones within the kernel.
  }];
  let constructor = "CreateGeneralizeKernelSignaturePass()";
}

def SparseAddEncodingPass : Pass<"sparse-add-encoding", "mlir::ModuleOp"> {
  let summary = "Add sparse encoding for all the arguments of a SparseDotOp.";
  let options = [
    Option<"num_warps_", "num-warps", "int32_t", /*default=*/"4",
           "Number of warps">,
    Option<"threads_per_warp_", "threads-per-warp", "int32_t", /*default=*/"32",
           "Number of threads per warp">,
    Option<"num_ctas_", "num-ctas", "int32_t", /*default=*/"1",
           "Number of CTAs in a CGA">,
  ];
  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
  ];
  let constructor = "CreateSparseAddEncodingPass()";
}

def SparseBlockedToMMAPass : Pass<"sparse-blocked-to-mma", "mlir::ModuleOp"> {
  let summary = "Add convert layouts to/from MMA before and after SparseDotOp.";
  let description = [{
     Add convert layouts to and from MMA before and after SparseDotOp. In MMAV3,
     shared memory allocations will be used for A and B operands.
  }];
  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
  ];
  let constructor = "CreateSparseBlockedToMMAPass()";
}

def SparseRemoveLayoutConversionPass
    : Pass<"sparse-remove-layout-conversion", "mlir::ModuleOp"> {
  let summary = "Replaces ConvertLayoutOp with sparse dot encoding";
  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
  ];
  let constructor = "CreateSparseRemoveLayoutConversionPass()";
}

def SparseLocalLoadToLLVMPass
    : Pass<"sparse-local-load-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers sparse local load to LLVM";
  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
    "mlir::LLVM::LLVMDialect"
  ];
  let constructor = "CreateSparseLocalLoadToLLVMPass()";
}

def SparseDotOpToLLVMPass : Pass<"sparse-dot-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers sparse dot to LLVM";
  let constructor = "CreateSparseDotOpToLLVMPass()";
  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
    "mlir::triton::nvgpu::NVGPUDialect",
  ];
}

def SparseWGMMAOpToLLVMPass : Pass<"sparse-wgmma-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lowers sparse WGMMA to LLVM";
  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
    "mlir::triton::nvgpu::NVGPUDialect",
  ];
  let constructor = "CreateSparseWGMMAOpToLLVMPass()";
}

def PreventMmaV3LoopUnrollingPass
    : Pass<"prevent-mmav3-loop-unrolling", "mlir::ModuleOp"> {
  let summary = "Prevent MMAv3 loop unrolling.";
  let description = [{
    This pass is a result of b/344841434:
    PTX sometimes unrolls wgmma loops that can cause a 1000x slow down in
    compilation time. Most unrolling has already been done before PTX,
    this pragma prevents ptxas from doing more.
  }];
  let constructor = "CreatePreventMmaV3LoopUnrollingPass()";
}

def LoadInt4RewritePass
    : Pass<"int4-to-packed-int4-rewrite", "mlir::ModuleOp"> {
  let summary = "Converts ops with int4 tensors to the ops with int4 packed to int8 tensors.";
  let description = [{
    This pass replaces the int4 tensors with the int4 packed to int8 tensor of
    the twice smaller size. It also replaces the plain ExtSIOp upcast to the
    int8 tensor with the unpack sequence.
  }];
  let constructor = "CreateInt4ToPackedInt4RewritePass()";
}

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_
