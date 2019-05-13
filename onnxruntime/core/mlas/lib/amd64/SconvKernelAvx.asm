;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;   SconvKernelAvx.asm
;
; Abstract:
;
;   This module implements the kernels for the single precision convolution
;   operation.
;
;   This implementation uses AVX instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
INCLUDE SconvKernelAvxCommon.inc
        .list

;
; ComputeBlock
;
;   This macro multiplies and accumulates for FilterCount by OutputCount block
;   of the output buffer.
;
; Arguments:
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
;   OutputCount - Supplies the number of output blocks to produce.
;
;   VectorOffset - Supplies the byte offset from the filter buffer to fetch
;       elements.
;
;   BroadcastOffset - Supplies the byte offset from the input buffer to fetch
;       elements.
;
; Implicit Arguments:
;
;   rcx - Supplies the address of the input buffer.
;
;   rdx - Supplies the address of the filter buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description).
;
;   rbx - Supplies the address of the filter buffer plus 2 * FilterStride.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   ymm0-ymm7 - Supplies the block accumulators.
;

ComputeBlock MACRO KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

IFIDNI <KernelType>, <Depthwise>
        vmovups ymm12,YMMWORD PTR [rdx]
        EmitIfCountGE OutputCount, 1, <vmulps ymm8,ymm12,YMMWORD PTR [rcx]>
        EmitIfCountGE OutputCount, 1, <vaddps ymm0,ymm0,ymm8>
        EmitIfCountGE OutputCount, 2, <vmulps ymm9,ymm12,YMMWORD PTR [rcx+r9]>
        EmitIfCountGE OutputCount, 2, <vaddps ymm4,ymm4,ymm9>
ELSE
        EmitIfCountGE OutputCount, 1, <vbroadcastss ymm13,DWORD PTR [rcx+BroadcastOffset]>
        EmitIfCountGE OutputCount, 2, <vbroadcastss ymm14,DWORD PTR [rcx+r9+BroadcastOffset]>
IF OutputCount EQ 1
        EmitIfCountGE FilterCount, 1, <vmulps ymm8,ymm13,YMMWORD PTR [rdx+VectorOffset]>
        EmitIfCountGE FilterCount, 1, <vaddps ymm0,ymm0,ymm8>
        EmitIfCountGE FilterCount, 2, <vmulps ymm9,ymm13,YMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 2, <vaddps ymm1,ymm1,ymm9>
        EmitIfCountGE FilterCount, 3, <vmulps ymm10,ymm13,YMMWORD PTR [rbx+VectorOffset]>
        EmitIfCountGE FilterCount, 3, <vaddps ymm2,ymm2,ymm10>
        EmitIfCountGE FilterCount, 4, <vmulps ymm11,ymm13,YMMWORD PTR [rbx+rsi+VectorOffset]>
        EmitIfCountGE FilterCount, 4, <vaddps ymm3,ymm3,ymm11>
ELSE
        EmitIfCountGE FilterCount, 1, <vmovups ymm12,YMMWORD PTR [rdx+VectorOffset]>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vmulps ymm8,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 1, <vaddps ymm0,ymm0,ymm8>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vmulps ymm9,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 1, OutputCount, 2, <vaddps ymm4,ymm4,ymm9>
        EmitIfCountGE FilterCount, 2, <vmovups ymm12,YMMWORD PTR [rdx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vmulps ymm10,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 2, OutputCount, 1, <vaddps ymm1,ymm1,ymm10>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vmulps ymm11,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 2, OutputCount, 2, <vaddps ymm5,ymm5,ymm11>
        EmitIfCountGE FilterCount, 3, <vmovups ymm12,YMMWORD PTR [rbx+VectorOffset]>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vmulps ymm8,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 1, <vaddps ymm2,ymm2,ymm8>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vmulps ymm9,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 3, OutputCount, 2, <vaddps ymm6,ymm6,ymm9>
        EmitIfCountGE FilterCount, 4, <vmovups ymm12,YMMWORD PTR [rbx+rsi+VectorOffset]>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vmulps ymm10,ymm13,ymm12>
        EmitIfCount2GE FilterCount, 4, OutputCount, 1, <vaddps ymm3,ymm3,ymm10>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vmulps ymm11,ymm14,ymm12>
        EmitIfCount2GE FilterCount, 4, OutputCount, 2, <vaddps ymm7,ymm7,ymm11>
ENDIF
ENDIF

        ENDM

;
; ProcessFilterCountN
;
;   This macro generates code to compute the convolution for a specified number
;   of filter rows.
;
; Arguments:
;
;   KernelFrame - Supplies the symbol name to access the convolution kernel
;       stack.
;
;   KernelType - Supplies the type of kernel to be generated.
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
; Implicit Arguments:
;
;   rdi - Supplies the address of the input buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description) when
;       KernelType!=Depthwise. Supplies the address of the filter buffer when
;       KernelType=Depthwise.
;
;   rbp - Supplies the DilationWidth parameter (see function description).
;
;   r8 - Supplies the address of the output buffer.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r15 - Supplies the InputStride parameter (see function description).
;

ProcessFilterCountN MACRO KernelFrame, KernelType, FilterCount

        LOCAL   ProcessOutputCount
        LOCAL   ProcessNextOutputCountBy2
        LOCAL   ProcessRemainingOutputCount
        LOCAL   ProcessOutputCountRightPadAndRemaining

;
; Process the output blocks that include left padding.
;

        mov     r10,KernelFrame.OutputCountLeftPad[rsp]
        test    r10,r10
        jz      ProcessOutputCount
        call    MlasConv&KernelType&FloatSingleAvxFilterCount&FilterCount

;
; Process the output blocks that do not include any padding.
;

ProcessOutputCount:
        mov     r10,KernelFrame.OutputCount[rsp]
        sub     r10,2
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy2:
        ProcessOutputCountN KernelFrame, KernelType, 8, FilterCount, 2
        lea     rdi,[rdi+r9*2]              ; advance input by 2 elements
        sub     r10,2
        jae     ProcessNextOutputCountBy2

ProcessRemainingOutputCount:
        add     r10,2                       ; correct for over-subtract above

;
; Process the output blocks that include right padding plus any remaining output
; blocks from above.
;

ProcessOutputCountRightPadAndRemaining:
        add     r10,KernelFrame.OutputCountRightPad[rsp]
        jz      ExitKernel
        call    MlasConv&KernelType&FloatSingleAvxFilterCount&FilterCount

        ENDM

;
; ProcessPointwiseFilterCountN
;
;   This macro generates code to compute the convolution for a specified number
;   of filter rows for a pointwise convolution.
;
; Arguments:
;
;   FilterCount - Supplies the number of rows from the filter to process.
;
; Implicit Arguments:
;
;   rdi - Supplies the address of the input buffer.
;
;   rsi - Supplies the FilterStride parameter (see function description).
;
;   rbp - Supplies the InputStride parameter (see function description).
;
;   r8 - Supplies the address of the output buffer.
;
;   r9 - Supplies the StrideWidth parameter (see function description).
;
;   r10 - Supplies the OutputCount parameter (see function description).
;
;   r12 - Supplies the address of the filter buffer.
;

ProcessPointwiseFilterCountN MACRO FilterCount

        LOCAL   ProcessNextOutputCountBy2
        LOCAL   ProcessRemainingOutputCount

        sub     r10,2
        jb      ProcessRemainingOutputCount

ProcessNextOutputCountBy2:
        ProcessPointwiseOutputCountN 8, FilterCount, 2
        lea     rdi,[rdi+r9*2]              ; advance input by 2 elements
        sub     r10,2
        jae     ProcessNextOutputCountBy2

ProcessRemainingOutputCount:
        add     r10,2                       ; correct for over-subtract above
        jz      ExitKernel
        ProcessPointwiseOutputCountN 8, FilterCount, 1

        ENDM

;
; Generate the convolution kernels.
;

SconvKernelFunction Nchw, 8, Avx
SconvKernelFunction Nchwc, 8, Avx, BiasFilter
SconvKernelDepthwiseFunction 8, Avx
SconvKernelPointwiseFunction Avx, BiasFilter

        END
