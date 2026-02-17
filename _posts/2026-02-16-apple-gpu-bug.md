---
layout: post
title:  "Finding a bug in Apple's Metal Shader compiler"
date:   2025-11-28 20:49:00 -0500
categories: jekyll update
---

# Metal GPU Compiler Bug: Discovery Process

## How the bug was found, step by step

### Origin: A Commented-Out Test in tinygrad

The trail starts at `test/null/test_uops.py:164`:

```python
# @unittest.skipIf(Device.DEFAULT == "METAL", "compiler bug")
```

This commented-out skip decorator sits above the `TestFastIdiv` class, which tests tinygrad's **fast integer division** optimization — replacing `x / D` and `x % D` with the classic multiply-shift trick: `(int)(((long)(x) * magic) >> shift)`. The comment tells the whole story: Metal was producing wrong results, and a compiler bug was suspected.

The actual triggering workload was **SHA3/Keccak**: `Tensor(bytes(range(9))).keccak("sha3_256")`. Keccak's permutation operates on a 5x5 state, so it heavily uses `% 5` to wrap indices — which tinygrad compiles into fast\_idiv with magic constant `1717986919` and shift `33`.

---

### Phase 1: `metalbug_minimal.py` — Finding the Input Size Boundary

The first investigation was at the **tensor level**, not the shader level. SHA3 was run on inputs of sizes 1 through 10 bytes (`metalbug_minimal.py:56`):

```python
for size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    result_hex = bytes(Tensor(data).keccak("sha3_256").tolist()).hex()
    expected_hex = hashlib.sha3_256(data).digest().hex()
```

**8 bytes: PASS. 9 bytes: FAIL.**

The reason 9 bytes is the critical boundary is visible in the generated kernel. tinygrad names the kernel `E_25n3` and its data buffer `data2_9` — 9 unsigned chars. The kernel conditionally loads from this buffer using the remainder as an index:

```metal
unsigned char val1 = (cond) ? *(data2_9 + (remainder << 3)) : (unsigned char)(0u);
```

When `remainder == 1`, the load address is `1 << 3 = 8`, i.e., the **9th byte** (`data2_9[8]`). With a 9-byte input like `bytes(range(9))`, that byte exists and has value **8**. The bug causes the load to return **0** instead. With 8 bytes of input, the buffer is only 8 bytes long — `data[8]` doesn't exist, so the kernel shape changes and the buggy pattern doesn't arise.

The earlier scratch file `metalbug.py` also shows the initial investigation path: first testing basic division and modulo operations (which passed), then testing at the SHA3-256 rate boundary of 136 bytes with `sha3_224` (which also failed), before narrowing down to the exact byte count.

---

### Phase 2: `metalbug_e25n3.py` — The Exact Generated Kernel

The exact kernel tinygrad generates for `Tensor(bytes(range(9))).keccak("sha3_256")` was captured — the `E_25n3` kernel (25 threads). This ~130-line kernel does:

1. Load a permuted index (`reorder_indexes` from the Keccak pi step)
2. Compute `(alu0+1) % 5` and `(alu0+4) % 5` using fast\_idiv — **twice each**, once via long arithmetic (for the condition) and once via int arithmetic (for the address)
3. Conditionally load bytes using the int-path remainder as address
4. Assemble 8 bytes into `unsigned long` values via XOR chains
5. Apply rotation and mix operations

The generated code has two separate computations of `(alu0+1) % 5`:

```metal
// Long-path: for the condition
long alu2 = (cast0 + 1ll);
int cast1 = ((int)((alu2 - (5ll * (((alu2 * 1717986919ll) >> 33ll) + alu3)))));
bool cond = (cast1 < 2) & ...;

// Int-path: for the buffer address (inside the ternary)
unsigned char val1 = (cond) ? *(data2_9 + ((alu1 - (5 * (((int)(((long)(alu1)) * 1717986919ll >> 33ll)) + alu4))) << 3)) : 0;
```

Both paths compute the same mathematical value (`(alu0+1) % 5`), but using different intermediate types. This is the pattern that triggers the optimizer to make different decisions about each one.

A diagnostic variant showed that the **fast\_idiv remainders themselves were correct** when output to a buffer. The bug was in how the remainder was used as a buffer load address inside a conditional.

---

### Phase 3: `metalbug_paredown.py` — Systematic Reduction

This was the key breakthrough file. The E\_25n3 kernel was stripped layer by layer:

| Version | What it tests | Result | Significance |
|---------|--------------|--------|-------------|
| **A** | Just the first conditional load (`val1`) — no XOR chains, no rotation | **FAIL** | Bug doesn't need full Keccak complexity |
| **B** | Just the second conditional load (`val2`, using `+4` instead of `+1`) | **FAIL** | Both modulo variants affected |
| **C** | Both loads + XOR assembly | **FAIL** | Expected (superset of A/B) |
| **D** | Int-path fast\_idiv for **both** condition and index (no long-path) | **PASS** | **Critical**: bug needs different variables for condition vs index |
| **E** | Direct buffer load using fast\_idiv, no conditional | **PASS** | The conditional branch is necessary |
| **F** | Full kernel but only ONE fast\_idiv (val1), rest use native `%` | **FAIL** | One fast\_idiv suffices |

**Key insight**: The bug triggers when the **condition** is derived from a **different computation** (long-path modulo) than the **address** (int-path fast\_idiv). Using the same variable for both avoids the bug because the optimizer makes structurally different decisions about the IR (see Phase 6).

---

### Phase 4: `metalbug_minimal_repro.py` — Hypothesis Testing

With the minimal failing kernel identified (Version A: ~15 lines of Metal), a systematic hypothesis battery was run:

| Test | Setup | Result | Conclusion |
|------|-------|--------|------------|
| **H1** | Native `%` for condition, fast\_idiv for index | FAIL | Condition doesn't need to be fast\_idiv — any separate variable triggers it |
| **H2** | Fast\_idiv for condition, native `%` for index | PASS | Bug requires fast\_idiv specifically in the **address/index** |
| **H3** | Both long-path fast\_idiv | PASS | Same-type computation avoids bug |
| **H4** | Both int-path fast\_idiv | PASS | Same variable for cond+addr avoids bug |
| **H5** | Different div method for condition + fast\_idiv index | FAIL | Confirmed: any separate condition + fast\_idiv index |
| **H6** | `volatile` to block CSE/optimizer fusion | **PASS** | **Smoking gun**: preventing optimizer fusion avoids the bug |
| **H7** | Store int-path to variable first | FAIL | Storing to variable alone doesn't help (optimizer sees through it) |

H6 was the smoking gun: `volatile` prevents the optimizer from fusing the multiply-by-(-5) with the shift, and the bug disappears.

---

### Phase 5: `metalbug_deep.py` — Pinpointing the Mechanism

To understand whether the cause was wrong **address** computation, wrong **branch elimination**, or wrong **load**:

- **H8/H10**: Unconditional loads with long-path present but unused in condition — remainders correct, loads correct. The fast\_idiv itself is computed correctly in isolation.
- **H9**: Always-true condition (`cast1 < 100`) with the buggy pattern — still **FAIL**. Even though the condition is provably always true for inputs 0-24, the optimizer still emits a conditional branch and moves the address computation inside it, triggering the bug.
- **H11**: Output the condition, remainder, address, AND loaded value simultaneously — revealed that the condition was true (correct), the remainder was correct, the address was correct, but **the load returned 0** for certain threads. The GPU backend was misexecuting the load from a correctly computed address.

---

### Phase 6: `metalbug_ir.py` + `metalbug_asm.py` — LLVM IR Analysis

The buggy and passing kernels were compiled to LLVM IR text (`xcrun metal -O2 -S -emit-llvm`). The generated `.ll` files are in `test/air-llvm/`. Comparing them reveals exactly what the optimizer does differently.

#### How -O2 transforms the buggy kernel

At **-O0** (`air-llvm/bug_O0.ll`), the buggy kernel's address computation is straightforward and correct. Inside the true-branch (label `%80`, line 106), the address is computed using the original formula with `mul nsw i32 5, %89` (multiply by 5), then subtracted:

```llvm
; bug_O0.ll lines 108-121 — inside the true-branch at -O0
%84 = sext i32 %83 to i64
%85 = mul nsw i64 %84, 1717986919     ; fast-div: alu1 * magic
%86 = ashr i64 %85, 33                ; >> 33 = quotient
%87 = trunc i64 %86 to i32
%89 = add nsw i32 %87, %88            ; q + neg_correction
%90 = mul nsw i32 5, %89              ; 5 * (q + neg)
%91 = sub nsw i32 %82, %90            ; alu1 - 5*q = remainder
%92 = shl i32 %91, 3                  ; remainder << 3 = byte offset
%93 = sext i32 %92 to i64
%94 = getelementptr inbounds i8, ... %93
%95 = load i8, ...                    ; load data[remainder * 8]
```

At **-O2** (`air-llvm/bug_O2.ll`), the optimizer **moves the address computation inside the conditional branch** and fuses `mul by -5` with `shl by 3`. The entire true-branch (label `%31`, lines 36-51) becomes:

```llvm
; bug_O2.ll lines 36-50 — inside the true-branch at -O2
31:
  %32 = zext i1 %14 to i32
  %33 = add nsw i32 %11, 1              ; alu1 = alu0 + 1
  %34 = sext i32 %33 to i64
  %35 = mul nsw i64 %34, 1717986919     ; fast-div: alu1 * magic
  %36 = lshr i64 %35, 33                ; >> 33 = quotient
  %37 = trunc i64 %36 to i32
  %38 = add nuw i32 %37, %32            ; q + neg_correction
  %39 = mul i32 %38, 536870907          ; <<< THE FUSED CONSTANT
  %40 = add i32 %39, %33                ; + alu1
  %41 = shl i32 %40, 3                  ; << 3
  %42 = sext i32 %41 to i64             ; sign-extend for address
  %43 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %42
  %44 = load i8, i8 addrspace(1)* %43   ; MISCOMPILED LOAD
  %45 = zext i8 %44 to i32
```

**Line 39 is the critical transformation.** Instead of `alu1 - 5 * q` then `<< 3`, the optimizer algebraically fused the operations into `q * 536870907 + alu1` then `<< 3`. This is valid because in 32-bit modular arithmetic: `(alu1 - 5*q) << 3 == (q * 536870907 + alu1) << 3`, since `536870907 * 8 = 4,294,967,256 = 2^32 - 40 ≡ -40 (mod 2^32)` and `-5 * 8 = -40`.

**The LLVM IR is mathematically correct. The GPU backend miscompiles it.**

#### How the passing kernel avoids the bug

In `air-llvm/pass_O2.ll`, the remainder is needed **before** the branch (for the condition), so the optimizer computes it pre-branch using `mul i32 %19, -5`:

```llvm
; pass_O2.ll lines 19-34 — BEFORE the branch
%16 = mul nsw i64 %15, 1717986919       ; fast-div: alu1 * magic
%17 = ashr i64 %16, 33                  ; >> 33 = quotient
%18 = trunc i64 %17 to i32
%19 = add nsw i32 %18, %14              ; q + neg_correction
%20 = mul i32 %19, -5                   ; q * (-5)          ← SAFE CONSTANT
%21 = add i32 %20, %12                  ; alu1 + (-5)*q = remainder
%22 = icmp slt i32 %21, 2              ; condition: remainder < 2
...
br i1 %30, label %37, label %31        ; branch on condition

31:                                      ; true-branch: just shift and load
  %32 = shl i32 %21, 3                  ; remainder << 3 (reuses %21 from above)
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %33
  %35 = load i8, i8 addrspace(1)* %34   ; CORRECT LOAD
```

The true-branch is trivial — just `shl`, `sext`, `gep`, `load`. No `mul i32 X, 536870907`, no i32 overflow.

#### The H9 kernel: even `always_true` gets the fused constant

In `air-llvm/h9_O2.ll`, the condition is `cast1 < 100` (always true for inputs 0-24). But the optimizer still emits a conditional branch (it can't statically prove always-true for arbitrary runtime inputs), and the true-branch at label `%23` still contains the fused constant:

```llvm
; h9_O2.ll lines 28-41 — inside the true-branch
23:
  ...
  %31 = mul i32 %30, 536870907          ; same fused constant
  %32 = add i32 %31, %25
  %33 = shl i32 %32, 3
  ...
  %36 = load i8, i8 addrspace(1)* %35   ; MISCOMPILED LOAD
```

This confirms the bug isn't about condition evaluation — it's purely about what IR pattern ends up inside the branch.

#### The T6 kernel: same variable forces `-5`

In `air-llvm/t6_O2.ll`, using the same `remainder` variable for both condition and address produces identical IR structure to `pass_O2.ll`:

```llvm
; t6_O2.ll lines 23-24 — BEFORE the branch
%20 = mul i32 %19, -5                   ; q * (-5)          ← SAFE
%21 = add i32 %20, %12                  ; remainder

; t6_O2.ll lines 36-40 — inside the true-branch
31:
  %32 = shl i32 %21, 3                  ; just shift and load
```

No `536870907`, no bug.

#### Summary of the IR evidence

| File | `536870907` in branch? | `mul i32 X, -5` pre-branch? | Runtime result |
|------|:---:|:---:|:---:|
| `bug_O2.ll` | **Yes** (line 44) | No | **FAIL** |
| `h9_O2.ll` | **Yes** (line 36) | No | **FAIL** |
| `t1_O2.ll` | **Yes** | No | **FAIL** |
| `t2_O2.ll` | **Yes** | No | **FAIL** |
| `pass_O2.ll` | No | **Yes** (line 23) | PASS |
| `t6_O2.ll` | No | **Yes** (line 23) | PASS |
| `reproducer_O2.ll` `@buggy` | **Yes** (line 39) | No | **FAIL** |
| `reproducer_O2.ll` `@working` | No | **Yes** (line 72) | PASS |

100% correlation across all test variants.

---

### Phase 7: `metalbug_constant.py` — Confirming the Constant

Six targeted tests varying the constant and the context:

| Test | Setup | Result | IR contains 536870907? |
|------|-------|--------|:---:|
| **T1** | Original bug pattern | FAIL | Yes |
| **T2** | Explicit `536870907` written in source | FAIL | Yes |
| **T3** | Explicit `-5`, remainder as variable | PASS | No |
| **T4** | `536870907` but **unconditional** (no branch) | PASS | Yes (but no branch) |
| **T5** | `volatile` prevents fusion | PASS | No |
| **T6** | Same variable for cond+addr | PASS | No (uses `-5`) |

T4 is especially revealing: `536870907` in the source without a conditional branch **PASSES**. The bug requires both ingredients: the fused constant `536870907` AND a conditional branch containing the address computation.

---

### Phase 8: The Final Reproducers

Three standalone reproducers were created for the Apple bug report:

1. **`metalbug_reproducer.m`** — Objective-C, self-contained, compiles with `clang -framework Metal -framework Foundation`
2. **`metalbug_reproducer.swift`** + **`metalbug_cpu.cpp`** — Swift+C++ interop, also runs the same logic on CPU to prove the algorithm is correct and the GPU backend is at fault
3. **`metalbug_reproducer.metal`** — Clean two-kernel Metal source (buggy vs working)

And the formal write-up: **`metalbug_apple_report.md`**.

---

## Root Cause Summary

The Metal GPU backend (the AGX code generator that runs **after** LLVM IR → AIR optimization) miscompiles the following IR pattern when it appears inside a conditional branch:

```llvm
; From reproducer_O2.ll @buggy, lines 39-44 (inside label %25, the true-branch)
%34 = mul i32 %33, 536870907            ; quotient * 536870907 (overflows i32)
%35 = add i32 %34, %28                  ; + alu1
%36 = shl i32 %35, 3                    ; << 3
%37 = sext i32 %36 to i64               ; sign-extend for 64-bit address
%38 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %37
%39 = load i8, i8 addrspace(1)* %38     ; ← returns 0 instead of correct value
```

The load returns **0** instead of the correct value for inputs where `%33` (the quotient) is > 0, causing `%34` to overflow i32. This happens for `alu0 ∈ {5, 10, 15, 20}` (quotient = 1, 2, 3, 4). The thread with `alu0 = 0` (quotient = 0) passes because `0 * 536870907 = 0` — no overflow.

The equivalent safe pattern, produced when the remainder is computed pre-branch:

```llvm
; From reproducer_O2.ll @working, lines 72-73 (BEFORE the branch)
%20 = mul i32 %19, -5                   ; quotient * (-5) — no problematic overflow
%21 = add i32 %20, %12                  ; + alu1 = remainder (small value 0-4)
```

Both patterns are algebraically identical in 32-bit modular arithmetic. The LLVM IR is provably correct in both cases. The bug is in the GPU backend's code generation for the first pattern.

## Workaround

Ensure the fast-division remainder is used for **both** the branch condition and the load address. This forces the LLVM optimizer to compute the remainder before the branch using `mul i32 X, -5` (no fusion with the shift), which the GPU backend handles correctly.

## Reproducing the LLVM IR

### Prerequisites

Install the Metal Toolchain (required for `xcrun metal` with `-S -emit-llvm`):

```bash
xcodebuild -downloadComponent MetalToolchain
```

### Compiling Metal to human-readable LLVM IR

Each `.metal` file is compiled at two optimization levels. The `-S -emit-llvm` flags tell the Metal compiler to emit LLVM IR as text instead of binary AIR bitcode.

```bash
# From the tinygrad root directory
mkdir -p test/air-llvm

# Compile all kernels in test/metalbug_ir/ at -O0 and -O2
for f in test/metalbug_ir/*.metal; do
  name=$(basename "$f" .metal)
  xcrun metal -O0 -fno-fast-math -std=metal3.1 -S -emit-llvm -o "test/air-llvm/${name}_O0.ll" "$f"
  xcrun metal -O2 -fno-fast-math -std=metal3.1 -S -emit-llvm -o "test/air-llvm/${name}_O2.ll" "$f"
done

# Compile the standalone reproducer (in test/, not test/metalbug_ir/)
xcrun metal -O0 -fno-fast-math -std=metal3.1 -S -emit-llvm -o test/air-llvm/reproducer_O0.ll test/metalbug_reproducer.metal
xcrun metal -O2 -fno-fast-math -std=metal3.1 -S -emit-llvm -o test/air-llvm/reproducer_O2.ll test/metalbug_reproducer.metal
```

### Compiling Metal to binary AIR (intermediate step)

If you want the binary AIR bitcode files (e.g. to link into a `.metallib` or inspect with `metal-objdump`):

```bash
# Compile to .air (binary LLVM bitcode in Apple's AIR format)
xcrun metal -O2 -fno-fast-math -std=metal3.1 -c -o test/metalbug_ir/bug.air test/metalbug_ir/bug.metal

# Link into a .metallib (Metal library archive)
xcrun metallib -o test/metalbug_ir/bug.metallib test/metalbug_ir/bug.air

# Disassemble the .metallib to see AIR metadata
xcrun metal-objdump -d test/metalbug_ir/bug.metallib
```

### Building the standalone reproducer binary

```bash
cd test
swiftc -g -cxx-interoperability-mode=default \
  -import-objc-header metalbug_cpu.h \
  metalbug_cpu.cpp metalbug_reproducer.swift \
  -framework Metal -framework Foundation \
  -o metalbug_reproducer
./metalbug_reproducer
```

### Quick verification: grep for the fused constant

```bash
# Should appear in all FAIL kernels, absent from all PASS kernels
grep -l '536870907' test/air-llvm/*_O2.ll
```

---

## File Index

| File | Role |
|------|------|
| `test/null/test_uops.py:164` | The commented-out skip that started it all |
| `test/metalbug.py` | Initial scratch: keccak impl, early modulo/division tests |
| `test/metalbug_minimal.py` | Phase 1: tensor-level SHA3 tests, 8-byte vs 9-byte boundary |
| `test/metalbug_e25n3.py` | Phase 2: exact generated E\_25n3 kernel reproduction |
| `test/metalbug_paredown.py` | Phase 3: systematic kernel reduction (Versions A-F) |
| `test/metalbug_minimal_repro.py` | Phase 4: hypothesis tests (H1-H7) on minimal kernel |
| `test/metalbug_deep.py` | Phase 5: deep diagnosis (H8-H11), pinpointing the load |
| `test/metalbug_ir.py` | Phase 6: LLVM IR compilation and comparison |
| `test/metalbug_asm.py` | Phase 6: broader Metal toolchain exploration |
| `test/metalbug_constant.py` | Phase 7: targeted constant tests (T1-T6) |
| `test/metalbug_reproducer.metal` | Final: clean two-kernel Metal source |
| `test/metalbug_reproducer.m` | Final: Objective-C standalone reproducer |
| `test/metalbug_reproducer.swift` | Final: Swift standalone reproducer |
| `test/metalbug_cpu.cpp` / `.h` | Final: CPU reference implementation proving algorithm correctness |
| `test/metalbug_apple_report.md` | Final: formal bug report for Apple |
| `test/air-llvm/*.ll` | Human-readable LLVM IR from all Metal kernels at -O0 and -O2 |


# Metal GPU Compiler Bug: Incorrect Code Generation for `mul i32 X, 536870907` Inside Conditional Branch

## Summary

The Metal GPU compiler (the runtime JIT that converts AIR to GPU machine code) produces incorrect results for a specific LLVM IR pattern: when the LLVM optimizer fuses a multiply-by-(-5) with a shift-left-by-3 into multiply-by-536870907 then shift-left-by-3, and this computation occurs inside a conditional branch used as a buffer load address, the load returns 0 instead of the correct value for certain inputs.

The LLVM IR produced by the Metal frontend is provably correct. The bug is in the GPU backend.

## Environment

- Apple M3 Pro, macOS 15.x
- Metal compiler: `metalfe-32023.622`
- Metal language: Metal 3.1
- Xcode toolchain: XcodeDefault

## Minimal Reproducer

Two Metal kernels perform mathematically equivalent computations but produce different results. Both kernels:

1. Read an index value from a buffer
2. Compute `(index + 1) % 5` using the standard multiply-shift fast division pattern (`x * 1717986919 >> 33`)
3. Use the remainder as a byte offset (`remainder * 8`) to conditionally load from a data buffer

The ONLY difference: in the **buggy** kernel, the condition and the address use **different intermediate variables** (causing the LLVM optimizer to move the address computation inside the branch and fuse it with the shift, producing the constant 536870907). In the **working** kernel, the condition and address use the **same intermediate** (so the optimizer computes the remainder before the branch and keeps it as a standalone value, using the constant -5).

### Kernel A: BUGGY (produces wrong results)

```metal
#include <metal_stdlib>
using namespace metal;
kernel void buggy(device int* output, device int* indices, device unsigned char* data,
                  uint3 gid [[threadgroup_position_in_grid]]) {
  int gidx0 = gid.x;
  int val0 = indices[gidx0];
  int alu0 = (val0 < 0) ? (val0 + 25) : val0;
  int alu1 = alu0 + 1;

  // Condition: compute (alu0+1) % 5 via LONG arithmetic (native modulo)
  long alu2 = (long)(alu0) + 1ll;
  int cast1 = (int)(alu2 % 5ll);
  bool cond = (cast1 < 2) & (alu0 >= 0) & (alu0 < 25);

  // Address: compute (alu1) % 5 via INT fast-division (multiply-shift)
  //   This is mathematically identical to cast1 since alu1 == (int)alu2
  int q = (int)(((long)(alu1) * 1717986919ll) >> 33ll);
  int alu4 = (alu1 < 0) ? 1 : 0;
  //   remainder = alu1 - 5 * (q + alu4), then address = remainder * 8
  //   The optimizer fuses "- 5 * X" with "<< 3" into "X * 536870907 ... << 3"
  unsigned char val1 = cond
    ? *(data + ((alu1 - 5 * (q + alu4)) << 3))
    : (unsigned char)(0u);

  output[gidx0] = (int)val1;
}
```

### Kernel B: WORKING (produces correct results)

```metal
#include <metal_stdlib>
using namespace metal;
kernel void working(device int* output, device int* indices, device unsigned char* data,
                    uint3 gid [[threadgroup_position_in_grid]]) {
  int gidx0 = gid.x;
  int val0 = indices[gidx0];
  int alu0 = (val0 < 0) ? (val0 + 25) : val0;
  int alu1 = alu0 + 1;

  // Compute (alu1) % 5 via INT fast-division (same formula as Kernel A's address)
  int q = (int)(((long)(alu1) * 1717986919ll) >> 33ll);
  int alu4 = (alu1 < 0) ? 1 : 0;
  int remainder = alu1 - 5 * (q + alu4);

  // Use the SAME remainder for BOTH condition and address
  bool cond = (remainder < 2) & (alu0 >= 0) & (alu0 < 25);
  unsigned char val1 = cond
    ? *(data + (remainder << 3))
    : (unsigned char)(0u);

  output[gidx0] = (int)val1;
}
```

### Why these kernels are equivalent

Both kernels compute `(alu0 + 1) % 5` and use it the same way:
- **Condition**: `remainder < 2` (and bounds checks on alu0)
- **Address**: `remainder * 8` (i.e., `remainder << 3`)

In Kernel A, the condition uses `cast1 = (int)((long)(alu0) + 1) % 5)` and the address uses `alu1 - 5 * quotient` where `alu1 = alu0 + 1`. Since `(long)(alu0) + 1 == (long)(alu0 + 1) == (long)(alu1)`, both `cast1` and the address remainder compute the same value: `(alu0 + 1) % 5`.

In Kernel B, a single `remainder` variable is used for both.

The mathematical equivalence is straightforward and can be verified for all 25 input values (alu0 = 0 through 24).

### Test setup

- `indices` buffer: 25 ints containing a permutation of 0-24: `[0, 6, 12, 18, 24, 3, 9, 10, 16, 22, 1, 7, 13, 19, 20, 4, 5, 11, 17, 23, 2, 8, 14, 15, 21]`
- `data` buffer: 9 bytes containing `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- Launch 25 threads (one per index)

### Expected results (both kernels)

For each thread, `(alu0 + 1) % 5` determines whether to load:
- If remainder < 2 (i.e., remainder is 0 or 1): load `data[remainder * 8]`
  - remainder 0 → `data[0]` = 0
  - remainder 1 → `data[8]` = 8
- Otherwise: output 0

Threads where `alu0 ∈ {0, 5, 10, 15, 20}` have remainder = 1, so they should output **8**.

### Actual results

| Thread (gidx0) | alu0 | Expected | Kernel B (working) | Kernel A (buggy) |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 8 | 8 | 8 |
| 7 | 10 | 8 | 8 | **0** |
| 14 | 20 | 8 | 8 | **0** |
| 16 | 5 | 8 | 8 | **0** |
| 23 | 15 | 8 | 8 | **0** |

Kernel A returns 0 (the else-branch value) for 4 out of 5 threads where remainder = 1. The thread with alu0 = 0 passes; the threads with alu0 = 5, 10, 15, 20 fail. All failing threads have fast-division quotient > 0.

## Root Cause Analysis (LLVM IR)

Compiling both kernels with `xcrun metal -O2 -S -emit-llvm` reveals the cause. The LLVM IR is correct for both kernels, but they have different structure due to valid optimizer transformations.

### Kernel B (working) — IR structure

The optimizer computes the remainder **before** the branch because it's needed for both the condition and the address:

```llvm
; Before branch: compute remainder using mul i32 %quotient, -5
%20 = mul i32 %19, -5             ; quotient * (-5)
%21 = add i32 %20, %12            ; alu1 + (-5)*quotient = remainder
%22 = icmp slt i32 %21, 2         ; condition: remainder < 2
...
br i1 %30, label %37, label %31   ; branch

31:                                 ; true branch: just shift + load
  %32 = shl i32 %21, 3            ; remainder << 3
  ...load...
```

### Kernel A (buggy) — IR structure

The optimizer computes the remainder **inside** the branch (since it's only needed for the address, not the condition), and fuses `*(-5)` with `<<3`:

```llvm
; Before branch: compute condition using srem
%14 = srem i64 %13, 5              ; native modulo for condition
...
br i1 %24, label %41, label %25    ; branch

25:                                  ; true branch: full recomputation
  %28 = mul nsw i64 %27, 1717986919 ; fast-div multiply
  %29 = lshr i64 %28, 33            ; shift for quotient
  ...
  %34 = mul i32 %33, 536870907      ; FUSED: quotient * 536870907
  %35 = add i32 %34, %26            ; + alu1
  %36 = shl i32 %35, 3              ; << 3
  %37 = sext i32 %36 to i64         ; sign-extend for address
  ...load...
```

### The fused constant 536870907

The optimizer transforms `(alu1 - 5*q) << 3` into `(q * 536870907 + alu1) << 3`. This is valid because `536870907 * 8 ≡ -40 (mod 2^32)` and `(-5) * 8 = -40`, so after the shift, both expressions produce the same 32-bit result. The constant 536870907 is the unique value in [0, 2^29) satisfying this equivalence.

This transformation is algebraically correct — the intermediate i32 value overflows but the final result after the shift and truncation is identical. **The LLVM IR is valid.**

### Correlation is perfect

Across 10+ test variants, **every kernel whose IR contains `mul i32 X, 536870907` fails, and every kernel without it passes**. The correlation is 100%.

## Conclusion

The Metal GPU compiler's runtime JIT (the stage that converts AIR/LLVM bitcode to AGX machine instructions) miscompiles the pattern:

```llvm
%a = mul i32 %x, 536870907    ; inside conditional branch
%b = add i32 %a, %y
%c = shl i32 %b, 3
%d = sext i32 %c to i64
%e = getelementptr inbounds i8, i8 addrspace(1)* %ptr, i64 %d
%f = load i8, i8 addrspace(1)* %e
```

The load returns 0 instead of the correct value for inputs where the intermediate `%a` causes i32 overflow (quotient > 0). The LLVM IR is provably correct; the bug is in the backend code generation.

## Workaround

Ensure the fast-division remainder is used for both the branch condition and the load address. This forces the LLVM optimizer to compute the remainder before the branch using `mul i32 X, -5` (no fusion with the shift), which the GPU backend handles correctly.

## Attachments

- `buggy.metal` — Kernel A source (reproduces the bug)
- `working.metal` — Kernel B source (correct behavior)
- `reproducer.m` — Self-contained Objective-C command-line program that compiles both kernels, runs them, and shows the discrepancy
