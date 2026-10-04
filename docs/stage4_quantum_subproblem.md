# Stage 4 — Quantum Linear-System Design

## 1. Objective

The Quantum Newton's Method solves a nonlinear system

F(x) = 0

by repeatedly solving a linear system inside each Newton iteration.

The Newton correction satisfies

J(x_k) Delta x_k = -F(x_k)

where J(x_k) is the Jacobian.

---

## 2. Fixed Newton Iteration

For the selected test problem,

x_0 = [2, 1]^T.

At this point,

F(x_0) = [1, 0]^T

and

J(x_0) =
[[4, 2],
 [1, -1]].

Therefore the linear system for the first Newton correction is

[[4, 2],
 [1, -1]]
[Delta x_1,
 Delta x_2]^T

=
[-1, 0]^T.

The exact classical solution is

Delta x =
[-1/6, -1/6]^T.

---

## 3. Matrix Normalization

Following the QNM formulation,

A = J / ||J||_max

where

||J||_max = max_ij |J_ij|.

For the selected Jacobian,

||J||_max = 4.

Therefore,

A =
[[1, 0.5],
 [0.25, -0.25]].

The scaling factor is

4.

---

## 4. RHS Quantum State

The RHS vector is

b = [-1, 0]^T.

Its L2 norm is

||b||_2 = 1.

Therefore the normalized quantum state is

|b> = [-1, 0]^T.

Since the vector dimension is 2,

ceil(log2(2)) = 1

qubit is required for amplitude encoding.

---

## 5. Solution State

The classical solution is

Delta x = [-1/6, -1/6]^T.

Its L2 norm is

||Delta x||_2 = 1/sqrt(18).

The normalized solution state is therefore

|Delta x> =
[-1/sqrt(2), -1/sqrt(2)]^T.

The quantum routine aims to recover this normalized state.

---

## 6. PennyLane Realization

The project uses PennyLane's default.qubit simulator.

The normalized RHS and solution are represented using
amplitude encoding.

The finite-size implementation prepares and measures the
solution state for the selected 2-dimensional system.

This is a simulator-level engineering approximation.

The implementation does NOT claim to reproduce the complete
QRAM, oracle, and asymptotic Childs-Kothari-Somma QLSS
hardware architecture of the original paper.

---

## 7. Classical Reference

The exact classical solution is retained as the ground-truth
reference.

The quantum result is compared using

||Delta x_quantum - Delta x_classical||_2.

The normalized-state error is also recorded.

---

## 8. Measurement

Two simulator modes are used:

1. Exact statevector simulation (`shots=None`)
2. Finite-shot measurement simulation (`shots=1000`)

Finite-shot measurements allow sampling error to be observed.

Full classical recovery/tomography is reserved for Stage 6.

---

## 9. Implementation Boundary

The original QNM paper uses:

Classical Newton step
        ↓
Normalized linear system
        ↓
Quantum linear-system solver
        ↓
Normalized solution state
        ↓
l-infinity tomography
        ↓
Classical correction
        ↓
Newton update

The present Stage 5 implementation covers the finite-size
quantum linear-system prototype.

Tomography and full Newton integration are implemented later.