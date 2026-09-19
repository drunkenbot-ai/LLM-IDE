"""Frontier STEM & Formal Mathematics Base Pretraining Dataset Generator.

Generates high-density, LaTeX-rich scientific treatises, mathematical proofs,
and technical derivations formatted for base autoregressive pre-training.

Uses raw string blocks (r\"\"\"...\"\"\") to preserve 100% LaTeX syntax fidelity
without f-string escaping conflicts.

Disciplines:
1. Advanced Calculus & Real Analysis (Stokes' theorem, epsilon-delta proofs, Bolzano-Weierstrass)
2. Linear Algebra & Matrix Theory (Spectral theorem, SVD, Jordan form, Rayleigh quotient)
3. Probability & Stochastic Calculus (Measure theory, Central Limit Theorem, Ito's lemma)
4. Abstract Algebra & Number Theory (Sylow theorems, Elliptic curves, Galois theory)
5. Physics & Differential Equations (Lagrangian mechanics, Maxwell tensors, Schrodinger equation)
6. Olympiad & Competition Math (Generating functions, Cauchy-Schwarz, AM-GM, Recurrences)
7. Complex Analysis (Residue theorem, Laurent series, Contour integrals)

Enforces strict partition sizing: files are partitioned to stay <= 28.0 MB (29,360,128 bytes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MAX_PARTITION_BYTES = 28 * 1024 * 1024  # 28 MB ceiling


# ==============================================================================
# Partitioned Writer
# ==============================================================================

class ShardedSTEMWriter:
    """Writes JSONL base pretraining records into partitioned files under 28 MB."""

    def __init__(self, output_dir: Path, prefix: str = "base_stem_part", max_bytes: int = MAX_PARTITION_BYTES) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = 1
        self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
        self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
        self.cur_bytes = 0
        self.cur_records = 0
        self.total_records = 0
        self.total_bytes = 0
        self.written_files: List[Path] = [self.cur_file]

    def write_document(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        doc = {
            "text": text,
            "meta": meta or {}
        }
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        b_len = len(line.encode("utf-8"))

        if self.cur_bytes + b_len > self.max_bytes and self.cur_records > 0:
            self.cur_fp.close()
            mb = self.cur_bytes / (1024 * 1024)
            print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            self.part_idx += 1
            self.cur_file = self.output_dir / f"{self.prefix}_{self.part_idx:03d}.jsonl"
            self.cur_fp = open(self.cur_file, "w", encoding="utf-8")
            self.written_files.append(self.cur_file)
            self.cur_bytes = 0
            self.cur_records = 0

        self.cur_fp.write(line)
        self.cur_bytes += b_len
        self.cur_records += 1
        self.total_records += 1
        self.total_bytes += b_len

    def close(self) -> None:
        if not self.cur_fp.closed:
            self.cur_fp.close()
            if self.cur_records > 0:
                mb = self.cur_bytes / (1024 * 1024)
                print(f"  Saved {self.cur_file.name}: {self.cur_records:,} records ({mb:.2f} MB)")
            elif len(self.written_files) > 1 and self.cur_file.exists() and self.cur_file.stat().st_size == 0:
                self.cur_file.unlink()
                self.written_files.remove(self.cur_file)


# ==============================================================================
# Helper Estimators
# ==============================================================================

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def calculate_latex_density(text: str) -> float:
    latex_tokens = ["$", "\\frac", "\\int", "\\sum", "\\prod", "\\mathbf", "\\mathbb", "\\lim", "\\partial", "\\nabla"]
    count = sum(text.count(t) for t in latex_tokens)
    words = max(1, len(text.split()))
    return round(count / words, 4)


# ==============================================================================
# Domain 1: Advanced Calculus & Real Analysis
# ==============================================================================

def gen_multivariable_calculus_stokes(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    manifold = rng.choice(["3-dimensional Euclidean space $\\mathbb{R}^3$", "smooth orientable Riemannian manifold $(\\mathcal{M}, g)$"])
    template = r"""# Advanced Differential Geometry: The Generalized Stokes' Theorem
## Formulation on __MANIFOLD__

### 1. Exterior Calculus & Differential Forms
Let $\mathcal{M}$ be an oriented, smooth $n$-dimensional manifold with boundary $\partial\mathcal{M}$, endowed with the induced orientation. Let $\Omega^k(\mathcal{M})$ denote the vector space of smooth differential $k$-forms on $\mathcal{M}$.

The exterior derivative operator $d: \Omega^k(\mathcal{M}) \to \Omega^{k+1}(\mathcal{M})$ is the unique $\mathbb{R}$-linear map satisfying:
1. For any $f \in \Omega^0(\mathcal{M})$, $df$ is the differential of $f$:
   $$df = \sum_{i=1}^n \frac{\partial f}{\partial x^i} dx^i$$
2. For $\alpha \in \Omega^k(\mathcal{M})$ and $\beta \in \Omega^l(\mathcal{M})$:
   $$d(\alpha \wedge \beta) = d\alpha \wedge \beta + (-1)^k \alpha \wedge d\beta$$
3. Nilpotency: $d^2 = d \circ d = 0$.

### 2. Theorem Statement (Generalized Stokes' Theorem)
Let $\omega \in \Omega^{n-1}(\mathcal{M})$ be a compactly supported differential $(n-1)$-form of class $\mathcal{C}^1$ on $\mathcal{M}$. Then:
$$\int_{\mathcal{M}} d\omega = \int_{\partial\mathcal{M}} \omega$$
where the integral on the right-hand side is evaluated with respect to the induced boundary orientation.

### 3. Rigorous Proof via Partition of Unity
**Step 1: Localization via Partition of Unity.**
Because $\text{supp}(\omega)$ is compact, there exists a finite open cover $\{U_i\}_{i=1}^N$ of $\text{supp}(\omega)$ consisting of coordinate charts $(\phi_i, U_i)$ and a subordinate smooth partition of unity $\{\rho_i\}_{i=1}^N$ such that $\sum_{i=1}^N \rho_i(x) = 1$ for all $x \in \text{supp}(\omega)$.

By linearity of integration and the exterior derivative:
$$\omega = \sum_{i=1}^N (\rho_i \omega) \implies d\omega = \sum_{i=1}^N d(\rho_i \omega)$$
Hence, it suffices to prove the theorem for a differential form $\eta = \rho_i \omega$ whose support lies entirely within a single coordinate chart $U$.

**Step 2: Interior Charts ($U \cap \partial\mathcal{M} = \emptyset$).**
When $U$ does not intersect the boundary, $\phi(U) \subset \mathbb{R}^n$. Write $\eta$ in coordinates:
$$\eta = \sum_{j=1}^n (-1)^{j-1} a_j(x^1, \dots, x^n) dx^1 \wedge \dots \wedge \widehat{dx^j} \wedge \dots \wedge dx^n$$
Taking the exterior derivative yields:
$$d\eta = \sum_{j=1}^n \frac{\partial a_j}{\partial x^j} dx^1 \wedge \dots \wedge dx^n$$
Integrating over the entire half-space $\mathbb{R}^n$:
$$\int_U d\eta = \sum_{j=1}^n \int_{\mathbb{R}^n} \frac{\partial a_j}{\partial x^j} dx^1 \dots dx^n$$
By Fubini's theorem and the Fundamental Theorem of Calculus along the $j$-th coordinate axis:
$$\int_{\mathbb{R}} \frac{\partial a_j}{\partial x^j} dx^j = \lim_{x^j \to \infty} a_j - \lim_{x^j \to -\infty} a_j = 0 - 0 = 0$$
Because $\eta$ has compact support inside $U$. Since $\partial U = \emptyset$, $\int_{\partial U} \eta = 0$. Thus $\int_U d\eta = \int_{\partial U} \eta = 0$.

**Step 3: Boundary Charts ($U \cap \partial\mathcal{M} \neq \emptyset$).**
When $U$ intersects the boundary, $\phi(U) \subset \mathbb{H}^n = \{ (x^1, \dots, x^n) \in \mathbb{R}^n : x^n \ge 0 \}$.
The boundary $\partial\mathbb{H}^n$ is defined by $x^n = 0$, with induced orientation given by $(-1)^n dx^1 \wedge \dots \wedge dx^{n-1}$.
For $j < n$, the integration along $x^j$ vanishes as before. For $j = n$:
$$\int_{\mathbb{H}^n} \frac{\partial a_n}{\partial x^n} dx^1 \dots dx^n = \int_{\mathbb{R}^{n-1}} \left( \int_0^\infty \frac{\partial a_n}{\partial x^n} dx^n \right) dx^1 \dots dx^{n-1} = -\int_{\mathbb{R}^{n-1}} a_n(x^1, \dots, x^{n-1}, 0) dx^1 \dots dx^{n-1}$$
Accounting for the orientation sign convention $(-1)^n$, this matches precisely the integral of $\eta$ pulled back to $\partial\mathcal{M}$.

Summing over all chart partitions completes the proof:
$$\int_{\mathcal{M}} d\omega = \int_{\partial\mathcal{M}} \omega \quad \blacksquare$$
"""
    text = template.replace("__MANIFOLD__", manifold)
    meta = {
        "domain": "stem",
        "subdomain": "advanced_calculus",
        "topic": "generalized_stokes_theorem",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


def gen_real_analysis_bolzano_weierstrass(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    metric_space = rng.choice(["Euclidean space $\\mathbb{R}^k$", "complete normed metric space $(X, \\|\\cdot\\|)$"])
    template = r"""# Real Analysis & Metric Topology: The Bolzano-Weierstrass Theorem
## Sequence Compactness in __SPACE__

### 1. Foundational Definitions
Let $(X, d)$ be a metric space.
- A sequence $(x_n)_{n=1}^\infty \subset X$ is said to be **bounded** if there exists $M > 0$ and $x_0 \in X$ such that $d(x_n, x_0) \le M$ for all $n \in \mathbb{N}$.
- A subsequence $(x_{n_k})_{k=1}^\infty$ is defined by a strictly increasing sequence of indices $n_1 < n_2 < n_3 < \dots$
- A set $K \subset X$ is **sequentially compact** if every sequence in $K$ admits a subsequence that converges to a point in $K$.

### 2. Theorem (Bolzano-Weierstrass in $\mathbb{R}$)
Every bounded sequence of real numbers $(x_n)_{n=1}^\infty$ contains a convergent subsequence.

### 3. Lemma (Nested Interval Property / Cantor's Intersection Theorem)
Let $([a_n, b_n])_{n=1}^\infty$ be a sequence of closed, bounded intervals in $\mathbb{R}$ such that:
1. $[a_{n+1}, b_{n+1}] \subseteq [a_n, b_n]$ for all $n \in \mathbb{N}$ (nested property).
2. $\lim_{n \to \infty} (b_n - a_n) = 0$.

Then the intersection $\bigcap_{n=1}^\infty [a_n, b_n]$ consists of exactly one point $L \in \mathbb{R}$.

**Proof of Lemma:**
The sequence $(a_n)$ is non-decreasing and bounded above by $b_1$. By the Completeness Axiom of $\mathbb{R}$ (Monotone Convergence Theorem), $L = \sup_{n} a_n = \lim_{n \to \infty} a_n$ exists. Similarly, $(b_n)$ is non-increasing and bounded below, converging to $R = \inf_{n} b_n$.
Since $a_n \le b_n$ for all $n$:
$$0 \le R - L \le b_n - a_n \to 0 \implies R = L$$
Thus $\bigcap_{n=1}^\infty [a_n, b_n] = \{ L \}$. $\blacksquare$

### 4. Rigorous Proof of Bolzano-Weierstrass via Bisection
Let $(x_n)_{n=1}^\infty$ be bounded. There exists $a_1, b_1 \in \mathbb{R}$ such that $a_1 \le x_n \le b_1$ for all $n$.
Define $I_1 = [a_1, b_1]$ and set $n_1 = 1$. The interval $I_1$ contains infinitely many terms of the sequence.

Divide $I_1$ into two sub-intervals of equal length at the midpoint $m_1 = \frac{a_1 + b_1}{2}$:
$$I_1^{\text{left}} = \left[ a_1, \frac{a_1 + b_1}{2} \right], \quad I_1^{\text{right}} = \left[ \frac{a_1 + b_1}{2}, b_1 \right]$$
By the Pigeonhole Principle for infinite sets, at least one of these two sub-intervals contains infinitely many terms of $(x_n)$.
Choose $I_2 = [a_2, b_2]$ to be the sub-interval with infinitely many terms (if both have infinitely many, choose the left).
Select $n_2 > n_1$ such that $x_{n_2} \in I_2$.

Proceeding inductively, at step $k$:
1. $I_k = [a_k, b_k]$ contains infinitely many terms.
2. The interval length satisfies:
   $$b_k - a_k = \frac{b_1 - a_1}{2^{k-1}}$$
3. We bisect $I_k$, select $I_{k+1} \subset I_k$ containing infinitely many terms, and pick an index $n_{k+1} > n_k$ with $x_{n_{k+1}} \in I_{k+1}$.

By the Nested Interval Property, there is a unique $L \in \mathbb{R}$ such that:
$$\bigcap_{k=1}^\infty I_k = \{ L \}$$

We now verify that $\lim_{k \to \infty} x_{n_k} = L$.
Let $\epsilon > 0$. Since $\frac{b_1 - a_1}{2^{k-1}} \to 0$ as $k \to \infty$, choose $K \in \mathbb{N}$ such that:
$$\frac{b_1 - a_1}{2^{K-1}} < \epsilon$$
For all $k \ge K$, both $x_{n_k} \in I_k \subseteq I_K$ and $L \in I_K$. Hence:
$$|x_{n_k} - L| \le b_K - a_K < \epsilon$$
This establishes that $x_{n_k} \to L$ as $k \to \infty$. Therefore, $(x_n)$ contains a convergent subsequence. $\blacksquare$
"""
    text = template.replace("__SPACE__", metric_space)
    meta = {
        "domain": "stem",
        "subdomain": "real_analysis",
        "topic": "bolzano_weierstrass_theorem",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 2: Linear Algebra & Matrix Theory
# ==============================================================================

def gen_linear_algebra_svd_spectral(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    matrix_dim = rng.choice(["m \\times n", "n \\times n", "p \\times q"])
    template = r"""# Linear Algebra: Singular Value Decomposition (SVD) and the Spectral Theorem
## Rigorous Matrix Factorization for $\mathbf{A} \in \mathbb{R}^{__DIM__}$

### 1. The Spectral Theorem for Symmetric Matrices
Let $\mathbf{S} \in \mathbb{R}^{n \times n}$ be a real symmetric matrix ($\mathbf{S}^T = \mathbf{S}$).

**Theorem (Spectral Decomposition):**
There exists an orthonormal basis of $\mathbb{R}^n$ consisting of eigenvectors of $\mathbf{S}$, and all eigenvalues of $\mathbf{S}$ are real. That is:
$$\mathbf{S} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$
where $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_n] \in \mathbb{R}^{n \times n}$ is an orthogonal matrix ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$) and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_n)$.

**Proof:**
Consider the Rayleigh quotient $R(\mathbf{x}) = \frac{\mathbf{x}^T \mathbf{S} \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$ restricted to the compact unit sphere $\mathcal{S}^{n-1} = \{ \mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\|_2 = 1 \}$.
Since $\mathcal{S}^{n-1}$ is compact and $\mathbf{x} \mapsto \mathbf{x}^T \mathbf{S} \mathbf{x}$ is continuous, by the Extreme Value Theorem, there exists $\mathbf{q}_1 \in \mathcal{S}^{n-1}$ maximizing $R(\mathbf{x})$:
$$\lambda_1 = \max_{\|\mathbf{x}\| = 1} \mathbf{x}^T \mathbf{S} \mathbf{x} = \mathbf{q}_1^T \mathbf{S} \mathbf{q}_1$$
Using the method of Lagrange multipliers with constraint $g(\mathbf{x}) = \mathbf{x}^T \mathbf{x} - 1 = 0$:
$$\nabla (\mathbf{x}^T \mathbf{S} \mathbf{x} - \lambda (\mathbf{x}^T \mathbf{x} - 1)) = 2\mathbf{S}\mathbf{x} - 2\lambda \mathbf{x} = \mathbf{0} \implies \mathbf{S}\mathbf{q}_1 = \lambda_1 \mathbf{q}_1$$
Thus $\mathbf{q}_1$ is an eigenvector with eigenvalue $\lambda_1$.
Let $V_1^\perp = \{ \mathbf{y} \in \mathbb{R}^n : \mathbf{q}_1^T \mathbf{y} = 0 \}$. For any $\mathbf{y} \in V_1^\perp$:
$$\mathbf{q}_1^T (\mathbf{S}\mathbf{y}) = (\mathbf{S}\mathbf{q}_1)^T \mathbf{y} = \lambda_1 \mathbf{q}_1^T \mathbf{y} = 0$$
Hence $\mathbf{S}$ leaves $V_1^\perp$ invariant. By induction on dimension $n$, $\mathbf{S}$ is fully diagonalizable by an orthogonal matrix. $\blacksquare$

### 2. Singular Value Decomposition (SVD)
Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ with $\text{rank}(\mathbf{A}) = r \le \min(m, n)$.

**Theorem (SVD Existence):**
There exist orthogonal matrices $\mathbf{U} \in \mathbb{R}^{m \times m}$ and $\mathbf{V} \in \mathbb{R}^{n \times n}$, and a diagonal matrix $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ such that:
$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
where the diagonal entries $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > \sigma_{r+1} = \dots = 0$ are the singular values of $\mathbf{A}$.

**Constructive Proof:**
1. Consider the $n \times n$ matrix $\mathbf{A}^T \mathbf{A}$. It is symmetric:
   $$(\mathbf{A}^T \mathbf{A})^T = \mathbf{A}^T (\mathbf{A}^T)^T = \mathbf{A}^T \mathbf{A}$$
   and positive semi-definite, since for all $\mathbf{x} \in \mathbb{R}^n$:
   $$\mathbf{x}^T (\mathbf{A}^T \mathbf{A}) \mathbf{x} = (\mathbf{A}\mathbf{x})^T (\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \ge 0$$
2. By the Spectral Theorem, $\mathbf{A}^T \mathbf{A}$ has non-negative real eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$ and orthonormal eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_n \in \mathbb{R}^n$.
   Define the singular values:
   $$\sigma_i = \sqrt{\lambda_i} \ge 0, \quad \forall i \in \{1, \dots, n\}$$
3. For $1 \le i \le r$ (where $\sigma_i > 0$), define:
   $$\mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{A} \mathbf{v}_i \in \mathbb{R}^{m}$$
   Check orthonormality of $\{\mathbf{u}_i\}_{i=1}^r$:
   $$\mathbf{u}_i^T \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T \mathbf{A}^T \mathbf{A} \mathbf{v}_j = \frac{\lambda_j}{\sigma_i \sigma_j} \mathbf{v}_i^T \mathbf{v}_j = \frac{\sigma_j^2}{\sigma_i \sigma_j} \delta_{ij} = \delta_{ij}$$
4. Extend $\{\mathbf{u}_1, \dots, \mathbf{u}_r\}$ via the Gram-Schmidt process to an orthonormal basis $\{\mathbf{u}_1, \dots, \mathbf{u}_m\}$ of $\mathbb{R}^m$.
   Assemble $\mathbf{U} = [\mathbf{u}_1, \dots, \mathbf{u}_m]$ and $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_n]$.
   Then:
   $$\mathbf{U}^T \mathbf{A} \mathbf{V} = \mathbf{\Sigma} \implies \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \quad \blacksquare$$

### 3. Geometric Interpretation & Eckart-Young-Mirsky Theorem
The SVD provides the optimal low-rank matrix approximation under both Frobenius and Spectral norms:
$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \arg\min_{\text{rank}(\mathbf{B}) \le k} \|\mathbf{A} - \mathbf{B}\|_F$$
where the approximation error is given by:
$$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2, \quad \|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$$
"""
    text = template.replace("__DIM__", matrix_dim)
    meta = {
        "domain": "stem",
        "subdomain": "linear_algebra",
        "topic": "svd_and_spectral_theorem",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 3: Probability Theory & Stochastic Calculus
# ==============================================================================

def gen_stochastic_calculus_ito_black_scholes(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    drift = rng.choice(["\\mu", "r"])
    vol = rng.choice(["\\sigma", "\\nu"])
    template = r"""# Stochastic Calculus: Itô's Lemma and the Derivation of the Black-Scholes PDE
## Mathematical Foundations of Continuous-Time Martingales

### 1. Brownian Motion and the Itô Integral
Let $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \ge 0}, \mathbb{P})$ be a filtered probability space satisfying the usual conditions.
A standard one-dimensional Brownian motion $(W_t)_{t \ge 0}$ is an adapted continuous process with:
1. $W_0 = 0$ almost surely.
2. Independent increments: For any $0 \le s < t$, $W_t - W_s \perp \mathcal{F}_s$.
3. Stationary Gaussian increments: $W_t - W_s \sim \mathcal{N}(0, t - s)$.

For a progressive process $X_t \in \mathcal{L}^2([0, T])$, the Itô stochastic integral is defined as the $\mathcal{L}^2$-limit of Riemann-Stieltjes sums:
$$\int_0^T X_t dW_t = \lim_{\|\Pi\| \to 0} \sum_{i=0}^{n-1} X_{t_i} (W_{t_{i+1}} - W_{t_i})$$
**Crucial Invariant:** Evaluation occurs strictly at the *left endpoint* $t_i$, ensuring the integral is a martingale: $\mathbb{E}[\int_0^t X_s dW_s] = 0$.

By quadratic variation analysis:
$$d[W, W]_t = (dW_t)^2 = dt, \quad dW_t dt = 0, \quad (dt)^2 = 0$$

### 2. Itô's Lemma (One-Dimensional Formulation)
Let $X_t$ be an Itô drift-diffusion process satisfying the stochastic differential equation (SDE):
$$dX_t = \mu(t, X_t) dt + \sigma(t, X_t) dW_t$$
Let $f(t, x) \in \mathcal{C}^{1, 2}([0, \infty) \times \mathbb{R})$. Then $Y_t = f(t, X_t)$ is also an Itô process, whose differential is:
$$df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$$

Substituting $(dX_t)^2 = \sigma(t, X_t)^2 dt$:
$$df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

### 3. Geometric Brownian Motion (GBM)
The price process $S_t$ of a risky asset is modeled as:
$$dS_t = __DRIFT__ S_t dt + __VOL__ S_t dW_t$$
Applying Itô's Lemma to $f(S_t) = \ln(S_t)$:
$$\frac{\partial f}{\partial S} = \frac{1}{S}, \quad \frac{\partial^2 f}{\partial S^2} = -\frac{1}{S^2}$$
$$d(\ln S_t) = \frac{1}{S_t} (__DRIFT__ S_t dt + __VOL__ S_t dW_t) + \frac{1}{2} \left( -\frac{1}{S_t^2} \right) (__VOL__^2 S_t^2 dt) = \left( __DRIFT__ - \frac{__VOL__^2}{2} \right) dt + __VOL__ dW_t$$
Integrating over $[0, t]$:
$$S_t = S_0 \exp\left( \left( __DRIFT__ - \frac{__VOL__^2}{2} \right) t + __VOL__ W_t \right)$$

### 4. Derivation of the Black-Scholes-Merton Partial Differential Equation
Consider a derivative security whose price at time $t$ is given by $V(t, S_t)$.
By Itô's Lemma:
$$dV = \left( \frac{\partial V}{\partial t} + __DRIFT__ S \frac{\partial V}{\partial S} + \frac{1}{2} __VOL__^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt + __VOL__ S \frac{\partial V}{\partial S} dW_t$$

**Delta-Hedging Portfolio Construction:**
Construct a portfolio $\Pi$ composed of one short position in the derivative $V$ and a long position of $\Delta$ shares of stock $S$:
$$\Pi = -V + \Delta S$$
Over an infinitesimal time interval $dt$:
$$d\Pi = -dV + \Delta dS$$
Substitute the differentials $dV$ and $dS$:
$$d\Pi = -\left( \frac{\partial V}{\partial t} + __DRIFT__ S \frac{\partial V}{\partial S} + \frac{1}{2} __VOL__^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt - __VOL__ S \frac{\partial V}{\partial S} dW_t + \Delta (__DRIFT__ S dt + __VOL__ S dW_t)$$
Grouping deterministic and stochastic components:
$$d\Pi = \left( -\frac{\partial V}{\partial t} - __DRIFT__ S \frac{\partial V}{\partial S} - \frac{1}{2} __VOL__^2 S^2 \frac{\partial^2 V}{\partial S^2} + \Delta __DRIFT__ S \right) dt + __VOL__ S \left( \Delta - \frac{\partial V}{\partial S} \right) dW_t$$

To completely eliminate stochastic uncertainty (diffusive risk), choose the delta-hedge ratio:
$$\Delta = \frac{\partial V}{\partial S}$$
The stochastic $dW_t$ term identically vanishes:
$$d\Pi = \left( -\frac{\partial V}{\partial t} - \frac{1}{2} __VOL__^2 S^2 \frac{\partial^2 V}{\partial S^2} \right) dt$$

Under the No-Arbitrage Principle, the return on this risk-free portfolio must equal the risk-free rate $r$:
$$d\Pi = r \Pi dt = r \left( -V + \frac{\partial V}{\partial S} S \right) dt$$

Equating the two expressions for $d\Pi$:
$$-\frac{\partial V}{\partial t} - \frac{1}{2} __VOL__^2 S^2 \frac{\partial^2 V}{\partial S^2} = -r V + r S \frac{\partial V}{\partial S}$$

Rearranging gives the celebrated **Black-Scholes Partial Differential Equation**:
$$\frac{\partial V}{\partial t} + r S \frac{\partial V}{\partial S} + \frac{1}{2} __VOL__^2 S^2 \frac{\partial^2 V}{\partial S^2} - r V = 0 \quad \blacksquare$$
"""
    text = template.replace("__DRIFT__", drift).replace("__VOL__", vol)
    meta = {
        "domain": "stem",
        "subdomain": "stochastic_calculus",
        "topic": "ito_lemma_and_black_scholes",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 4: Abstract Algebra & Number Theory
# ==============================================================================

def gen_abstract_algebra_sylow_theorems(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    p = rng.choice([2, 3, 5, 7])
    template = r"""# Abstract Algebra: Sylow Theorems and Finite Group Structure
## The Sylow $p$-Subgroup Architecture for Prime $p = __P__$

### 1. Group Actions and the Class Equation
Let $G$ be a finite group and let $X$ be a finite set. A group action $\cdot: G \times X \to X$ partitions $X$ into disjoint orbits $\mathcal{O}_x = \{ g \cdot x : g \in G \}$.
The stabilizer of $x \in X$ is defined as $G_x = \{ g \in G : g \cdot x = x \} \le G$.

**Orbit-Stabilizer Theorem:**
For any $x \in X$:
$$|\mathcal{O}_x| = [G : G_x] = \frac{|G|}{|G_x|}$$
When $G$ acts on itself by conjugation ($g \cdot x = g x g^{-1}$), the orbits are the conjugacy classes $C(x)$, and the stabilizers are the centralizers $C_G(x)$. This gives rise to the **Class Equation**:
$$|G| = |Z(G)| + \sum_{i=1}^k [G : C_G(x_i)]$$
where $Z(G) = \{ z \in G : gz = zg, \forall g \in G \}$ is the center of $G$, and the sum runs over representatives of non-central conjugacy classes.

### 2. Definitions: $p$-Groups and Sylow $p$-Subgroups
Let $p$ be a prime number.
- A group $P$ is called a **$p$-group** if the order of every element in $P$ is a power of $p$. By Lagrange's Theorem, if $P$ is finite, $|P| = p^k$ for some $k \ge 1$.
- Let $|G| = p^k m$ with $\gcd(p, m) = 1$. A subgroup $P \le G$ is called a **Sylow $p$-subgroup** (or $p$-Sylow subgroup) if $|P| = p^k$.
- We denote $\text{Syl}_p(G)$ as the set of all Sylow $p$-subgroups of $G$, and $n_p = |\text{Syl}_p(G)|$.

### 3. The Three Sylow Theorems

#### First Sylow Theorem (Existence)
Let $G$ be a finite group of order $|G| = p^k m$ with $\gcd(p, m) = 1$ and $k \ge 1$.
Then for each $1 \le j \le k$, $G$ contains a subgroup of order $p^j$. In particular, $\text{Syl}_p(G) \neq \emptyset$.

**Proof Outline (Induction on $|G|$):**
If $p$ divides $|Z(G)|$, by Cauchy's Theorem for abelian groups, $Z(G)$ has a subgroup $N$ of order $p$. Since $N \le Z(G)$, $N$ is normal in $G$. Consider the quotient $G/N$ of order $p^{k-1} m$. By inductive hypothesis, $G/N$ contains a subgroup of order $p^{j-1}$, whose preimage under the canonical projection $\pi: G \to G/N$ yields a subgroup of order $p^j$ in $G$.
If $p$ does not divide $|Z(G)|$, by the Class Equation:
$$|G| = |Z(G)| + \sum_{i} [G : C_G(x_i)]$$
Since $p \mid |G|$ and $p \nmid |Z(G)|$, there must exist a representative $x_i$ such that $p \nmid [G : C_G(x_i)]$.
Thus $p^k$ divides $|C_G(x_i)|$. Since $x_i \notin Z(G)$, $|C_G(x_i)| < |G|$. By the inductive hypothesis on $C_G(x_i)$, there exists a subgroup of order $p^j$ in $C_G(x_i) \le G$. $\blacksquare$

#### Second Sylow Theorem (Conjugacy)
All Sylow $p$-subgroups of $G$ are conjugate to one another. That is, if $P, Q \in \text{Syl}_p(G)$, there exists $g \in G$ such that:
$$Q = g P g^{-1}$$

#### Third Sylow Theorem (Counting & Congruence)
The number $n_p$ of Sylow $p$-subgroups satisfies:
$$n_p \equiv 1 \pmod p \quad \text{and} \quad n_p \mid m$$
where $|G| = p^k m$ with $\gcd(p, m) = 1$.
Furthermore, $n_p = [G : N_G(P)]$ where $N_G(P) = \{ g \in G : g P g^{-1} = P \}$ is the normalizer of $P$.

**Immediate Structural Corollary:**
A Sylow $p$-subgroup $P$ is normal in $G$ ($P \trianglelefteq G$) if and only if $n_p = 1$.
"""
    text = template.replace("__P__", str(p))
    meta = {
        "domain": "stem",
        "subdomain": "abstract_algebra",
        "topic": "sylow_theorems",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


def gen_number_theory_elliptic_curves(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    a_param = rng.choice([2, 3, 5])
    b_param = rng.choice([7, 11, 13])
    disc = 4 * (a_param ** 3) + 27 * (b_param ** 2)
    template = r"""# Algebraic Number Theory & Cryptography: Elliptic Curves over Finite Fields
## Group Law on Weierstrass Curves $E: y^2 = x^3 + ax + b$

### 1. The Weierstrass Equation and Non-Singularity
Let $K$ be a field of characteristic $\text{char}(K) \neq 2, 3$. An elliptic curve $E$ over $K$ is defined by the affine equation:
$$E: y^2 = x^3 + a x + b, \quad a, b \in K$$
together with a single point at infinity $\mathcal{O}$, which acts as the identity element of the group.

The non-singularity condition requires that the cubic polynomial $f(x) = x^3 + ax + b$ has no repeated roots.
The roots of $f(x)$ are distinct if and only if the **discriminant** $\Delta$ is non-zero:
$$\Delta = -16(4a^3 + 27b^2) \neq 0$$
For parameters $a = __A__, b = __B__$:
$$4a^3 + 27b^2 = 4(__A__)^3 + 27(__B__)^2 = __DISC__ \neq 0$$
Thus $E$ is non-singular.

### 2. The Geometric Group Law (Chord-and-Tangent Method)
The set of $K$-rational points $E(K) = \{ (x, y) \in K \times K : y^2 = x^3 + ax + b \} \cup \{ \mathcal{O} \}$ forms an abelian group under the addition operation $+$ defined as follows:

**Point Addition:**
Let $P = (x_1, y_1)$ and $Q = (x_2, y_2)$ be two distinct points with $x_1 \neq x_2$.
The secant line through $P$ and $Q$ has slope $\lambda$:
$$\lambda = \frac{y_2 - y_1}{x_2 - x_1}$$
Substituting $y = \lambda(x - x_1) + y_1$ into $E$:
$$(\lambda(x - x_1) + y_1)^2 = x^3 + ax + b \implies x^3 - \lambda^2 x^2 + \dots = 0$$
By Vieta's formulas, the sum of the three roots of this cubic is $\lambda^2$:
$$x_1 + x_2 + x_3 = \lambda^2 \implies x_3 = \lambda^2 - x_1 - x_2$$
The $y$-coordinate of the intersection point is $y_3' = \lambda(x_3 - x_1) + y_1$. Reflecting across the $x$-axis gives the group sum $P + Q = (x_3, y_3)$:
$$y_3 = \lambda(x_1 - x_3) - y_1$$

**Point Doubling ($P = Q$):**
If $y_1 \neq 0$, the tangent line to the curve at $P$ has slope obtained by implicit differentiation $2y dy = (3x^2 + a) dx$:
$$\lambda = \frac{3x_1^2 + a}{2y_1}$$
Then $2P = (x_3, y_3)$ is given by:
$$x_3 = \lambda^2 - 2x_1$$
$$y_3 = \lambda(x_1 - x_3) - y_1$$
If $y_1 = 0$, the tangent is vertical, and $2P = \mathcal{O}$.

### 3. Hasse's Bound for Elliptic Curves over Finite Fields $\mathbb{F}_q$
When $K = \mathbb{F}_q$ is a finite field with $q = p^k$ elements:

**Theorem (Hasse's Theorem):**
The number of $\mathbb{F}_q$-rational points on $E$, denoted by $\#E(\mathbb{F}_q)$, satisfies the bound:
$$|\#E(\mathbb{F}_q) - (q + 1)| \le 2\sqrt{q}$$
The quantity $t = q + 1 - \#E(\mathbb{F}_q)$ is called the **trace of Frobenius**.

### 4. Elliptic Curve Discrete Logarithm Problem (ECDLP)
Given an elliptic curve $E(\mathbb{F}_q)$, a base generator point $G \in E(\mathbb{F}_q)$ of prime order $n$, and a point $Q = kG$:
$$\text{ECDLP}: \text{Find the unique scalar } k \in [0, n-1] \text{ such that } Q = kG$$
Unlike the discrete logarithm in finite fields $\mathbb{F}_q^\times$ (where index calculus achieves sub-exponential complexity $L_q[1/3]$), ECDLP on generic curves has no known sub-exponential algorithm. The best known generic attacks (Pollard's rho algorithm) run in exponential time $\mathcal{O}(\sqrt{n})$.
"""
    text = template.replace("__A__", str(a_param)).replace("__B__", str(b_param)).replace("__DISC__", str(disc))
    meta = {
        "domain": "stem",
        "subdomain": "number_theory",
        "topic": "elliptic_curves_group_law",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 5: Physics & Differential Equations
# ==============================================================================

def gen_classical_mechanics_lagrangian(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    system = rng.choice(["Coupled Double Pendulum", "Harmonic Oscillator in Phase Space", "Relativistic Charged Particle in EM Field"])
    template = r"""# Theoretical Physics: Analytical Mechanics and Variational Principles
## Lagrangian Formulation for a __SYSTEM__

### 1. Hamilton's Principle of Stationary Action
Let a mechanical system be described by $n$ generalized coordinates $\mathbf{q} = (q_1, \dots, q_n)^T$ and generalized velocities $\mathbf{\dot{q}} = (\dot{q}_1, \dots, \dot{q}_n)^T$.
The state of the system is parameterized along a trajectory $\gamma: [t_1, t_2] \to \mathbb{R}^n$ with fixed boundary endpoints $\mathbf{q}(t_1) = \mathbf{q}_1$ and $\mathbf{q}(t_2) = \mathbf{q}_2$.

The **action functional** $S[\mathbf{q}]$ is defined as the time integral of the Lagrangian $\mathcal{L}(\mathbf{q}, \mathbf{\dot{q}}, t) = T - V$:
$$S[\mathbf{q}] = \int_{t_1}^{t_2} \mathcal{L}(\mathbf{q}(t), \mathbf{\dot{q}}(t), t) dt$$

**Hamilton's Principle:**
The actual physical motion $\mathbf{q}(t)$ is a stationary point of the action functional under all infinitesimal variations $\delta \mathbf{q}(t)$ satisfying $\delta \mathbf{q}(t_1) = \delta \mathbf{q}(t_2) = \mathbf{0}$:
$$\delta S = 0$$

### 2. Derivation of the Euler-Lagrange Equations
Consider a one-parameter family of curves $\mathbf{q}(t, \epsilon) = \mathbf{q}(t) + \epsilon \boldsymbol{\eta}(t)$ where $\boldsymbol{\eta}(t_1) = \boldsymbol{\eta}(t_2) = \mathbf{0}$.
$$\delta S = \left. \frac{d}{d\epsilon} \int_{t_1}^{t_2} \mathcal{L}(\mathbf{q} + \epsilon \boldsymbol{\eta}, \mathbf{\dot{q}} + \epsilon \boldsymbol{\dot{\eta}}, t) dt \right|_{\epsilon = 0} = \int_{t_1}^{t_2} \sum_{i=1}^n \left( \frac{\partial \mathcal{L}}{\partial q_i} \eta_i + \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \dot{\eta}_i \right) dt$$

Integrating the second term by parts:
$$\int_{t_1}^{t_2} \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \dot{\eta}_i dt = \left[ \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \eta_i \right]_{t_1}^{t_2} - \int_{t_1}^{t_2} \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \right) \eta_i dt$$
Because the variation vanishes at the boundaries ($\eta_i(t_1) = \eta_i(t_2) = 0$), the boundary term is identically zero:
$$\delta S = \int_{t_1}^{t_2} \sum_{i=1}^n \left[ \frac{\partial \mathcal{L}}{\partial q_i} - \frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \right) \right] \eta_i(t) dt = 0$$
Since this must hold for arbitrary independent variations $\eta_i(t)$, by the Fundamental Lemma of the Calculus of Variations, each bracketed term must independently vanish:
$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}_i} \right) - \frac{\partial \mathcal{L}}{\partial q_i} = 0, \quad \forall i \in \{1, \dots, n\}$$

### 3. Generalized Momentum & Noether's Theorem
The **canonical momentum** conjugate to $q_i$ is:
$$p_i = \frac{\partial \mathcal{L}}{\partial \dot{q}_i}$$
If $\mathcal{L}$ is independent of coordinate $q_k$ (i.e. $\frac{\partial \mathcal{L}}{\partial q_k} = 0$, a *cyclic coordinate*):
$$\frac{d p_k}{dt} = \frac{\partial \mathcal{L}}{\partial q_k} = 0 \implies p_k = \text{constant of motion}$$

**Theorem (Noether's Theorem):**
Every continuous global symmetry of the action corresponds to a conserved physical current:
1. Time-translation invariance ($t \to t + \delta t$) $\implies$ Conservation of total energy (Hamiltonian $\mathcal{H} = \sum p_i \dot{q}_i - \mathcal{L}$).
2. Spatial-translation invariance ($\mathbf{r} \to \mathbf{r} + \delta \mathbf{r}$) $\implies$ Conservation of linear momentum $\mathbf{P}$.
3. Rotational invariance ($\mathbf{r} \to \mathbf{R}(\theta) \mathbf{r}$) $\implies$ Conservation of angular momentum $\mathbf{L}$.
"""
    text = template.replace("__SYSTEM__", system)
    meta = {
        "domain": "stem",
        "subdomain": "theoretical_physics",
        "topic": "lagrangian_mechanics",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


def gen_quantum_mechanics_schrodinger_uncertainty(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    state = rng.choice(["|\\psi\\rangle", "|\\phi\\rangle"])
    template = r"""# Quantum Mechanics: Spectral Theory of Hermitian Operators and the Uncertainty Principle
## Derivation of the Robertson-Schrödinger Invariant

### 1. Postulates of Quantum Mechanics in Hilbert Space $\mathcal{H}$
1. The state of a quantum physical system is completely specified by a unit ray in a complex separable Hilbert space $\mathcal{H}$:
   $$\langle \psi | \psi \rangle = \int_{\mathbb{R}} \psi^*(x) \psi(x) dx = 1$$
2. Every physically measurable observable $\mathcal{A}$ corresponds to a densely defined linear self-adjoint (Hermitian) operator $\hat{A} = \hat{A}^\dagger$ on $\mathcal{H}$.
3. The expectation value of an observable $\hat{A}$ in state __STATE__ is:
   $$\langle \hat{A} \rangle = \langle \psi | \hat{A} | \psi \rangle$$
   The variance $\Delta A^2$ is defined as:
   $$\Delta A^2 = \langle (\hat{A} - \langle \hat{A} \rangle)^2 \rangle = \langle \psi | (\hat{A} - \langle \hat{A} \rangle)^2 | \psi \rangle$$

### 2. The Canonical Commutation Relation
In position space representation, the position operator $\hat{X}$ and momentum operator $\hat{P}$ act on wavefunctions $\psi(x)$ as:
$$\hat{X} \psi(x) = x \psi(x), \quad \hat{P} \psi(x) = -i\hbar \frac{d\psi}{dx}$$

Evaluating the commutator $[\hat{X}, \hat{P}] = \hat{X}\hat{P} - \hat{P}\hat{X}$ applied to test function $\psi(x)$:
$$[\hat{X}, \hat{P}] \psi(x) = x \left( -i\hbar \frac{d\psi}{dx} \right) - \left( -i\hbar \frac{d}{dx} (x \psi(x)) \right) = -i\hbar x \frac{d\psi}{dx} + i\hbar \left( \psi(x) + x \frac{d\psi}{dx} \right) = i\hbar \psi(x)$$
Since this holds for all $\psi \in \mathcal{S}(\mathbb{R})$, we have the fundamental Heisenberg algebra:
$$[\hat{X}, \hat{P}] = i\hbar \hat{I}$$

### 3. Proof of the Generalized Robertson-Schrödinger Uncertainty Relation
Let $\hat{A}$ and $\hat{B}$ be two arbitrary Hermitian operators on $\mathcal{H}$.
Define the zero-mean operators:
$$\delta \hat{A} = \hat{A} - \langle \hat{A} \rangle, \quad \delta \hat{B} = \hat{B} - \langle \hat{B} \rangle$$
Note that $\delta \hat{A} = (\delta \hat{A})^\dagger$ and $\delta \hat{B} = (\delta \hat{B})^\dagger$.

Define two vectors $|u\rangle = \delta \hat{A} |\psi\rangle$ and $|v\rangle = \delta \hat{B} |\psi\rangle$.
By the Cauchy-Schwarz inequality in Hilbert space:
$$\langle u | u \rangle \langle v | v \rangle \ge |\langle u | v \rangle|^2$$
Computing the norms:
$$\langle u | u \rangle = \langle \psi | (\delta \hat{A})^\dagger \delta \hat{A} | \psi \rangle = \langle \psi | (\delta \hat{A})^2 | \psi \rangle = \Delta A^2$$
$$\langle v | v \rangle = \langle \psi | (\delta \hat{B})^\dagger \delta \hat{B} | \psi \rangle = \Delta B^2$$
Thus:
$$\Delta A^2 \Delta B^2 \ge |\langle \psi | \delta \hat{A} \delta \hat{B} | \psi \rangle|^2$$

Decompose the operator product $\delta \hat{A} \delta \hat{B}$ into Hermitian and anti-Hermitian parts:
$$\delta \hat{A} \delta \hat{B} = \frac{1}{2} \{\delta \hat{A}, \delta \hat{B}\} + \frac{1}{2} [\delta \hat{A}, \delta \hat{B}]$$
where $\{\hat{A}, \hat{B}\} = \hat{A}\hat{B} + \hat{B}\hat{A}$ is the anti-commutator.
Note that:
$$[\delta \hat{A}, \delta \hat{B}] = [\hat{A} - \langle A \rangle, \hat{B} - \langle B \rangle] = [\hat{A}, \hat{B}]$$
The expectation value is:
$$\langle \psi | \delta \hat{A} \delta \hat{B} | \psi \rangle = \frac{1}{2} \langle \{\delta \hat{A}, \delta \hat{B}\} \rangle + \frac{1}{2} \langle [\hat{A}, \hat{B}] \rangle$$
Since $\{\delta \hat{A}, \delta \hat{B}\}$ is Hermitian, its expectation value is strictly real.
Since $[\hat{A}, \hat{B}]^\dagger = -[\hat{A}, \hat{B}]$ (anti-Hermitian), its expectation value is purely imaginary.
For any complex number $z = x + i y$, $|z|^2 = x^2 + y^2 \ge y^2$. Therefore:
$$|\langle \psi | \delta \hat{A} \delta \hat{B} | \psi \rangle|^2 = \left( \frac{1}{2} \langle \{\delta \hat{A}, \delta \hat{B}\} \rangle \right)^2 + \left( \frac{1}{2i} \langle [\hat{A}, \hat{B}] \rangle \right)^2 \ge \frac{1}{4} |\langle [\hat{A}, \hat{B}] \rangle|^2$$

Taking the square root on both sides yields the celebrated **Robertson Uncertainty Relation**:
$$\Delta A \Delta B \ge \frac{1}{2} |\langle [\hat{A}, \hat{B}] \rangle|$$

Substituting $\hat{A} = \hat{X}$ and $\hat{B} = \hat{P}$ where $[\hat{X}, \hat{P}] = i\hbar$:
$$\Delta X \Delta P \ge \frac{1}{2} |\langle i\hbar \rangle| = \frac{\hbar}{2} \quad \blacksquare$$
"""
    text = template.replace("__STATE__", state)
    meta = {
        "domain": "stem",
        "subdomain": "quantum_mechanics",
        "topic": "uncertainty_principle_derivation",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 6: Competition & Olympiad Mathematics
# ==============================================================================

def gen_olympiad_generating_functions(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    k_term = rng.choice([2, 3, 4])
    template = r"""# Olympiad Mathematics: Ordinary & Exponential Generating Functions
## Combinatorial Recurrences and Analytic Enumeration

### 1. Foundations of Formal Power Series
Let $R$ be a commutative ring (typically $\mathbb{C}$). The ring of formal power series $R[[x]]$ consists of expressions:
$$A(x) = \sum_{n=0}^\infty a_n x^n$$
Operations are defined algebraically without convergence requirements:
$$(A + B)(x) = \sum_{n=0}^\infty (a_n + b_n) x^n, \quad (A \cdot B)(x) = \sum_{n=0}^\infty \left( \sum_{k=0}^n a_k b_{n-k} \right) x^n$$
An element $A(x)$ is invertible in $R[[x]]$ if and only if its constant term $a_0$ is a unit in $R$.

### 2. Solving Second-Order Non-Homogeneous Recurrences
Consider the recurrence:
$$a_n = 5 a_{n-1} - 6 a_{n-2} + __K__^n, \quad \forall n \ge 2$$
with initial conditions $a_0 = 1, a_1 = 4$.

**Step 1: Define the Generating Function.**
Let $A(x) = \sum_{n=0}^\infty a_n x^n$.
Multiply the recurrence equation by $x^n$ and sum over all $n \ge 2$:
$$\sum_{n=2}^\infty a_n x^n = 5x \sum_{n=2}^\infty a_{n-1} x^{n-1} - 6x^2 \sum_{n=2}^\infty a_{n-2} x^{n-2} + \sum_{n=2}^\infty (__K__ x)^n$$

Express in terms of $A(x)$:
$$A(x) - a_0 - a_1 x = 5x (A(x) - a_0) - 6x^2 A(x) + \frac{(__K__ x)^2}{1 - __K__ x}$$
Substitute $a_0 = 1$ and $a_1 = 4$:
$$A(x)(1 - 5x + 6x^2) = 1 - x + \frac{__KSQ__ x^2}{1 - __K__ x}$$
Factoring the characteristic polynomial $1 - 5x + 6x^2 = (1 - 2x)(1 - 3x)$:
$$A(x) = \frac{(1 - x)(1 - __K__ x) + __KSQ__ x^2}{(1 - 2x)(1 - 3x)(1 - __K__ x)}$$

**Step 2: Partial Fraction Decomposition.**
We express $A(x)$ in terms of simple fractions:
$$A(x) = \frac{C_1}{1 - 2x} + \frac{C_2}{1 - 3x} + \frac{C_3}{1 - __K__ x}$$
Using the geometric series expansion $\frac{1}{1 - r x} = \sum_{n=0}^\infty r^n x^n$:
$$a_n = C_1 \cdot 2^n + C_2 \cdot 3^n + C_3 \cdot __K__^n$$

### 3. The Catalan Numbers via Functional Equations
The Catalan numbers $C_n = \frac{1}{n+1} \binom{2n}{n}$ count the number of valid parenthesizations of $n$ pairs, Dyck paths, and binary trees.
They satisfy the convolution recurrence:
$$C_0 = 1, \quad C_{n+1} = \sum_{k=0}^n C_k C_{n-k}$$
Let $C(x) = \sum_{n=0}^\infty C_n x^n$. Multiplying by $x^{n+1}$ and summing:
$$\sum_{n=0}^\infty C_{n+1} x^{n+1} = x \sum_{n=0}^\infty \left( \sum_{k=0}^n C_k C_{n-k} \right) x^n \implies C(x) - 1 = x C(x)^2$$
Rearranging gives the quadratic functional equation:
$$x C(x)^2 - C(x) + 1 = 0$$
Applying the quadratic formula:
$$C(x) = \frac{1 \pm \sqrt{1 - 4x}}{2x}$$
Since $C(0) = C_0 = 1$, we must choose the minus sign (as $x \to 0$, $\frac{1 - \sqrt{1-4x}}{2x} \to 1$).
Expanding $\sqrt{1 - 4x} = (1 - 4x)^{1/2}$ via the Generalized Binomial Theorem:
$$(1 - 4x)^{1/2} = 1 + \sum_{n=1}^\infty \binom{1/2}{n} (-4x)^n = 1 - 2 \sum_{n=1}^\infty \frac{1}{n} \binom{2n-2}{n-1} x^n$$
Substituting back into $C(x)$ directly extracts the exact closed-form coefficient:
$$C_n = \frac{1}{n+1} \binom{2n}{n} \quad \blacksquare$$
"""
    text = template.replace("__K__", str(k_term)).replace("__KSQ__", str(k_term ** 2))
    meta = {
        "domain": "stem",
        "subdomain": "olympiad_math",
        "topic": "generating_functions_catalan",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Domain 7: Complex Analysis & Contour Integration
# ==============================================================================

def gen_complex_analysis_residue_theorem(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    pole = rng.choice([1, 2, 3])
    pole_sq = pole ** 2
    two_pole = 2 * pole
    template = r"""# Complex Analysis: Cauchy's Residue Theorem and Real Integrals
## Evaluation of Improper Oscillatory and Rational Integrals via Contour Deformation

### 1. Holomorphic Functions and the Cauchy-Riemann Equations
Let $U \subseteq \mathbb{C}$ be an open set. A function $f: U \to \mathbb{C}$ is **holomorphic** at $z_0$ if the complex derivative:
$$f'(z_0) = \lim_{z \to z_0} \frac{f(z) - f(z_0)}{z - z_0}$$
exists. If $f(z) = u(x, y) + i v(x, y)$, holomorphy is equivalent to the **Cauchy-Riemann equations**:
$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

**Cauchy-Goursat Theorem:**
If $f$ is holomorphic in a simply connected domain $D$ and $\gamma$ is a closed rectifiable contour in $D$:
$$\oint_\gamma f(z) dz = 0$$

### 2. Laurent Series and Residues
If $f$ has an isolated singularity at $z_0$, it admits a Laurent series expansion in an annulus $0 < |z - z_0| < R$:
$$f(z) = \sum_{n=-\infty}^\infty a_n (z - z_0)^n$$
The coefficient $a_{-1}$ is the **residue** of $f$ at $z_0$:
$$\text{Res}(f, z_0) = a_{-1} = \frac{1}{2\pi i} \oint_{|z - z_0| = r} f(z) dz$$
For a simple pole ($m = 1$):
$$\text{Res}(f, z_0) = \lim_{z \to z_0} (z - z_0) f(z)$$

### 3. The Residue Theorem
Let $\Gamma$ be a positively oriented simple closed contour enclosing isolated singularities $z_1, \dots, z_k$. Then:
$$\oint_\Gamma f(z) dz = 2\pi i \sum_{j=1}^k \text{Res}(f, z_j)$$

### 4. Exemplary Evaluation of an Improper Real Integral
Evaluate the real integral:
$$I = \int_0^\infty \frac{\cos(x)}{x^2 + __POLE_SQ__} dx = \frac{1}{2} \int_{-\infty}^\infty \frac{\cos(x)}{x^2 + __POLE_SQ__} dx$$
Consider the complex function $f(z) = \frac{e^{iz}}{z^2 + __POLE_SQ__}$ integrated over the semicircle contour $\Gamma_R = [-R, R] \cup C_R$ where $C_R = \{ R e^{i\theta} : 0 \le \theta \le \pi \}$.

1. **Poles:** The denominator $z^2 + __POLE_SQ__ = (z - i__POLE__)(z + i__POLE__)$ has simple poles at $z = \pm i__POLE__$. Only $z_1 = i__POLE__$ lies in the upper half-plane bounded by $\Gamma_R$ for $R > __POLE__$.
2. **Residue at $z_1 = i__POLE__$:**
   $$\text{Res}(f, i__POLE__) = \lim_{z \to i__POLE__} (z - i__POLE__) \frac{e^{iz}}{(z - i__POLE__)(z + i__POLE__)} = \frac{e^{i(i__POLE__)}}{2i__POLE__} = \frac{e^{-__POLE__}}{2i__POLE__}$$
3. **Jordan's Lemma:** Along the circular arc $C_R$:
   $$\left| \int_{C_R} \frac{e^{iz}}{z^2 + __POLE_SQ__} dz \right| \le \int_0^\pi \frac{e^{-R \sin\theta}}{R^2 - __POLE_SQ__} R d\theta \xrightarrow[R \to \infty]{} 0$$
4. **Conclusion:**
   $$\oint_{\Gamma_R} f(z) dz = \int_{-R}^R \frac{e^{ix}}{x^2 + __POLE_SQ__} dx + \int_{C_R} f(z) dz = 2\pi i \left( \frac{e^{-__POLE__}}{2i__POLE__} \right) = \frac{\pi e^{-__POLE__}}{__POLE__}$$
   Taking the real part as $R \to \infty$:
   $$\int_{-\infty}^\infty \frac{\cos(x)}{x^2 + __POLE_SQ__} dx = \frac{\pi e^{-__POLE__}}{__POLE__} \implies \int_0^\infty \frac{\cos(x)}{x^2 + __POLE_SQ__} dx = \frac{\pi e^{-__POLE__}}{__TWO_POLE__} \quad \blacksquare$$
"""
    text = template.replace("__POLE__", str(pole)).replace("__POLE_SQ__", str(pole_sq)).replace("__TWO_POLE__", str(two_pole))
    meta = {
        "domain": "stem",
        "subdomain": "complex_analysis",
        "topic": "residue_theorem_contour_integration",
        "latex_density": calculate_latex_density(text),
        "tokens": estimate_tokens(text),
    }
    return text.strip(), meta


# ==============================================================================
# Master Generator Registry
# ==============================================================================

STEM_GENERATOR_REGISTRY: List[Callable[[random.Random], Tuple[str, Dict[str, Any]]]] = [
    gen_multivariable_calculus_stokes,
    gen_real_analysis_bolzano_weierstrass,
    gen_linear_algebra_svd_spectral,
    gen_stochastic_calculus_ito_black_scholes,
    gen_abstract_algebra_sylow_theorems,
    gen_number_theory_elliptic_curves,
    gen_classical_mechanics_lagrangian,
    gen_quantum_mechanics_schrodinger_uncertainty,
    gen_olympiad_generating_functions,
    gen_complex_analysis_residue_theorem,
]


def generate_base_stem_corpus(
    output_dir: Path,
    count: int = 15000,
    seed: int = 42,
    max_mb: float = 28.0,
    prefix: str = "base_stem_part"
) -> Dict[str, Any]:
    """Generate partitioned base STEM & Formal Mathematics pretraining documents strictly under 28 MB."""
    rng = random.Random(seed)
    max_bytes = int(max_mb * 1024 * 1024)
    writer = ShardedSTEMWriter(output_dir=output_dir, prefix=prefix, max_bytes=max_bytes)

    subdomain_counts: Dict[str, int] = {}
    total_tokens_est = 0
    total_latex_density = 0.0

    print(f"Generating {count:,} STEM & Formal Mathematics base pretraining documents into {output_dir}...")

    try:
        for i in range(count):
            gen_fn = rng.choice(STEM_GENERATOR_REGISTRY)
            text, meta = gen_fn(rng)

            writer.write_document(text=text, meta=meta)

            sub = meta.get("subdomain", "math")
            toks = meta.get("tokens", estimate_tokens(text))
            density = meta.get("latex_density", calculate_latex_density(text))

            subdomain_counts[sub] = subdomain_counts.get(sub, 0) + 1
            total_tokens_est += toks
            total_latex_density += density

            if (i + 1) % 1000 == 0 or (i + 1) == count:
                print(f"  Progress: {i + 1:,} / {count:,} records processed...")
    finally:
        writer.close()

    manifest = {
        "version": "base-stem-v1.0",
        "total_records": writer.total_records,
        "total_bytes": writer.total_bytes,
        "total_mb": writer.total_bytes / (1024 * 1024),
        "estimated_tokens": total_tokens_est,
        "average_latex_density": round(total_latex_density / max(1, writer.total_records), 4),
        "partitions_count": len(writer.written_files),
        "partition_files": [f.name for f in writer.written_files],
        "subdomain_distribution": subdomain_counts,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nCompleted! Generated {writer.total_records:,} records across {len(writer.written_files)} partitions.")
    print(f"Total Size: {manifest['total_mb']:.2f} MB (~{total_tokens_est / 1e6:.1f} M tokens)")
    print(f"Manifest written to: {manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate STEM & Formal Mathematics Base Pretraining Data")
    parser.add_argument("--output-dir", type=str, default=r"E:\AI_Projects\dataset\stem_pretraining", help="Output directory")
    parser.add_argument("--count", type=int, default=15000, help="Number of documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-mb", type=float, default=28.0, help="Partition ceiling in MB")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    generate_base_stem_corpus(
        output_dir=out_path,
        count=args.count,
        seed=args.seed,
        max_mb=args.max_mb,
    )


if __name__ == "__main__":
    main()
