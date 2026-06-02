---
layout: post
title: "Eigenvectors: vectors for which a matrix doesn't cause a change in direction"
author: "Rishi Jain"
date: 2026-05-30
categories: notes
math: true
---

Picking up the linear-algebra thread from [the basics notes]({% post_url 2026-05-29-basic-linear-algebra-notes %}), where I left eigenvectors sitting on the "next" list. These follow Zachary Huang's [Give me 25 min, I will make Eigenvalue click forever](https://www.youtube.com/watch?v=oFdPHskO3Fk), worked out with the small numerical examples and a NumPy snippet per section. Same format as last time, written down because I wanted them pinned, and maybe useful if you are making the same pass. Everything here leans on the [transformations and determinant]({% post_url 2026-05-29-basic-linear-algebra-notes %}) ideas from those notes.

## Vectors a matrix leaves on their line

A matrix is an action. It takes an input vector \\( v \\) and produces an output \\( v^\prime = Av \\), which can rotate, shear, scale, or any mix of those.

Take a shear, \\( A = \begin{bmatrix} 1 & 1 \\\\ 0 & 1 \end{bmatrix} \\), and apply it to \\( v = [2,\, 3] \\). Working row by row,

\\[
v^\prime = \begin{bmatrix} (1)(2) + (1)(3) \\\\ (0)(2) + (1)(3) \end{bmatrix} = \begin{bmatrix} 5 \\\\ 3 \end{bmatrix}
\\]

\\( v \\) went from \\( [2, 3] \\) to \\( [5, 3] \\): it got bent to the right, knocked clean off the line it was sitting on, and its direction changed.

So here is the question. Are there non-zero vectors this matrix does not knock off their line, vectors whose direction survives the transformation entirely? Let's try a couple.

\\[
A \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\\\ 1 \end{bmatrix}, \qquad
A \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}.
\\]

The vertical vector turned. The horizontal one came back as itself. Anything lying on the x-axis points the same way after the shear as it did before.

```python
import numpy as np

A = np.array([[1, 1],
              [0, 1]])

v       = np.array([2, 3])
v_up    = np.array([0, 1])
v_right = np.array([1, 0])

print(A @ v)        # [5 3]
print(A @ v_up)     # [1 1]
print(A @ v_right)  # [1 0], the eigenvector
```

## Eigenvectors and eigenvalues

That surviving direction has a name. An eigenvector of \\( A \\) is a non-zero vector whose direction is unchanged when you apply \\( A \\). For the shear, every vector on the x-axis qualifies, and a whole line of them works, which is why the set is called an eigenspace rather than a single eigenvector.

The direction is preserved, but the length need not be. An eigenvalue is the stretch factor along that direction, written \\( \lambda \\). For the shear, \\( A[1, 0] = [1, 0] \\), so the vector was scaled by \\( \lambda = 1 \\). The sign and size of \\( \lambda \\) tell you what happened on that line: when \\( \lvert \lambda \rvert > 1 \\) the vector is stretched, when \\( \lvert \lambda \rvert < 1 \\) it is squished toward the origin, when \\( \lambda < 0 \\) it is flipped 180° and scaled by \\( \lvert \lambda \rvert \\), and when \\( \lambda = 0 \\) it collapses onto the origin, with the whole eigenspace mapping to the zero vector.

## The characteristic equation

Both of those facts pack into one equation,

\\[
Av = \lambda v.
\\]

The action of \\( A \\) on its eigenvector is identical to simply scaling that vector by a number. The trouble is solving it, since there are two unknowns at once, the scalar \\( \lambda \\) and the vector \\( v \\), so we have to pin one of them down first.

Move everything to one side and factor out \\( v \\),

\\[
Av - \lambda v = 0 \quad\Rightarrow\quad (A - \lambda)v = 0,
\\]

where \\( 0 \\) is the zero vector. There is a snag here: \\( A \\) is a matrix and \\( \lambda \\) is a single number, and you cannot subtract a scalar from a matrix. The fix is the identity matrix \\( I \\), the matrix version of 1: since \\( Iv = v \\), rewriting \\( \lambda v \\) as \\( \lambda I v \\) changes nothing, and now the subtraction is matrix minus matrix,

\\[
(A - \lambda I)\, v = 0.
\\]

Call the matrix part \\( M = A - \lambda I \\), so that \\( Mv = 0 \\). This asks which vector \\( M \\) sends to the zero vector. The answer is always \\( v = 0 \\), but eigenvectors have to be non-zero, and a non-zero \\( v \\) only exists if \\( M \\) is singular, a matrix that collapses space to a lower dimension so that a whole line of vectors lands on the origin. The test for singular is a zero determinant (this is the determinant idea from the [basics notes]({% post_url 2026-05-29-basic-linear-algebra-notes %}): \\( \det = 0 \\) means area has collapsed), which leaves us with

\\[
\det(A - \lambda I) = 0.
\\]

This is the characteristic equation. It has no \\( v \\) in it, only \\( \lambda \\), so we solve it first, and since it is always a polynomial its roots are the eigenvalues. The recipe from here is short: first solve \\( \det(A - \lambda I) = 0 \\) for the eigenvalues \\( \lambda \\), and then, for each \\( \lambda \\), plug it back into \\( (A - \lambda I)v = 0 \\) and solve for the eigenvector \\( v \\).

## A worked example

Take \\( A = \begin{bmatrix} 3 & 1 \\\\ 1 & 3 \end{bmatrix} \\), which is symmetric across the diagonal (that matters in a second).

For the eigenvalues, we subtract \\( \lambda \\) off the diagonal and take the determinant with \\( \det = ad - bc \\),

\\[
\det \begin{bmatrix} 3 - \lambda & 1 \\\\ 1 & 3 - \lambda \end{bmatrix}
= (3 - \lambda)^2 - 1
= \lambda^2 - 6\lambda + 8 = 0.
\\]

Factoring gives \\( (\lambda - 4)(\lambda - 2) = 0 \\), so \\( \lambda_1 = 4 \\) and \\( \lambda_2 = 2 \\). The transformation has two special axes, and these are how much it stretches along each.

For the eigenvectors, we plug \\( \lambda_1 = 4 \\) into \\( (A - \lambda I)v = 0 \\),

\\[
\begin{bmatrix} -1 & 1 \\\\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\\\ y \end{bmatrix} = 0
\quad\Rightarrow\quad -x + y = 0 \quad\Rightarrow\quad y = x.
\\]

Both rows say the same thing, which is exactly the point: the system is built to have infinitely many solutions, because we are after a whole line of vectors rather than one. Any non-zero vector with \\( y = x \\) works, so we take \\( v_1 = [1,\, 1] \\). For \\( \lambda_2 = 2 \\),

\\[
\begin{bmatrix} 1 & 1 \\\\ 1 & 1 \end{bmatrix} \begin{bmatrix} x \\\\ y \end{bmatrix} = 0
\quad\Rightarrow\quad x + y = 0 \quad\Rightarrow\quad y = -x,
\\]

giving \\( v_2 = [1,\, -1] \\). Along the \\( y = x \\) line vectors stretch 4×, and along \\( y = -x \\) they stretch 2×. Checking the bigger one, \\( A v_1 = [4, 4] = 4 [1, 1] = \lambda_1 v_1 \\). It holds.

```python
import numpy as np

A = np.array([[3, 1],
              [1, 3]])

vals, vecs = np.linalg.eig(A)
print(vals)        # [4. 2.]
print(vecs)
```

NumPy returns the same eigenvalues and the same directions, just normalized to unit length, so \\( [1, 1] \\) comes back as \\( [0.707, 0.707] \\) (and the sign can flip, since an eigenvector pins down a direction and the orientation along it is free to point either way).

## Symmetric matrices

That example was symmetric on purpose. Symmetric matrices (\\( A = A^\top \\)) come with two guarantees. The eigenvalues are always real, with no complex roots sneaking out of the characteristic polynomial, and the eigenvectors are always orthogonal, which you can check with the dot product: \\( v_1 \cdot v_2 = (1)(1) + (1)(-1) = 0 \\), perpendicular. This is why symmetric matrices are the comfortable case, and it is exactly the structure that PCA and covariance matrices lean on.

## Predicting a system 50 years out

The payoff is computing far-future states cheaply. Model a city and its suburb with a transition matrix,

\\[
A = \begin{bmatrix} 0.95 & 0.10 \\\\ 0.05 & 0.90 \end{bmatrix}, \qquad v_k = \begin{bmatrix} \text{city}_k \\\\ \text{suburb}_k \end{bmatrix}.
\\]

Each year 5% of the city leaves for the suburb and 10% of the suburb moves back, and the state after \\( k \\) years is \\( v_k = A^k v_0 \\). Want the picture 50 years out? Naively that is 50 matrix multiplications, and eigenvalues collapse it.

The eigenvalues here are \\( \lambda_1 = 1.0 \\) (eigenvector \\( v_1 = [2, 1] \\)) and \\( \lambda_2 = 0.85 \\) (eigenvector \\( v_2 = [1, -1] \\)). Because they form a basis, any starting vector is a mix of them, \\( v_0 = c_1 v_1 + c_2 v_2 \\). The trick is that \\( A \\) acting on an eigenvector is just multiplication by its eigenvalue, and that carries straight through powers, so

\\[
v_{50} = A^{50} v_0 = A^{50}(c_1 v_1 + c_2 v_2) = c_1 \lambda_1^{50} v_1 + c_2 \lambda_2^{50} v_2.
\\]

A 50-step simulation has become raising two numbers to the 50th power, which is rather satisfying, and it is the general formula for the long-run state of any linear system.

Start everyone in the city, \\( v_0 = [1{,}500{,}000,\, 0] \\). Solving \\( v_0 = c_1 v_1 + c_2 v_2 \\) gives \\( c_1 = c_2 = 500{,}000 \\), so

\\[
v_{50} = 500{,}000 \cdot (1.0)^{50} \begin{bmatrix} 2 \\\\ 1 \end{bmatrix} + 500{,}000 \cdot (0.85)^{50} \begin{bmatrix} 1 \\\\ -1 \end{bmatrix}.
\\]

Now look at the two scalars. \\( 1.0^{50} = 1 \\), but \\( 0.85^{50} \approx 0.000296 \\). The second term has been crushed to almost nothing, and the first term, riding the dominant eigenvalue \\( \lambda = 1 \\), is all that is left,

\\[
v_{50} \approx 500{,}000 \begin{bmatrix} 2 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 1{,}000{,}000 \\\\ 500{,}000 \end{bmatrix}.
\\]

```python
import numpy as np

A  = np.array([[0.95, 0.10],
               [0.05, 0.90]])
v0 = np.array([1_500_000, 0])     # everyone starts in the city

v50 = np.linalg.matrix_power(A, 50) @ v0
print(v50)                        # ~[1000148 499852], the 2:1 steady state
```

No matter where you start, any eigenvalue with \\( \lvert \lambda \rvert < 1 \\) decays away, and the system converges to the dominant eigenvector's direction. Here that is the 2:1 city-to-suburb split, which the population settles into regardless of the initial mix.

## What's next

These pick up where the [basics notes]({% post_url 2026-05-29-basic-linear-algebra-notes %}) stopped, and there are still two to go from the original list. PCA is eigenvectors of a covariance matrix, where PC1 is the direction of maximum variance and PC2 is orthogonal to it (the symmetric-matrix guarantees from above are exactly why those axes come out perpendicular and real). SVD is the generalization that drops the square-matrix requirement. Both are the same move as the population model: find the directions a transformation only stretches, then read off which one dominates.
