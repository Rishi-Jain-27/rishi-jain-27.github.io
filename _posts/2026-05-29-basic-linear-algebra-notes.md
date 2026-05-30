---
layout: post
title: "Basic linear algebra notes"
author: "Rishi Jain"
date: 2026-05-29
categories: notes
math: true
---

These are my notes from working back through the basics of linear algebra, cleaned up enough to share. They follow Zachary Huang's [Give Me 30 min, I will make Linear Algebra Click Forever](https://www.youtube.com/watch?v=5oZ84mlt7tM), roughly one section per idea, with the small numerical examples worked out and a short NumPy snippet for each. Nothing past the foundations here. I wanted them written down, and they might be useful if you're making the same pass.

## Vectors and the dot product

Start with three vectors:

\\[
a = [5,\, 4,\, 1], \qquad b = [4,\, 5,\, 2], \qquad c = [1,\, 2,\, 5]
\\]

Drawn from the origin, they live in 3D space:

![Vectors a, b and c drawn from the origin in 3D space. a and b point in nearly the same direction; c points off on its own.](/assets/lin-alg-vectors-3d.png)

\\( a \\) and \\( b \\) (gold) sit close together; \\( c \\) (blue) heads off on its own. The smaller the angle between two vectors, the more similar they are, and the dot product is what turns that into a number. It has two formulas that describe the same quantity.

**Calculation formula.** Multiply matching components and add them up:

\\[
a \cdot b = a_1 b_1 + a_2 b_2 + a_3 b_3 + \dots + a_n b_n
\\]

**Geometric formula.** Connects the dot product to the angle \\( \theta \\) between the vectors:

\\[
a \cdot b = \lVert a \rVert \, \lVert b \rVert \cos\theta
\\]

where the length (norm) of a vector is

\\[
\lVert a \rVert = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}
\\]

To recover the angle, run both formulas and solve:

1. Get \\( a \cdot b \\) from the calculation formula.
2. Compute the lengths \\( \lVert a \rVert \\) and \\( \lVert b \rVert \\).
3. Rearrange the geometric formula for \\( \theta \\):

\\[
\cos\theta = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}
\\]

Worked out for \\( a \\) and \\( b \\):

\\[
a \cdot b = (5)(4) + (4)(5) + (1)(2) = 20 + 20 + 2 = 42
\\]

\\[
\lVert a \rVert = \sqrt{5^2 + 4^2 + 1^2} = \sqrt{42} \approx 6.48
\\]

\\[
\lVert b \rVert = \sqrt{4^2 + 5^2 + 2^2} = \sqrt{45} \approx 6.71
\\]

\\[
\cos\theta = \frac{42}{6.48 \times 6.71} \approx 0.966
\qquad\Rightarrow\qquad
\theta = \arccos(0.966) \approx 15.0^\circ
\\]

Not much of an angle. \\( a \\) and \\( b \\) are similar, so \\( \theta \\) is small.

Now \\( a \\) and \\( c \\), which look much less alike:

\\[
a \cdot c = (5)(1) + (4)(2) + (1)(5) = 5 + 8 + 5 = 18
\\]

\\[
\lVert a \rVert \approx 6.48, \qquad \lVert c \rVert = \sqrt{30} \approx 5.48
\\]

\\[
\cos\theta = \frac{18}{6.48 \times 5.48} \approx 0.507
\qquad\Rightarrow\qquad
\theta = \arccos(0.507) \approx 59.5^\circ
\\]

This is cosine similarity: similar vectors give a cosine near 1 and a small angle, dissimilar ones give a smaller cosine and a wider angle.

Now the same thing in code:

```python
import numpy as np

a = np.array([5, 4, 1])
b = np.array([4, 5, 2])
c = np.array([1, 2, 5])

dot_ab = np.dot(a, b)
cos_ab = dot_ab / (np.linalg.norm(a) * np.linalg.norm(b))
angle_ab = np.degrees(np.arccos(cos_ab))

dot_ac = np.dot(a, c)
cos_ac = dot_ac / (np.linalg.norm(a) * np.linalg.norm(c))
angle_ac = np.degrees(np.arccos(cos_ac))

print(f"AB angle: {angle_ab:.1f} degrees")   # 15.0
print(f"AC angle: {angle_ac:.1f} degrees")   # 59.5
```

The numbers above are 3D, but the geometry is the same in 2D. Drag either arrow below and watch the dot product, norms, and angle update live: pull them close together and the cosine climbs toward 1, push them apart and it falls.

{% include viz/vectors.html a="4,1" b="1,3" %}

## Linear systems and Gaussian elimination

Instead of writing out linear equations one by one, pack them into \\( Ax = b \\):

\\[
\begin{bmatrix} 300 & 100 \\\\ 100 & 200 \end{bmatrix}
\begin{bmatrix} x \\\\ y \end{bmatrix}
=
\begin{bmatrix} 11000 \\\\ 8000 \end{bmatrix}
\\]

To solve it by hand, use Gaussian elimination: massage \\( A \\) into an upper-triangular shape (row echelon form) using three legal moves.

1. Swap any two rows.
2. Multiply an entire row by a non-zero number.
3. Add a multiple of one row to another row.

Working the system above:

**1. Write the augmented matrix.**

\\[
\left[\begin{array}{cc|c} 300 & 100 & 11000 \\\\ 100 & 200 & 8000 \end{array}\right]
\\]

**2. Simplify (move 2), dividing each row by 100.**

\\[
\left[\begin{array}{cc|c} 3 & 1 & 110 \\\\ 1 & 2 & 80 \end{array}\right]
\\]

**3. Get a 1 in the top-left pivot, by swapping the two rows (move 1).**

\\[
\left[\begin{array}{cc|c} 1 & 2 & 80 \\\\ 3 & 1 & 110 \end{array}\right]
\\]

**4. Eliminate below the pivot (move 3):** \\( R_2 \rightarrow R_2 - 3 R_1 \\).

\\[
\left[\begin{array}{cc|c} 1 & 2 & 80 \\\\ 0 & -5 & -130 \end{array}\right]
\\]

**5. Back-substitute.**

\\[
-5y = -130 \quad\Rightarrow\quad y = 26
\\]

\\[
x + 2y = 80 \quad\Rightarrow\quad x = 28
\\]

NumPy does the whole thing in one call:

```python
import numpy as np

A = np.array([
    [300, 100],
    [100, 200]
])
b = np.array([11000, 8000])

x = np.linalg.solve(A, b)
print(f"x soln: {x[0]}")   # 28.0
print(f"y soln: {x[1]}")   # 26.0
```

## Vector spaces, span, and basis

The standard 2D grid is built from two building-block vectors:

\\[
i = [1,\, 0], \qquad j = [0,\, 1]
\\]

To reach \\( (3, 4) \\), take a scaled sum:

\\[
3i + 4j = (3, 4)
\\]

That's a **linear combination**. Three pieces of vocabulary fall out of it:

1. **Span.** The set of all points you can reach.
2. **Linear independence.** Are any moves redundant? If one vector is just a multiple of another, it's linearly dependent, because it unlocks no new direction.
3. **Basis.** The efficient set of moves: linearly independent (nothing redundant) and spanning the whole space (reaches everywhere).

A few examples in the plane:

1. \\( \\{[1, 0],\, [0, 1]\\} \\). Spans the plane, linearly independent. This is the standard basis (it's \\( i \\) and \\( j \\)).
2. \\( \\{[1, 1],\, [2, 2]\\} \\). Stuck on a line, linearly dependent. Not a basis.
3. \\( \\{[1, 0],\, [0, 1],\, [1, 1]\\} \\). Spans the plane, but not a basis: one vector is redundant, so it isn't efficient.
4. \\( \\{[1, 2],\, [3, 1]\\} \\). Spans the plane and linearly independent. A bit skewed, but a valid basis.

Why does this matter? A basis lets you describe the same data from different points of view. It's the core idea behind JPEG compression and Principal Component Analysis (PCA).

**Change of basis.** Say a point sits at \\( P = [7, 5] \\) in the standard basis, and you want its coordinates in the new, skewed basis \\( b = \\{[1, 2],\, [3, 1]\\} \\). You're looking for \\( c_1 \\) and \\( c_2 \\) with

\\[
c_1 b_1 + c_2 b_2 = P
\\]

which is exactly the \\( Ax = b \\) equation again, with the basis vectors sitting in the columns of the matrix.

```python
import numpy as np

b = np.array([
    [1, 3],
    [2, 1]
])
P = np.array([7, 5])

c = np.linalg.solve(b, P)
print(f"Coordinates of P in basis b: {c}")   # [1.6 1.8]
```

Same point, described in a different basis.

## Linear transformations

A linear transformation is just \\( V^\prime = M V \\), where \\( V^\prime \\) is the transformed vector. The matrix \\( M \\) can rotate, shear, scale, and a lot more.

The useful trick: you only need to watch where the basis vectors \\( i \\) and \\( j \\) land. The columns of the transformation matrix are exactly the coordinates of where the original basis vectors end up.

For a 90° rotation, \\( i \\) lands at \\( [0, 1] \\) and \\( j \\) lands at \\( [-1, 0] \\), which gives

\\[
R = \begin{bmatrix} 0 & -1 \\\\ 1 & 0 \end{bmatrix}
\\]

Take a triangle with vertices \\( P_1 = [1, 1] \\), \\( P_2 = [3, 1] \\), \\( P_3 = [2, 2] \\) and rotate it. Transform each vertex with \\( P^\prime = R P \\):

\\[
P_1^\prime = \begin{bmatrix} -1 \\\\ 1 \end{bmatrix}, \qquad
P_2^\prime = \begin{bmatrix} -1 \\\\ 3 \end{bmatrix}, \qquad
P_3^\prime = \begin{bmatrix} -2 \\\\ 2 \end{bmatrix}
\\]

Plot the before and after and the whole triangle has turned 90°.

```python
import numpy as np

# 90 degree rotation matrix
R = np.array([
    [0, -1],
    [1,  0]
])
P = np.array([
    [1, 3, 2],   # x coords
    [1, 1, 2]    # y coords
])
P_transformed = R @ P   # @ is matrix multiply

print(f"Original:\n{P}")
print(f"Transformed:\n{P_transformed}")
```

## Determinants

The determinant is the area scaling factor of a transformation. For a 2×2 matrix, it's the area of the parallelogram that the unit square gets mapped into:

\\[
\text{unit square, area } 1 \quad\longrightarrow\quad \lvert \det(M) \rvert
\\]

That single number tells you a lot:

- \\( \det(M) = 1 \\): areas are preserved exactly.
- \\( \det(M) = 2 \\): the transformation doubles every area.
- \\( \det(M) < 0 \\): the orientation of space gets flipped.
- \\( \det(M) = 0 \\): area collapses to zero, the matrix has squished the world onto a line or a point. A dimension is lost and the transformation is irreversible. Such a matrix is called singular, or non-invertible.

For a 2×2 matrix the calculation is short:

\\[
A = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix}, \qquad \det(A) = ad - bc
\\]

A few transformations through that lens:

1. The 90° rotation \\( R = \begin{bmatrix} 0 & -1 \\\\ 1 & 0 \end{bmatrix} \\) has \\( \det(R) = 1 \\). Rotation preserves area perfectly.
2. The scaling matrix \\( S = \begin{bmatrix} 2 & 0 \\\\ 0 & 3 \end{bmatrix} \\) has \\( \det(S) = 6 \\). It multiplies every area by 6.
3. The singular matrix \\( C = \begin{bmatrix} 1 & 2 \\\\ 2 & 4 \end{bmatrix} \\) has \\( \det(C) = 0 \\). It collapses area onto a line.

A small diagnostic toolkit:

- \\( \det(A) \neq 0 \\): \\( A \\) is invertible, the transformation can be reversed.
- \\( \det(A) = 0 \\): \\( A \\) is singular, space is squashed to a lower dimension.
- \\( \det(AB) = \det(A)\det(B) \\): the scaling factor of two combined transformations is the product of their individual factors.
- \\( \det(A^{-1}) = \dfrac{1}{\det(A)} \\).

```python
import numpy as np

R = np.array([
    [0, -1],
    [1,  0]
])
det_R = np.linalg.det(R)
print(f"det(R): {det_R}")   # 1.0
```

## What's next

These notes stop at the foundations. The next three on my list:

- Eigenvectors
- PCA
- SVD

They build straight on the dot product, change of basis, and determinant ideas here, which is why I wanted the basics pinned down first.
