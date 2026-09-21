# Cross-track error discussion

Date: 2026-09-18

Transcript of the user questions and assistant answers. Tool output and interim progress messages are omitted. Mathematical and code formatting is normalized for Markdown. Earlier statements are retained as discussed, including subsequent corrections and clarifications.

## User

explain the math behind

```python
n_surf = self.surface_rotation[:, 2]
left_normal = np.cross(n_surf, tangent)  # left of tangent within surface plane
return float(np.dot(pos_xyz - self.path_points[idx], left_normal))
```

## Assistant

This computes the **signed sideways offset from the path**, measured within the surface plane.

Let $\mathbf p$ be `pos_xyz`, $\mathbf q$ be `self.path_points[idx]`, and $\mathbf t$ be `tangent`.

1. **Get the surface normal**

   ```python
   n_surf = self.surface_rotation[:, 2]
   ```

   This extracts the third column of the rotation matrix. Assuming the matrix maps surface coordinates into world coordinates, that column is the surface’s local $z$-axis expressed in world coordinates—the vector perpendicular to the surface.

2. **Find the direction to the left of the path**

   ```python
   left_normal = np.cross(n_surf, tangent)
   ```

   The cross product

   $$\mathbf l = \mathbf n \times \mathbf t$$

   is perpendicular to both the surface normal and the tangent. Thus, it lies in the surface plane and points sideways across the path.

   For example, on a horizontal surface, with the path pointing along $+x$:

   $$\mathbf n=(0,0,1),\qquad \mathbf t=(1,0,0)
   \quad\Longrightarrow\quad \mathbf l=(0,1,0).$$

   That is left when looking forward along the path with $+z$ up. Reversing the cross-product order would point right.

3. **Project the position offset onto that left direction**

   ```python
   return float(np.dot(pos_xyz - self.path_points[idx], left_normal))
   ```

   Mathematically:

   $$\boxed{e=(\mathbf p-\mathbf q)\cdot(\mathbf n\times\mathbf t)}$$

   - $e>0$: position is left of the path.
   - $e<0$: position is right of the path.
   - $e=0$: no sideways offset.

**For this to be a distance**, `n_surf` and `tangent` must be perpendicular unit vectors, making `left_normal` a unit vector too. The projection ignores displacement along the path and perpendicular to the surface. It measures offset from the **local tangent line** at `idx`; for a curved path, this is not necessarily the shortest distance to the entire path.

I couldn’t access the surrounding code, so the rotation convention and normalization assumptions above remain unverified.

## User

provide an external refernce (web link)

## Assistant

These two sections from **OpenStax, Calculus Volume 3** explain the math directly:

- [The Cross Product](https://openstax.org/books/calculus-volume-3/pages/2-4-the-cross-product) — explains perpendicular vectors and the right-hand rule behind `np.cross(n_surf, tangent)`.
- [The Dot Product — Vector Projections](https://openstax.org/books/calculus-volume-3/pages/2-3-the-dot-product) — explains projecting displacement onto a direction. When `left_normal` has unit length, `(pos_xyz - path_point) · left_normal` gives the **signed distance in that direction**.

## User

link to a source were you got the cross tracking error not the base linear algebra operations

## Assistant

A direct reference is [**Stanley: The Robot That Won the DARPA Grand Challenge** — Thrun et al.](https://robots.stanford.edu/papers/thrun.stanley05.pdf). See **Figure 24 and the accompanying steering-control discussion**, which defines cross-track error as the lateral distance between the vehicle’s front-wheel center and the nearest point on the trajectory.

To clarify provenance: my earlier explanation was derived from your code, not taken from that paper. Your code expresses the same lateral-offset concept within a surface plane:

$$e=(\mathbf p-\mathbf q)\cdot(\mathbf n_{\text{surf}}\times\mathbf t).$$

The surface-normal formulation and positive-left convention are interpretations of your implementation, not formulas attributed to the Stanley paper.

## User

provide a source that computes the cross tracking error in the same way as the code

## Assistant

Yes—**Fossen and Pettersen (2014), [“On Uniform Semiglobal Exponential Stability (USGES) of Proportional Line-of-Sight Guidance Laws”](https://www.fossen.biz/publications/2014%20Fossen%20and%20Pettersen%20Automatica.pdf#page=2)**, Section 1.1, **Equation (4)**, computes cross-track error as:

$$y_e=-(x-x_p)\sin\gamma_p+(y-y_p)\cos\gamma_p,$$

where $\gamma_p$ is the path tangent angle.

This is **algebraically equivalent to your code in surface coordinates**. To see the correspondence, take:

$$\mathbf n=(0,0,1),\qquad
\mathbf t=(\cos\gamma_p,\sin\gamma_p,0).$$

Then your cross product gives:

$$\mathbf n\times\mathbf t=(-\sin\gamma_p,\cos\gamma_p,0),$$

and your dot product gives:

$$
(\mathbf p-\mathbf p_{\text{path}})\cdot(\mathbf n\times\mathbf t)
=-(x-x_p)\sin\gamma_p+(y-y_p)\cos\gamma_p.
$$

The paper supplies the planar formula; your code expresses that same calculation for a rotated surface plane, assuming unit normal and unit tangent vectors perpendicular to each other.

## User

what does `self.surface_rotation[:, 2]` do?

## Assistant

`self.surface_rotation[:, 2]` selects **all rows of column index 2**—the third column, because Python indexes from zero.

For example:

```python
self.surface_rotation = np.array([
    [a, b, c],
    [d, e, f],
    [g, h, i],
])

self.surface_rotation[:, 2]  # array([c, f, i])
```

Here:

- `self.surface_rotation` is the rotation matrix stored on this object.
- `:` means “take every row.”
- `2` means “take the third column.”

If this matrix rotates vectors **from surface coordinates to world coordinates**, its third column is the surface’s local z-axis expressed in world coordinates. Since the local z-axis is perpendicular to the surface, this column gives the **surface normal**:

$$\mathbf n_{\text{surf}}
=R\begin{bmatrix}0\\0\\1\end{bmatrix}
=R[:,2].$$

## User

you math appear wrong and it pains me you can't see that

## Assistant

I may have assumed the wrong rotation convention. I should have verified how `surface_rotation` is defined before interpreting its columns.

**The indexing is certain:** `[:, 2]` extracts the third column.

**Its meaning depends on the matrix:**

- If $v_{\text{world}}=R\,v_{\text{surface}}$, the surface normal in world coordinates is `R[:, 2]`.
- If $v_{\text{surface}}=R\,v_{\text{world}}$, it is `R[2, :]` instead—the third row.

The identity $R[:,2]=R(0,0,1)^T$ is correct, but calling that vector the world-space surface normal requires the first convention.

Can you paste where `surface_rotation` is constructed, or point to the equation you’re challenging? That will let me address the actual error instead of guessing.

## User

tangent should not be involved in the math buddy

## Assistant

For **computing the surface normal**, you’re right: the path tangent is not involved.

```python
n_surf = self.surface_rotation[:, 2]
```

This simply extracts the third column. If the rotation maps surface coordinates to world coordinates, that column is the surface normal.

The tangent enters only in the **next calculation**:

```python
left_normal = np.cross(n_surf, tangent)
```

That computes a different vector: the sideways direction **within** the surface, perpendicular to the path. Cross-track error needs the path direction to define “sideways”; the surface normal alone cannot define it.

## User

the whole point of the transpose in the ref paper is that it gets you the left normal

## Assistant

Yes—you’re right. In the paper, transposing the **path-frame rotation matrix** puts the left normal in its second row:

$$R=
\begin{bmatrix}
\cos\gamma & -\sin\gamma\\
\sin\gamma & \cos\gamma
\end{bmatrix},
\qquad
R^T=
\begin{bmatrix}
\cos\gamma & \sin\gamma\\
-\sin\gamma & \cos\gamma
\end{bmatrix}.$$

Multiplying $R^T(\mathbf p-\mathbf p_{\text{path}})$ directly gives the along-track and cross-track components. **No explicit tangent or cross product is needed** because the path orientation is already encoded in $R$.

I conflated that **path-frame rotation** with your **surface rotation**. If `surface_rotation` also has its local x-axis aligned with the path, then its second column already provides the left normal:

```python
left_normal = self.surface_rotation[:, 1]
```

If it describes only the surface’s orientation, however, it does not necessarily encode the path’s left direction. That distinction is what I should have made clear.

## User

the surface orientation encodes no path information

## Assistant

Correct. Then these are **two different rotation matrices**:

- `surface_rotation` describes the surface orientation. Its third column supplies the surface normal.
- The paper’s $R$ describes the **path frame**. Its second column supplies the path’s left normal; transposing it makes that normal a row for projecting the position error.

Your code constructs the missing path-relative direction explicitly:

$$\mathbf l=\mathbf n_{\text{surface}}\times\mathbf t_{\text{path}},
\qquad
e=\mathbf l^T(\mathbf p-\mathbf p_{\text{path}}).$$

That is equivalent to taking the lateral component using the paper’s path-frame transpose. **The tangent is explicit in your code and encoded in the paper’s rotation matrix.** My mistake was conflating the surface frame and path frame.

## User

why do we need the surface normal provide a source where they do the same

## Assistant

The surface normal tells the code **which plane “sideways” must lie in**. In 3D, many directions are perpendicular to the path tangent. Using

$$\mathbf l=\mathbf n_{\text{surf}}\times\mathbf t$$

selects the one that also lies within the surface plane.

An explicit source for **this exact construction** is [Cornell’s *Curves on Surfaces*, Section 1: The Darboux Frame](https://e.math.cornell.edu/people/belk/differentialgeometry/Outline%20-%20Curves%20on%20Surfaces.pdf). It defines:

> “The third vector $\vec U=\vec N\times\vec T$, which is tangent to the surface and normal to the curve.”

Here, $\vec N$ is the surface normal and $\vec T$ is the curve tangent—exactly your `np.cross(n_surf, tangent)`.

**You don’t need the surface normal separately if you already have the complete path-frame rotation.** Its lateral axis already contains that information. Your code constructs that axis because the surface rotation alone contains no path direction.

This source supports the frame construction; it does not itself present a cross-track-error algorithm. Projecting the position offset onto that constructed axis is the additional step in your code.

## User

here is a gotcha the cost waypoint is not the closest point on the path. Should we not use a vector to find the closest point between the two waypoints?

## Assistant

Yes—**if you want error relative to the closest point on the actual path**, you should project the vehicle position onto the segment between waypoints, rather than assume a selected waypoint is the closest point.

For segment endpoints $\mathbf a,\mathbf b$ and position $\mathbf p$:

$$\mathbf d=\mathbf b-\mathbf a,\qquad
u=\operatorname{clip}\left(
\frac{(\mathbf p-\mathbf a)\cdot\mathbf d}{\mathbf d\cdot\mathbf d},
0,1
\right),
\qquad
\mathbf q=\mathbf a+u\mathbf d.$$

Then $\mathbf q$ is the closest point **on that segment**. Clipping keeps it between the endpoints.

```python
d = b - a
length_sq = np.dot(d, d)  # Handle duplicate waypoints separately.
u = np.clip(np.dot(pos_xyz - a, d) / length_sq, 0.0, 1.0)
closest = a + u * d

tangent = d / np.sqrt(length_sq)
left_normal = np.cross(n_surf, tangent)
cross_track_error = np.dot(pos_xyz - closest, left_normal)
```

One subtlety: **using the segment’s endpoint instead of `closest` gives the same lateral projection**, provided the tangent is that segment’s direction:

$$(\mathbf p-\mathbf q)\cdot\mathbf l
=(\mathbf p-\mathbf a)\cdot\mathbf l,$$

because $\mathbf q-\mathbf a$ runs along the segment and is perpendicular to $\mathbf l$.

So the potential problem is **choosing the wrong segment or tangent**, not merely subtracting a waypoint. To find the closest point on a polyline, compute these projections for the candidate segments and select the smallest distance $\|\mathbf p-\mathbf q\|$. At an endpoint, that distance can include an along-track component, so it is not necessarily the absolute lateral error.

## User

can you save this chat as a marksdown file?
