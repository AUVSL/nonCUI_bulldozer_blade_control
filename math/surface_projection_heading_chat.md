# Surface projection and heading — chat transcript

Date: 2026-09-18

This transcript records the user's mathematical questions and the assistant's answers. Tool output and progress updates are omitted. Escaped underscores and formatting in the original questions have been normalized for readability.

## User

Explain the math here. Also provide a reference or two for this:

```python
d_vec = lookahead_pt - pos
d_proj = d_vec - np.dot(d_vec, n_surf) * n_surf  # project onto surface plane
angle = np.arctan2(np.dot(d_proj, e2_surf), np.dot(d_proj, e1_surf))
fwd = np.asarray(body_rotation)[:, 0]
heading = np.arctan2(np.dot(fwd, e2_surf), np.dot(fwd, e1_surf))
```

## Assistant

This code computes **the direction toward the lookahead point and the vehicle’s current heading, both measured in the local surface plane**. That lets you compare directions on sloped terrain.

It assumes:

- `n_surf` is a **unit vector** perpendicular to the surface.
- `e1_surf` and `e2_surf` are perpendicular unit vectors lying in that plane—its local x- and y-axes.
- All vectors use the same coordinate frame.

**1. Find the displacement toward the target**

```python
d_vec = lookahead_pt - pos
```

Mathematically,

$$
\mathbf d=\mathbf p_{\text{target}}-\mathbf p_{\text{vehicle}}.
$$

This is a 3D vector pointing from the vehicle to the lookahead point.

**2. Remove the component perpendicular to the surface**

```python
d_proj = d_vec - np.dot(d_vec, n_surf) * n_surf
```

For a unit normal n, the dot product d · n gives the signed amount of displacement along that normal. Multiplying by n turns it into a vector, which is subtracted:

$$
\boxed{\mathbf d_{\text{proj}}=\mathbf d-(\mathbf d\cdot\mathbf n)\mathbf n.}
$$

The result lies parallel to the surface because

$$
\mathbf d_{\text{proj}}\cdot\mathbf n
=\mathbf d\cdot\mathbf n-(\mathbf d\cdot\mathbf n)\underbrace{(\mathbf n\cdot\mathbf n)}_{1}=0.
$$

This is orthogonal projection onto a plane; see [Trinity College Dublin’s notes, page 2](https://www.maths.tcd.ie/~odunlain/u11s02/online_notes/pdf_linear1.pdf). If the normal is not unit length, divide the subtracted term by n · n.

**3. Convert the projected direction into an angle**

```python
angle = np.arctan2(
    np.dot(d_proj, e2_surf),
    np.dot(d_proj, e1_surf)
)
```

The dot products give the vector’s coordinates along the two surface axes:

$$
x_s=\mathbf d_{\text{proj}}\cdot\mathbf e_1,\qquad
y_s=\mathbf d_{\text{proj}}\cdot\mathbf e_2.
$$

Then

$$
\text{angle}=\operatorname{atan2}(y_s,x_s).
$$

This measures the target direction from +e1, with positive angles toward +e2. `atan2` uses both signs to select the correct quadrant and returns radians in [-π, π]. Notice the argument order: **y first, x second**. [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html)

**4. Extract the vehicle’s forward direction**

```python
fwd = np.asarray(body_rotation)[:, 0]
```

This selects the first column of the rotation matrix. **If `body_rotation` maps body coordinates to world coordinates and the body’s local x-axis points forward**, then

$$
\mathbf f=R\begin{bmatrix}1\\0\\0\end{bmatrix}=R[:,0]
$$

is the forward direction in world coordinates.

**5. Measure that forward direction in the same surface plane**

```python
heading = np.arctan2(
    np.dot(fwd, e2_surf),
    np.dot(fwd, e1_surf)
)
```

This applies the same angle calculation to the vehicle’s forward vector.

An explicit projection of `fwd` is unnecessary: the surface axes are perpendicular to the normal, so dotting with them already ignores the normal component:

$$
[\mathbf f-(\mathbf f\cdot\mathbf n)\mathbf n]\cdot\mathbf e_i
=\mathbf f\cdot\mathbf e_i.
$$

For example, if the projected target direction has surface coordinates (1,1), `angle` is 45°. If the vehicle points along `e1_surf`, `heading` is 0°. The required heading change is +45°.

You can compute the heading error, wrapped to [-π, π), with:

```python
heading_error = (angle - heading + np.pi) % (2 * np.pi) - np.pi
```

If either vector has zero length **within the surface plane**, its direction is mathematically undefined; near zero, the calculated angle can be unstable.

## User

Are there sources that compute the 3D heading in the same way?

## Assistant

Yes—**Pololu’s tilt-compensated compass code uses exactly the same dot-product-plus-`atan2` calculation.** ESA also provides the corresponding mathematical derivation.

**1. Pololu: a direct implementation match**

In [Pololu’s LSM303 compass implementation, line 246](https://pololu.github.io/zumo-shield-arduino-library/_l_s_m303_8h_source.html#l00246), heading is computed as:

```cpp
float heading = atan2(vector_dot(&E, &from),
                      vector_dot(&N, &from)) * 180 / PI;
```

The correspondence to your code is:

| Pololu | Your code | Meaning |
|---|---|---|
| `from` | `fwd` | 3D direction whose heading is measured |
| `N` | `e1_surf` | Reference axis for zero heading |
| `E` | `e2_surf` | Axis toward positive heading |

Their comments explicitly describe constructing a basis for the horizontal plane and measuring the angle of the projected direction. Your code applies that same geometry to the **local terrain plane**, using its two tangent axes instead of north and east.

**2. ESA Navipedia: the mathematical formulation**

[ESA’s “Transformations between ECEF and ENU coordinates,” equations (9)–(11)](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates#Elevation_and_azimuth_computation) computes a 3D line-of-sight vector’s azimuth using its dot products with local east and north:

$$
\hat{\boldsymbol\rho}\cdot\hat{\mathbf e}=\cos E\sin A,
\qquad
\hat{\boldsymbol\rho}\cdot\hat{\mathbf n}=\cos E\cos A.
$$

These relations give the quadrant-aware implementation:

$$
A=\operatorname{atan2}
\left(\hat{\boldsymbol\rho}\cdot\hat{\mathbf e},
\hat{\boldsymbol\rho}\cdot\hat{\mathbf n}\right).
$$

ESA prints this as an arctangent of the ratio; the `atan2` expression above follows from their sine and cosine relations and preserves the quadrant.

In your notation, this becomes exactly:

$$
\boxed{\text{heading}=\operatorname{atan2}
(\mathbf{fwd}\cdot\mathbf e_{2,\text{surf}},
\mathbf{fwd}\cdot\mathbf e_{1,\text{surf}})}
$$

The precise term for your result is **heading relative to the local surface plane**: one angle derived from a 3D direction, with the plane and reference axis specifying what “heading” means.

## User

Provide a better source for:

```python
d_proj = d_vec - np.dot(d_vec, n_surf) * n_surf  # project onto surface plane
```

## Assistant

A more direct academic source is **Jerry L. Kazdan’s [Linear Algebra Problems, Math 504–505, University of Pennsylvania](https://www2.math.upenn.edu/~kazdan/504/la.pdf)**. It explicitly gives the projection onto the plane perpendicular to a unit normal n:

$$
P\mathbf x=\mathbf x-\langle\mathbf x,\mathbf n\rangle\mathbf n
=(I-\mathbf n\mathbf n^\mathsf T)\mathbf x.
$$

This matches your line exactly:

```python
d_proj = d_vec - np.dot(d_vec, n_surf) * n_surf
```

Here, x is `d_vec`, n is `n_surf`, and the inner product is `np.dot(d_vec, n_surf)`. The subtraction removes the displacement perpendicular to the surface, leaving the displacement parallel to it.

**The normal must have length 1.** For a non-unit normal, use:

```python
d_proj = d_vec - (
    np.dot(d_vec, n_surf) / np.dot(n_surf, n_surf)
) * n_surf
```

Because `d_vec = lookahead_pt - pos` is already a displacement, this projects relative to the plane through `pos`; the projected target point would be `pos + d_proj`.

## User

Provide a source for:

```python
angle = np.arctan2(np.dot(d_proj, e2_surf), np.dot(d_proj, e1_surf))
```

## Assistant

A direct source is **[Pololu’s LSM303 implementation, line 246](https://pololu.github.io/zumo-shield-arduino-library/_l_s_m303_8h_source.html#l00246)**, which computes:

```cpp
atan2(vector_dot(&E, &from), vector_dot(&N, &from))
```

This matches your expression with `from = d_proj`, `E = e2_surf`, and `N = e1_surf`. Pololu uses horizontal east/north axes; your code uses the local surface axes.

Mathematically, for perpendicular unit surface axes and a nonzero projected vector:

$$
\mathbf d_{\rm proj}=r\cos\theta\,\mathbf e_1+r\sin\theta\,\mathbf e_2.
$$

Taking dot products extracts the coordinates:

$$
\mathbf d_{\rm proj}\cdot\mathbf e_1=r\cos\theta,
\qquad
\mathbf d_{\rm proj}\cdot\mathbf e_2=r\sin\theta.
$$

Therefore,

$$
\boxed{\theta=\operatorname{atan2}
\left(\mathbf d_{\rm proj}\cdot\mathbf e_2,
\mathbf d_{\rm proj}\cdot\mathbf e_1\right)}
$$

The [NumPy `arctan2` documentation](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html) confirms the argument order—**y, then x**—and the signed result in radians.

## User

Code is not a good source.

## Assistant

Use this mathematical reference:

**J. Sanz Subirana, J. M. Juan Zornoza, and M. Hernández-Pajares (2011), [“Transformations between ECEF and ENU coordinates,” ESA Navipedia, equations (9)–(11)](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates#Elevation_and_azimuth_computation).**

Equation (11) expresses the azimuth of a 3D direction using dot products with two tangent-plane axes:

$$
A=\arctan\left(
\frac{\hat{\boldsymbol\rho}\cdot\hat{\mathbf e}}
{\hat{\boldsymbol\rho}\cdot\hat{\mathbf n}}
\right),
$$

where e and n are east and north unit vectors.

Your formula applies this construction with:

$$
\hat{\mathbf e}\rightarrow\mathbf e_{2,\mathrm{surf}},
\qquad
\hat{\mathbf n}\rightarrow\mathbf e_{1,\mathrm{surf}},
\qquad
\hat{\boldsymbol\rho}\rightarrow
\frac{\mathbf d_{\mathrm{proj}}}{\|\mathbf d_{\mathrm{proj}}\|}.
$$

The common normalization cancels, giving:

$$
\boxed{\theta=\operatorname{atan2}
\left(\mathbf d_{\mathrm{proj}}\cdot\mathbf e_{2,\mathrm{surf}},
\mathbf d_{\mathrm{proj}}\cdot\mathbf e_{1,\mathrm{surf}}\right).}
$$

**The source prints an arctangent of a ratio, rather than `atan2`.** Your implementation uses the quadrant-aware form, which follows from the sine and cosine relations in their equation (9). Thus, this reference supports the mathematical construction, with that implementation distinction made explicit.

## User

Give me a source for this that isn't from homework problems:

```python
d_proj = d_vec - np.dot(d_vec, n_surf) * n_surf
```

## Assistant

A published research source is **Yang, Belongie, Hariharan, and Koltun (2021), *Geometry Processing with Neural Fields*, NeurIPS**.

Their [supplementary material, Theorem 2.1, page 2](https://proceedings.neurips.cc/paper/2021/file/bd686fd640be98efaae0091fa301e613-Supplemental.pdf#page=2) explicitly states:

> “The matrix P = I − nnᵀ projects vectors onto the tangent plane.”

Applying that matrix to your displacement gives:

$$
\mathbf d_{\mathrm{proj}}=P\mathbf d
=(I-\mathbf n\mathbf n^\mathsf T)\mathbf d
=\mathbf d-(\mathbf d\cdot\mathbf n)\mathbf n,
$$

which is exactly:

```python
d_proj = d_vec - np.dot(d_vec, n_surf) * n_surf
```

Here, `n_surf` must be a **unit surface normal**. The theorem also provides a proof that the result lies in the tangent plane and is the closest vector in that plane to the original displacement.

## User

Provide a source for `np.dot(d_vec, n_surf) * n_surf = n_surf n_surf^T d_vec`.

## Assistant

**Sam Buss’s textbook *3-D Computer Graphics: A Mathematical Introduction with OpenGL* ([second-edition draft](https://mathweb.ucsd.edu/~sbuss/CourseWeb/Math155A_2019Winter/SecondEdDraft.pdf)) explicitly gives this identity:**

$$
(\mathbf u\cdot\mathbf v)\mathbf u
=\mathbf u(\mathbf u\cdot\mathbf v)
=\mathbf u(\mathbf u^\mathsf T\mathbf v)
=(\mathbf u\mathbf u^\mathsf T)\mathbf v.
$$

Substitute u = `n_surf` and v = `d_vec`. Since the real dot product is symmetric,

$$
\boxed{(\mathbf d\cdot\mathbf n)\mathbf n
=\mathbf n(\mathbf n^\mathsf T\mathbf d)
=(\mathbf n\mathbf n^\mathsf T)\mathbf d.}
$$

The equality follows from the dot product’s matrix representation and associativity of matrix multiplication. **It holds for any real vectors of matching size**; unit length is required only to interpret this expression as an orthogonal projection.

## User

On what page?

## Assistant

**Printed page 465** of [Buss’s second-edition draft](https://mathweb.ucsd.edu/~sbuss/CourseWeb/Math155A_2019Winter/SecondEdDraft.pdf), immediately before Section **A.3.2, “Determinants, inverses, and adjoints.”**

It gives:

$$
(u\cdot v)u=u(u\cdot v)=u(u^Tv)=(uu^T)v.
$$

Use the page number printed on the page; your PDF viewer’s page count may differ.

## User

Shouldn't it be n dot d?

## Assistant

Yes—n · d matches nᵀd directly. But for real vectors, the dot product is **commutative**:

$$
\mathbf d\cdot\mathbf n=\mathbf n\cdot\mathbf d.
$$

Therefore,

$$
(\mathbf d\cdot\mathbf n)\mathbf n
=(\mathbf n\cdot\mathbf d)\mathbf n
=\mathbf n(\mathbf n^\mathsf T\mathbf d)
=(\mathbf n\mathbf n^\mathsf T)\mathbf d.
$$

Both `np.dot(d_vec, n_surf)` and `np.dot(n_surf, d_vec)` give the same scalar here.

## User

Save this chat to a Markdown and put it in the math folder.
