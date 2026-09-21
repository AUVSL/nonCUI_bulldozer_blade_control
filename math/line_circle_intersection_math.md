# Lookahead point: line–circle intersection

This code finds a point on a path segment that is exactly distance $L$ from a reference position—typically the vehicle’s position. Geometrically, it finds where a **line segment intersects a circle of radius $L$**.

```python
b = 2.0 * float(np.dot(f, d))
c = float(np.dot(f, f)) - L * L
disc = b * b - 4 * a * c
if disc < 0:
    continue
t2 = (-b + np.sqrt(disc)) / (2 * a)  # forward intersection
if 0.0 <= t2 <= 1.0:
    lookahead_pt = p1 + t2 * d
    break
```

The derivation assumes the preceding code defines:

```python
d = p2 - p1
f = p1 - position
a = float(np.dot(d, d))
```

The derivation depends on those definitions, particularly the direction of `f`. The surrounding project code was not verified.

## 1. Describe points along the path segment

Let:

- $\mathbf p_1,\mathbf p_2$: the segment’s endpoints.
- $\mathbf q$: the vehicle’s position, which is the circle’s center.
- $L$: the lookahead distance, which is the circle’s radius.
- $\mathbf d=\mathbf p_2-\mathbf p_1$: the displacement along the segment.
- $\mathbf f=\mathbf p_1-\mathbf q$: the displacement from the vehicle to the segment’s start.

A point along the line is

$$
\mathbf p(t)=\mathbf p_1+t\mathbf d.
$$

Here $t$ is a fraction of the segment:

$$
\mathbf p(0)=\mathbf p_1,\qquad
\mathbf p(1)=\mathbf p_2,\qquad
\mathbf p(0.5)=\frac{\mathbf p_1+\mathbf p_2}{2}.
$$

Therefore, a point lies **on the segment** when $0\le t\le1$.

Substituting a parametric line into a circle or sphere’s equation is a standard intersection method; Scratchapixel explains the same construction for a sphere. In two dimensions, it gives the circle calculation used here. [Source: Scratchapixel, “Ray-Sphere Intersection,” analytical solution](https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html).

## 2. Require the point to be distance L from the vehicle

The circle equation is

$$
\|\mathbf p(t)-\mathbf q\|^2=L^2.
$$

Substitute the line equation:

$$
\|\mathbf p_1+t\mathbf d-\mathbf q\|^2=L^2.
$$

Since $\mathbf f=\mathbf p_1-\mathbf q$,

$$
\|\mathbf f+t\mathbf d\|^2=L^2.
$$

A vector’s squared length equals its dot product with itself:

$$
(\mathbf f+t\mathbf d)\cdot(\mathbf f+t\mathbf d)=L^2.
$$

Expand every term:

$$
\mathbf f\cdot\mathbf f
+\mathbf f\cdot(t\mathbf d)
+(t\mathbf d)\cdot\mathbf f
+(t\mathbf d)\cdot(t\mathbf d)
=L^2.
$$

Pull the scalar $t$ outside the dot products:

$$
\mathbf f\cdot\mathbf f
+t(\mathbf f\cdot\mathbf d)
+t(\mathbf d\cdot\mathbf f)
+t^2(\mathbf d\cdot\mathbf d)
=L^2.
$$

Because $\mathbf f\cdot\mathbf d=\mathbf d\cdot\mathbf f$, combine the middle terms:

$$
\mathbf f\cdot\mathbf f
+2t(\mathbf f\cdot\mathbf d)
+t^2(\mathbf d\cdot\mathbf d)
=L^2.
$$

Move $L^2$ to the left and arrange by powers of $t$:

$$
\boxed{
(\mathbf d\cdot\mathbf d)t^2
+2(\mathbf f\cdot\mathbf d)t
+(\mathbf f\cdot\mathbf f-L^2)=0
}.
$$

This has the familiar quadratic form

$$
at^2+bt+c=0,
$$

with

$$
\boxed{
a=\mathbf d\cdot\mathbf d,\qquad
b=2(\mathbf f\cdot\mathbf d),\qquad
c=\mathbf f\cdot\mathbf f-L^2.
}
$$

That explains:

```python
b = 2.0 * float(np.dot(f, d))
c = float(np.dot(f, f)) - L * L
```

For two-dimensional vectors, these mean

$$
a=d_x^2+d_y^2,
$$

$$
b=2(f_xd_x+f_yd_y),
$$

$$
c=f_x^2+f_y^2-L^2.
$$

`float(...)` converts the numerical result to a Python floating-point number; it does not change the underlying formula.

## 3. Solve for t, showing the algebra

For a nonzero-length segment, $a=\|\mathbf d\|^2>0$. Start with

$$
at^2+bt+c=0.
$$

Multiply by $4a$:

$$
4a^2t^2+4abt+4ac=0.
$$

Move the constant term:

$$
4a^2t^2+4abt=-4ac.
$$

Add $b^2$ to both sides:

$$
4a^2t^2+4abt+b^2=b^2-4ac.
$$

Factor the left-hand side:

$$
(2at+b)^2=b^2-4ac.
$$

Take square roots:

$$
2at+b=\pm\sqrt{b^2-4ac}.
$$

Subtract $b$, then divide by $2a$:

$$
\boxed{t=\frac{-b\pm\sqrt{b^2-4ac}}{2a}}.
$$

This is the quadratic formula, obtained by completing the square. [Source: Wolfram MathWorld, “Quadratic Equation”](https://mathworld.wolfram.com/QuadraticEquation.html).

The expression under the square root is the **discriminant**:

$$
\Delta=b^2-4ac.
$$

For the infinite line:

| Discriminant | Geometry |
|---|---|
| $\Delta<0$ | No real intersection |
| $\Delta=0$ | One intersection: the line touches the circle |
| $\Delta>0$ | Two distinct intersections |

Thus:

```python
disc = b * b - 4 * a * c
if disc < 0:
    continue
```

skips the current segment when its line misses the circle. Real roots must still pass the segment bounds check. [Source: Scratchapixel, intersection cases](https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html).

## 4. Choose an intersection and reconstruct its position

The two roots are

$$
t_1=\frac{-b-\sqrt{\Delta}}{2a},
\qquad
t_2=\frac{-b+\sqrt{\Delta}}{2a}.
$$

Because $a>0$,

$$
t_2-t_1=\frac{\sqrt{\Delta}}{a}\ge0.
$$

So this line chooses the intersection farther along the direction from $\mathbf p_1$ toward $\mathbf p_2$:

```python
t2 = (-b + np.sqrt(disc)) / (2 * a)
```

The comment `# forward intersection` means **farther along that line direction**. Whether this is ahead of the vehicle depends on how the path is ordered and which segments are searched.

Finally:

```python
if 0.0 <= t2 <= 1.0:
    lookahead_pt = p1 + t2 * d
    break
```

checks that this intersection lies on the segment, computes its coordinates, and exits the enclosing loop.

## 5. Numerical example

Suppose

$$
\mathbf q=(0,0),\quad
L=5,\quad
\mathbf p_1=(-6,3),\quad
\mathbf p_2=(6,3).
$$

Then

$$
\mathbf d=(6,3)-(-6,3)=(12,0),
\qquad
\mathbf f=(-6,3)-(0,0)=(-6,3).
$$

Calculate the coefficients:

$$
a=12^2+0^2=144,
$$

$$
b=2[(-6)(12)+(3)(0)]=-144,
$$

$$
c=(-6)^2+3^2-5^2=36+9-25=20.
$$

The quadratic is

$$
144t^2-144t+20=0.
$$

Its discriminant is

$$
\Delta=(-144)^2-4(144)(20)
=20736-11520
=9216,
$$

so

$$
\sqrt{\Delta}=96.
$$

The selected root is

$$
t_2=\frac{144+96}{288}
=\frac{240}{288}
=\frac56.
$$

Since $0\le\frac56\le1$, the code accepts it:

$$
\begin{aligned}
\mathbf p(t_2)
&=(-6,3)+\frac56(12,0)\\
&=(-6,3)+(10,0)\\
&=\boxed{(4,3)}.
\end{aligned}
$$

Check its distance from the vehicle:

$$
\sqrt{4^2+3^2}
=\sqrt{25}
=5=L.
$$

Two implementation details matter: a zero-length segment requires handling before dividing by $2a$, and checking only $t_2$ can miss a valid $t_1$ intersection when $t_2>1$. Whether ignoring that earlier intersection is intentional depends on the lookahead algorithm.

The sources above are online technical references, not textbooks, so they have no textbook page numbers.
