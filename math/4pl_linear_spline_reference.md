# Mathematical reference for `_eval_4pl`

The function evaluates a continuous piecewise-linear spline with three knots and subtracts it from a constant threshold:

```python
def _eval_4pl(self, ang):
    k1, k2, k3 = self._4pl_knots
    c = self._4pl_coeffs
    T = (c[0]*ang
         + c[1]*max(ang-k1, 0)
         + c[2]*max(ang-k2, 0)
         + c[3]*max(ang-k3, 0))
    return self._4pl_threshold - T
```

## Spline representation

For fixed, distinct knots $k_1 < k_2 < k_3$, every continuous piecewise-linear function with possible slope changes only at those knots can be written as

$$
s(x) = a + bx + \sum_{j=1}^{3} d_j(x-k_j)_+,
\qquad (u)_+ = \max(u,0).
$$

This is a degree-one (order-two) spline in the truncated-power basis:

$$
\{1,\ x,\ (x-k_1)_+,\ (x-k_2)_+,\ (x-k_3)_+\}.
$$

In the code, the returned function corresponds to

$$
x = \mathrm{ang},\qquad
a = \texttt{self.\_4pl\_threshold},\qquad
b = -c[0],\qquad d_j = -c[j].
$$

The threshold supplies the constant term in this basis. It equals the value at zero if all knots are nonnegative; otherwise, active hinge terms also contribute at zero.

## Why the coefficients represent slope changes

Away from the knots,

$$
s'(x) = b + \sum_{j=1}^{3} d_j\mathbf{1}_{\{x>k_j\}}.
$$

The derivative is piecewise constant, with jump

$$
s'(k_j^+) - s'(k_j^-) = d_j.
$$

Therefore, for desired segment slopes $m_0,m_1,m_2,m_3$, choose

$$
b = m_0,\qquad d_j = m_j-m_{j-1}.
$$

This also proves the representation: these choices reproduce the slope on every interval, and choosing the constant term to match one function value reproduces the entire continuous function.

For the implementation, the slopes are:

| Angle interval | Slope of `T` | Slope of returned value |
|---|---|---|
| $x<k_1$ | $c_0$ | $-c_0$ |
| $k_1<x<k_2$ | $c_0+c_1$ | $-(c_0+c_1)$ |
| $k_2<x<k_3$ | $c_0+c_1+c_2$ | $-(c_0+c_1+c_2)$ |
| $x>k_3$ | $c_0+c_1+c_2+c_3$ | $-(c_0+c_1+c_2+c_3)$ |

Here $c_j$ denotes `c[j]`. Each hinge is zero at its knot, ensuring continuity. The derivative need not exist at a knot with a nonzero slope change. A zero coefficient leaves adjacent segments with the same slope.

The returned value is a signed difference from the threshold: positive below it, zero at equality, and negative above it. The function does not clamp the result or perform a Boolean comparison.

## Mathematical reference

Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Second edition. Springer. Section 5.2, "Piecewise Polynomials and Splines," pp. 141–144.

- [Authors' website and free book download](https://hastie.su.domains/ElemStatLearn/main.html)
- [Authors' table of contents](https://hastie.su.domains/ElemStatLearn/contents.pdf)
- [Supplementary reference: SAS documentation on the truncated-power basis](https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_introcom_sect022.htm)

The formula above specializes the truncated-power spline representation to degree one and maps it to the supplied implementation.

Suggested wording for a report:

> The angular response is represented by a continuous linear spline with three interior knots, expressed in the truncated-power basis (Hastie, Tibshirani, and Friedman, 2009, Section 5.2).
