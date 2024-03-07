
### Step 1 : Color Correction :-
for each channel in RGB image do the color correction according to the given formula :-
```math
U^c=\frac{255}{2} \left(1+\frac{S^c-M^c}{\mu V^c} \right)
```
where
$S^c$ = Color channel having values in [0,1].
$M^c$ = Mean of the channel .
$V^c$ = Varience of the channel.
### Step 2 : HSV to RGB conversion :-
Digital color images has 3 color channels R,G,B . Every value of RGB channels lies between 0 to 255 including both 0 and 255.
RGB stands for RED,GREEN,BLUE and HSV stands for HUE, SATURATION, VALUE
#### conversion from RGB to HSV :-
```math
\begin{align*}
    &R'=R/255, G'=G/255, B'=B/255 \\
    &cmax=max(R',G',B'), \\ &cmin=min(R',G',B') \\
    &\Delta=cmax-cmin \\
    &H=\begin{cases}
60\degree \times \left(\frac{G'-B'}{\Delta} \mod 6 \right) & \text{if } cmax=R' \\
60\degree \times \left(\frac{B'-R'}{\Delta} + 2 \right) & \text{if } cmax=G' \\
60\degree \times \left(\frac{R'-G'}{\Delta} + 4 \right) & \text{if } cmax=B' \\
\end{cases} \\
&S=\begin{cases}
0 & \text{if } cmax=0 \\
\frac{\Delta}{cmax} & \text{if } cmax \neq 0 \\
\end{cases} \\
&V=cmax \\

\end{align*}

```
### Step 3 : Model Description :-
Now select the V channel and let L=V 
$L=\frac{L}{255}$ . 
By Retinex theory, $L=I\circ R $ where $R$ is Reflectance and $I$ is Illumination. $\circ$ is the element wise multiplication.
```math
I_{ij} \in [0,255] \\
R_{ij} \in [0,1]

```
Now consider L,I,R as Random Variables then :-
```math
p(L,I|R) \propto p(L|R,I)p(I)p(R) \\
```
where $p(L|R,I)$ is the likelihood of $L$ given $R$ and $I$, $p(I)$ is the prior of $I$ and $p(R)$ is the prior of $R$.
Now consider the following assumptions :-
```math
\begin{align*}
&e=L-I\circ R \\
&p(L|I,R) \thicksim \mathcal{N}(e|0,\sigma^2 I) \\
&p_1(R) \thicksim \mathcal{L}(\nabla R|0,s_1 I)\\
&p_2(R) \thicksim \mathcal{L}(\triangle R|0,s_2 I)\\
&p(R)=p_1(R)p_2(R) \\
&p_3(I) \thicksim \mathcal{N}(\nabla I|0,\sigma_1^2 I)\\
&p_4(I) \thicksim \mathcal{N}(\triangle I|0,\sigma_2^2 I)\\
&p(I)=p_3(I)p_4(I) \\
\end{align*}
```
where $\mathcal{N}$ is the Gaussian distribution and $\mathcal{L}$ is the Laplacian distribution.
```math
\begin{align*}
&\nabla_h=[-1,1] , \nabla_v=[-1;1]  \\
&\triangle=\begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0 \\
\end{bmatrix}\\
&\mathcal{L}(x|\mu,b)=\frac{1}{2b}exp{\biggl(-\frac{|x-\mu|}{b}\biggr)} \\
&\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}exp{\biggl(-\frac{(x-\mu)^2}{2\sigma^2}\biggr)} \\
\end{align*}
```
### Step 4 : Objective Function :-
Since we want to maximize the probability of $p(L,I|R)$ so we will minimize the negative log likelihood of $p(L,I|R)$.
```math
\begin{align*}
&\mathcal{L}(L,I,R)=-\log p(L,I|R) \\
&\mathcal{L}(L,I,R)=-\log p(L|R,I)p(I)p(R) \\
&\mathcal{L}(L,I,R)=-\log p(L|R,I)-\log p(I)-\log p(R) \\
&\mathcal{L}(L,I,R)=-\log p(L|R,I)-\log p(I)-\log p_1(R)-\log p_2(R) \\
&\mathcal{L}(L,I,R)=-\log p(L|R,I)-\log p_3(I)-\log p_4(I)-\log p_1(R)-\log p_2(R) \\
&\mathcal{L}(L,I,R)=
-\sum_{\substack{0<i<m\\0<j<n}}\log \frac{1}{\sqrt{2\pi}\sigma}exp{\biggl(-\frac{(I_{ij} \circ R_{ij}-L_{ij})^2}{2\sigma^2}\biggr)}
-\sum_{\substack{0<i<m\\0<j<n}}\log \frac{1}{\sqrt{2\pi}\sigma_1}exp{\biggl(-\frac{(\nabla I_{ij})^2}{2\sigma_1^2}\biggr)} \\
&-\sum_{\substack{0<i<m\\0<j<n}}\log \frac{1}{\sqrt{2\pi}\sigma_2}exp{\biggl(-\frac{(\triangle I_{ij})^2}{2\sigma_2^2}\biggr)}
-\sum_{\substack{0<i<m\\0<j<n}}\log \frac{1}{2s_1}exp{\biggl(-\frac{|\nabla R_{ij}|}{s_1}\biggr)}
-\sum_{\substack{0<i<m\\0<j<n}}\log \frac{1}{2s_2}exp{\biggl(-\frac{|\triangle R_{ij}|}{s_2}\biggr)}\\
& \mathcal{L}(L,I,R)=\sum_{\substack{0<i<m\\0<j<n}}\frac{(I_{ij} \circ R_{ij}-L_{ij})^2}{2\sigma^2}
+\sum_{\substack{0<i<m\\0<j<n}}\frac{(\nabla I_{ij})^2}{2\sigma_1^2}+\sum_{\substack{0<i<m\\0<j<n}}\frac{(\triangle I_{ij})^2}{2\sigma_2^2} +\sum_{\substack{0<i<m\\0<j<n}}\frac{|\nabla R_{ij}|}{s_1}+\sum_{\substack{0<i<m\\0<j<n}}\frac{|\triangle R_{ij}|}{s_2} + C\\
&\mathcal{L}(L,I,R)=\text{\textbardbl} I \circ R-L \text{\textbardbl}_2^2 + \frac{\sigma^2}{2\sigma^2}\text{\textbardbl} \nabla I \text{\textbardbl}_2^2 + \frac{\sigma^2}{2\sigma_1^2}\text{\textbardbl} \triangle I \text{\textbardbl}_2^2 + \frac{\sigma^2}{s_1}\text{\textbardbl} \nabla R \text{\textbardbl}_1 + \frac{\sigma^2}{s_2}\text{\textbardbl} \triangle R \text{\textbardbl}_1 + C \\
&\mathcal{E}(I,R)=\text{\textbardbl} I \circ R-L \text{\textbardbl}_2^2 + \frac{\sigma^2}{2\sigma^2}\text{\textbardbl} \nabla I \text{\textbardbl}_2^2 + \frac{\sigma^2}{2\sigma_1^2}\text{\textbardbl} \triangle I \text{\textbardbl}_2^2 + \frac{\sigma^2}{s_1}\text{\textbardbl} \nabla R \text{\textbardbl}_1 + \frac{\sigma^2}{s_2}\text{\textbardbl} \triangle R \text{\textbardbl}_1 \\
&\mathcal{E}(I,R)=\text{\textbardbl} I \circ R-L \text{\textbardbl}_2^2 + \nu_1\text{\textbardbl} \nabla I \text{\textbardbl}_2^2 + \nu_2\text{\textbardbl} \triangle I \text{\textbardbl}_2^2 + \nu_3\text{\textbardbl} \nabla R \text{\textbardbl}_1 + \nu_4\text{\textbardbl} \triangle R \text{\textbardbl}_1 \\
\end{align*}
```
where $\nu_1=\frac{\sigma^2}{2\sigma^2}$, $\nu_2=\frac{\sigma^2}{2\sigma_1^2}$, $\nu_3=\frac{\sigma^2}{s_1}$, $\nu_4=\frac{\sigma^2}{s_2}$ and $C$ is a constant.
### Step 5 : ADMM Algorithm :-
#### (1) Closed Set :- 
A set is closed if it contains all its limit points. 
```math
\text{if} \enspace \{x_n\} \in A \enspace \{x_n\} \to x \text{ then } x \in A
```
#### (2) Convex Function :-
A function is convex if the line segment between any two points on the graph of the function lies above the graph.
```math
f(\lambda x+(1-\lambda)y) \leq \lambda f(x)+(1-\lambda)f(y) \enspace \forall x,y \in \mathbb{R}^n \enspace \forall \lambda \in [0,1]
```
#### (3) Convex Conjugate :-
The convex conjugate of a function $f$ is defined as :-
```math
f^*(y)=\sup_{x \in \mathbb{R}^n} \{x^Ty-f(x)\}
```
$f^*(y)$ is always a convex function even if $f$ is not convex.
#### (4) Dual Function :-
The dual function of a convex optimization problem is defined as :-
```math
\text{minimize} \enspace f(x) +g(y) \\
\text{subject to} \enspace Ax=y\\
\text{where} \enspace x \in \mathbb{R}^n, y \in \mathbb{R}^m
\mathcal{L}(x,y,\lambda)=f(x)+g(y)+\lambda^T(Ax-y)\\
\text{Dual of the problem is} \\
\text{maximize} \enspace \mathcal{g}(y)=-f^*(-A^Ty)-g^*(y) \\

```
#### conversion from HSV to RGB :-
```math 
\begin{align*}
    &C=V \times S \\
    &X=C \times (1-|(\frac{H}{60} \mod 2)-1|) \\
    &m=V-C \\
    &\begin{cases}
    (R',G',B')=(C,X,0) & \text{if } 0 \leq H < 60 \\
    (R',G',B')=(X,C,0) & \text{if } 60 \leq H < 120 \\
    (R',G',B')=(0,C,X) & \text{if } 120 \leq H < 180 \\
    (R',G',B')=(0,X,C) & \text{if } 180 \leq H < 240 \\
    (R',G',B')=(X,0,C) & \text{if } 240 \leq H < 300 \\
    (R',G',B')=(C,0,X) & \text{if } 300 \leq H < 360 \\
    \end{cases} \\
    &(R,G,B)=(R'+m,G'+m,B'+m) \\
\end{align*}
```