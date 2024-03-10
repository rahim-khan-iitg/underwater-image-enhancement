
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
&\mathcal{L}(x|\mu,b)=\frac{1}{2b}exp{\left(-\frac{|x-\mu|}{b}\right)} \\
&\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}exp{\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)} \\
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
\end{align*}
```
```math
\begin{align*}
&\mathcal{L}(L,I,R)=
-\sum\sum\log \frac{1}{\sqrt{2\pi}\sigma}exp{\left(-\frac{(I_{ij} \circ R_{ij}-L_{ij})^2}{2\sigma^2}\right)}

-\sum\sum\log \frac{1}{\sqrt{2\pi}\sigma_1}exp{\left(-\frac{(\nabla I_{ij})^2}{2\sigma_1^2}\right)} \\
&-\sum\sum\log \frac{1}{\sqrt{2\pi}\sigma_2}exp{\left(-\frac{(\triangle I_{ij})^2}{2\sigma_2^2}\right)}
-\sum\sum\log \frac{1}{2s_1}exp{\left(-\frac{|\nabla R_{ij}|}{s_1}\right)}
-\sum\sum\log \frac{1}{2s_2}exp{\left(-\frac{|\triangle R_{ij}|}{s_2}\right)}\\
\end{align*}
```
```math
\begin{align*}
& \mathcal{L}(L,I,R)=\sum\sum\frac{(I_{ij} \circ R_{ij}-L_{ij})^2}{2\sigma^2}
+\sum\sum\frac{(\nabla I_{ij})^2}{2\sigma_1^2}+\sum\frac{(\triangle I_{ij})^2}{2\sigma_2^2} +\sum\sum\frac{|\nabla R_{ij}|}{s_1}+\sum\sum\frac{|\triangle R_{ij}|}{s_2} + C\\
\end{align*}
```
```math
\begin{align*}
&\mathcal{L}(L,I,R)=\text{||} I \circ R-L \text{||}_2^2 + \frac{\sigma^2}{2\sigma^2}\text{||} \nabla I \text{||}_2^2 + \frac{\sigma^2}{2\sigma_1^2}\text{||} \triangle I \text{||}_2^2 + \frac{\sigma^2}{s_1}\text{||} \nabla R \text{||}_1 + \frac{\sigma^2}{s_2}\text{||} \triangle R \text{||}_1 + C \\
&\mathcal{E}(I,R)=\text{||} I \circ R-L \text{||}_2^2 + \frac{\sigma^2}{2\sigma^2}\text{||} \nabla I \text{||}_2^2 + \frac{\sigma^2}{2\sigma_1^2}\text{||} \triangle I \text{||}_2^2 + \frac{\sigma^2}{s_1}\text{||} \nabla R \text{||}_1 + \frac{\sigma^2}{s_2}\text{||} \triangle R \text{||}_1 \\
&\mathcal{E}(I,R)=\text{||} I \circ R-L \text{||}_2^2 + \nu_1\text{||} \nabla I \text{||}_2^2 + \nu_2\text{||} \triangle I \text{||}_2^2 + \nu_3\text{||} \nabla R \text{||}_1 + \nu_4\text{||} \triangle R \text{||}_1 \\
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
#### (5) Dual Ascent Algorithm :-
The dual ascent algorithm is defined as :-
```math
\begin{align*}
&\mathcal{f}: \mathbb{R}^n \to \mathbb{R} , \enspace b,x \in \mathbb{R}^n \enspace and \enspace \mathcal{f} \enspace \text{is convex} \\
&minimize \enspace \mathcal{f}(x) \\
&\text{subject to} \enspace Ax=b \\
&\mathcal{L}(x,y)=\mathcal{f}(x)+y^T(Ax-b) \\
&\text{Dual of the problem is} \\
&\text{maximize} \enspace \mathcal{g}(y)=-\mathcal{f}^*(-A^Ty)-b^Ty\\
&\text{where} \enspace \mathcal{f}^*(y)=\sup_{x \in \mathbb{R}^n} \{x^Ty-f(x)\} \\
&x^*=\arg\min_{x} \mathcal{L}(x,y^*) \enspace where \enspace y^* \enspace \text{is optimal solution of } \mathcal{g}(y). \\
\end{align*}
```
we optimize it by the following steps :-
```math
\begin{align*}
& x^{k+1}=\arg\min_{x} \mathcal{L}(x,y^k) \\
& y^{k+1}=y^k+\rho(Ax^{k+1}-b) \
\end{align*}
```
#### (6) Augmented Lagrangian Method of Multipliers :-
The augmented lagrangian method of multipliers is defined as :-
```math
\begin{align*}
&\text{minimize} \enspace f(x) +\frac{\rho}{2}\text{||}Ax-b\text{||}_2^2 \\
&\text{subject to} \enspace Ax=b \\
&\text{then } \mathcal{L}_\rho(x,y)=f(x)+y^T(Ax-b)+\frac{\rho}{2}\text{||}Ax-b\text{||}_2^2 \\
&\text{now } x^{k+1}=\arg\min_{x} \mathcal{L}_\rho(x,y^k) \\
&\text{and } y^{k+1}=y^k+\rho(Ax^{k+1}-b) \\ 
\end{align*}
```
#### (7) ADMM Algorithm :-
The ADMM algorithm is defined as :-
```math
\begin{align*}
&\mathcal{f,g}: \mathbb{R}^n \to \mathbb{R} , \enspace x,z,c \in \mathbb{R}^n \enspace and \enspace \mathcal{f,g} \enspace \text{are convex and } \mathcal{A,B}\in \mathbb{R}^{p\times n}, c\in \mathbb{R}^p\\
&\text{minimize} \enspace \mathcal{f}(x)+\mathcal{g}(z) \\
&\text{subject to} \enspace \mathcal{A}x+\mathcal{B}z=c \\
&\text{then the Augmented Langrangian of the problem is  } \\ &\mathcal{L}_\rho(x,z,y)=\mathcal{f}(x)+\mathcal{g}(z)+y^T(\mathcal{A}x+\mathcal{B}z-c)+\frac{\rho}{2}\text{||}\mathcal{A}x+\mathcal{B}z-c\text{||}_2^2 \\
&\text{now } x^{k+1}=\arg\min_{x} \mathcal{L}_\rho(x,z^k,y^k) \\
&\text{and } z^{k+1}=\arg\min_{z} \mathcal{L}_\rho(x^{k+1},z,y^k)  \\
&\text{and } y^{k+1}=y^k+\rho(\mathcal{A}x^{k+1}+\mathcal{B}z^{k+1}-c) \\
\end{align*}
```
scaled form of the ADMM algorithm is defined as :-
```math
\begin{align*}
&\text {let }r = \mathcal{A}x+\mathcal{B}z-c \enspace \text{then} \\
&y^Tx+\frac{\rho}{2}\text{||}r\text{||}_2^2 = \frac{\rho}{2} \text{||}r+\frac{1}{\rho}y \text{||}_2^2 - \frac{1}{2\rho}\text{||}y\text{||}_2^2\\
& \text{let } u=\frac{1}{\rho}y \enspace \text{then} \\
&y^Tx+\frac{\rho}{2}\text{||}r\text{||}_2^2 = \frac{\rho}{2} \text{||}r+u \text{||}_2^2 - \frac{\rho}{2}\text{||}u\text{||}_2^2\\
&\text{now } x^{k+1}=\arg\min_{x} (\mathcal{f}(x)+\frac{\rho}{2}\text{||}\mathcal{A}x+\mathcal{B}z^k-c+u^k\text{||}_2^2) \\
& z^{k+1}=\arg\min_{z} (\mathcal{g}(z)+\frac{\rho}{2}\text{||}\mathcal{A}x^{k+1}+\mathcal{B}z-c+u^k\text{||}_2^2)  \\
&u^{k+1}=u^k+\mathcal{A}x^{k+1}+\mathcal{B}z^{k+1}-c \\
&\text{and } r^{k+1}=\mathcal{A}x^{k+1}+\mathcal{B}z^{k+1}-c \\
&u^k=u^0+\sum_{j=0}^{k}r^j \\
\end{align*}
```
### Step 6 : Numerical Optimization :-
To optimize the objective function . We have to convert $l_1$ norm to $l_2$ norm. We introduce two auxiliary variables $d,h$ and two error terms $m,n$.
```math
\begin{align*}
&\mathcal{E}(I,R)=\text{||} I \circ R-L \text{||}_2^2 + \nu_3\text{||} \nabla I \text{||}_2^2 + \nu_4\text{||} \triangle I \text{||}_2^2 + \nu_1\text{||} \nabla R \text{||}_1 + \nu_1\text{||} \triangle R \text{||}_1 \\
&\mathcal{E}(I,R)=\text{||} I \circ R-L \text{||}_2^2 + \nu_3 \text{||} \nabla I \text{||}_2^2 + \nu_4\text{||} \triangle I \text{||}_2^2 + \nu_1 \left(\text{||}d\text{||}_1 +\lambda_1\text{||} \nabla R -d+m\text{||}_2^2 \right) + \nu_1 \left(\text{||}h\text{||}_1+\lambda_2\text{||} \triangle R -h+n\text{||}_2^2\right) \\
\end{align*}
```
Now we split this into three parts and optimize it using ADMM algorithm $\rightarrow$\
P-1
```math
\begin{align*}
&d^{k}=\arg\min_{d} \left(\text{||}d\text{||}_1 +\lambda_1\text{||} \nabla R^{k-1} -d+m^{k-1}\text{||}_2^2 \right) \\
&h^{k}=\arg\min_{h} \left(\text{||}h\text{||}_1+\lambda_2\text{||} \triangle R^{k-1} -h+n^{k-1}\text{||}_2^2\right) \\
\end{align*}
```
P-2
```math
\begin{align*}
&R^{k}=\arg\min_{R} \left(\text{||} R-\frac{L}{I^{k-1}} \text{||}_2^2 + \nu_1\lambda_1\text{||} \nabla R -d^{k}+m^{k-1}\text{||}_2^2 + \nu_2 \lambda_2\text{||} \triangle R -h^{k}+n^{k-1}\text{||}_2^2\right) \\
&m^k=m^{k-1}+\nabla R^k-d^k \\
&n^k=n^{k-1}+\triangle R^k-h^k \\
\end{align*}
```
P-3
```math
\begin{align*}
&I^{k}=\arg\min_{I} \left(\text{||} I-\frac{L}{R^k} \text{||}_2^2 + \nu_3\text{||} \nabla I\text{||}_2^2 + \nu_4\text{||} \triangle I\text{||}_2^2\right) \\
\end{align*}
```
### Step 7 : Update for P-1 :-
```math
\begin{align*}
&d^{k}_h=\text{shrink}(\nabla_h R^{k-1}+m^{k-1}_h,\frac{1}{2\lambda_1}) \\
&d^{k}_v=\text{shrink}(\nabla_v R^{k-1}+m^{k-1}_v,\frac{1}{2\lambda_1}) \\
&h^{k}_h=\text{shrink}(\triangle_h R^{k-1}+n^{k-1}_h,\frac{1}{2\lambda_2}) \\
&\text{where } \text{shrink}(x,\gamma)=\max(0,|x|-\gamma) \times \frac{x}{|x|} \text{ and } \frac{x}{|x|}=0 \text{ if } x=0\\
\end{align*}
```
### Step 8 : Update for P-2 :-
```math
\begin{align*}
&R^k=\mathcal{F}^{-1}\left(\frac{\mathcal{F}(L/I^{k-1})+\nu_1\lambda_1\varPhi_1+\nu_2\lambda_2\varPhi_2}{1+\nu_1\varPsi_1+\nu_2\varPsi_2}\right) \\
&\text{where } \varPhi_1=\mathcal{F}^{*}(\nabla_h).\mathcal{F}(d^{k}_h+m^{k-1}_h)+\mathcal{F}^{*}(\nabla_v).\mathcal{F}(d^{k}_v+m^{k-1}_v) \text{ and }\\
&\varPhi_2=\mathcal{F}^{*}(\triangle).\mathcal{F}( h^{k}+n^{k-1}) \\
&\varPsi_1=\mathcal{F}^{*}(\nabla_h).\mathcal{F}(\nabla_h)+\mathcal{F}^{*}(\nabla_v).\mathcal{F}(\nabla_v) \text{ and }\\
&\varPsi_2=\mathcal{F}^{*}(\triangle).\mathcal{F}(\triangle)  \enspace \mathcal{F} \text{ is FFT Operator} \\
&m^k_h=m^{k-1}_h+\nabla R^k_h-d^k_h \\
&m^k_v=m^{k-1}_v+\nabla R^k_v-d^k_v \\
&n^k_h=n^{k-1}_h+\triangle R^k_h-h^k_h \\
\end{align*}
```
### Step 9 : Update for P-3 :-
```math
\begin{align*}
&I^k=\mathcal{F}^{-1}\left(\frac{\mathcal{F}(\frac{L}{R^k})}{\mathcal{F}(1)+\nu_3\varPsi_3+\nu_4\varPsi_4}\right) \\
\end{align*}
```
### Algorithm :-
Input:- input value channel L, weighting parameters $\nu_1,\nu_2,\nu_3,\nu_4$ and the number of iterations K 


#### $\text{\textcolor{blue}{conversion}}$ from HSV to RGB :-
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