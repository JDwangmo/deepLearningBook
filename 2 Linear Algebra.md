# 2 Linear Algebra:

[pdf文档](https://raw.githubusercontent.com/JDwangmo/deepLearningBook/master/book/www.deeplearningbook.org_contents_linear_algebra.pdf) from http://www.deeplearningbook.org/contents/linear_algebra.html

2.1 Scalar,Vectors,Matrices and Tensors

- 标量（Scalars）：一个单独的数字（a single number）,可以是实数（real-valued scalar）或自然数（nutural numer scalar）。一般用小写的变量名：比如 “Let s ∈ R be the slope of the line.”
- 向量（Vectors）：数字的有序数组，可以通过索引（index）来访问（identify）每个数字。通常向量的变量名用加粗的小写字母表示（lower case names written in bold typeface）,比如**x**，访问每个元素的写法是斜体字母加下标（identified by writing its name in italic typeface,with a subscript）,比如第一个元素为_$$x_1$$_。$$R^n$$表示元素是实数的任意一个n维向量（ If each element is in R, and the vector has n elements, then the vector lies in the set formed by taking the Cartesian product of R n times, denoted as $$R^n$$ . ）。
    还访问一系列的元素，Sometimes we need to index a set of elements of a vector. In this case, we define a set containing the indices and write the set as a subscript. For example, to access $$x_1$$, $$x_3$$ and $$x_6$$, we define the set S = {1,3,6} and write xS. We use the − sign to index the complement of a set. For example $$x_{-1}$$ is the vector containing all elements of x except for $$x_1$$, and $$x_{-S}$$ is the vector containing all of the elements of **x** except for $$x_1$$, $$x_3$$ and $$x_6$$.（补集等）
- 矩阵（Matrices）：二维数组，可以通过行列索引来访问矩阵的元素。我们通常用一个粗体大写变量名来表示矩阵（give matrices upper-case variable names with bold typeface,such as **A**）。类似的$$R^{m*n}$$表示一个买m\*n的任意实数矩阵。访问元素的话，一般是使用一个斜体但不加粗的变量来表示，比如 $$A_{i,j}$$。$$A_{i,:}$$ denotes the horizontal cross section of A with vertical coordinate i（第i行）. This is known as the i-th row of A. Likewise, $$A_{:,i}$$ 表示第i列。 
    - `P3-top`：$$f(A)_{i,j}$$ gives element (i,j) of the matrix computed by applying the function f to **A**. 
- 张量（Tensors）：坐标维度是变量的向量。In some cases we will need an array with **more than two axes**. In the general case, an array of numbers arranged on a regular grid with **a variable number of axes is known as a tensor**. We denote a tensor named “A” with this typeface:** A**. We identify the element of A at coordinates (i,j,k) by writing $$A_{i,j,k}$$ .


- 矩阵操作：
    - 转置（transpose）：$$(A^T)_{i,j}$$ = $$A_{j,i}$$。
        - 注意：向量（Vectors）通常被当作一个**一列的矩阵**。所以向量的转置就变成了一行的矩阵。x = $$[x_1,x_2,x_3]^T$$.
    - 矩阵加减乘除法： for examples,
        - **C=A+B** ==> C_{i,j}=$$A_{i,j}$$+$$B_{i,j}$$。
        - **D** = a·**B** +**c** where $$D_{i,j}$$ = a · $$B_{i,j}$$ + c.
        - 矩阵和向量的加法：向量被加到矩阵的每一列，具体为：向量先扩展(broadcasting)，然后再加到矩阵中去。（In the context of deep learning, we also use some less conventional notation. We allow the addition of matrix and a vector, yielding another matrix: C = A +b, where Ci,j = Ai,j +bj. In other words, the vector b is added to each row of the matrix. This shorthand eliminates the need to define a matrix with b copied into each row before doing the addition. **This implicit copying of b to many locations is called broadcasting**）
    - 矩阵多种乘的区分：`P4-bottom`
        - element-wise product or Hadamard product: 矩阵对应元素相乘。 A $$\odot$$B
        - dot product: 点积,matrix product C = AB as computing $$C_{i,j}$$ as the dot product between row i of A and column j of B.
        - 遵循分配律（distributive）和结合律（associative）：**A ( B + C) = AB + AC**. **A(BC) = (AB)C** 。
        - 矩阵之间不符合交换律（commutative），但是向量之间符合。
    - 单位矩阵(identity matrix)：`P6-center,2.3`: 单位矩阵是指其与任何矩阵相乘时，不会改变任何矩阵的数值（An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix）,其实就是个对角线方向数值全为1,而其他位置都为0的方0矩阵（all of the entries along the main diagonal are 1, while all of the other entries are zero）。通常用$$I_n$$表示n维的单位矩，$$I_n$$ ∈ $$R^{n*n} $$, and ∀x ∈ $$R^n$$,$$I_nx = x$$
    - 矩阵逆（matrix inverse）: $$A^{-1}A=I_n$$ ,也可以写作$$AA^{-1}=I_n$$，左逆和右逆是一样。 
- 我们学习线性代数，目标就是要解方程：**Ax = b**（`P5-center,equation 2.11`），A是一个矩阵(∈$$R^{m*n}$$)，b是一个向量( ∈$$R^m$$)，都是已知，要求解向量x( ∈$$R^n$$)的值。如下：

    **Ax = b**

    $$A^{-1}Ax=A^{-1}b$$.

    $$I_nx=A^{-1}b$$

    $$x = A^{-1}b$$

- 但有个关键问题就是，能够找到$$A^{-1}$$（this process depends on it being possible to find $$A^{-1}$$）。不过实际应用中很少用到$$A^{-1}$$，更多的是作为一个理论工具。

- linear combination
- **Important point：**`P7-center`:A的每一列$$c_i$$可以当作从原点出发到这个坐标的方向，$$x_i$$当作这个方向上的距离（走多远）。这种转换理解非常棒。对于后面的理解很有帮助。看原文：To analyze how many solutions the equation has, we can think of **the columns of A as specifying different directions we can travel from the origin (the point specified by the vector of all zeros，即原点)**, and **determine how many ways there are of reaching b**. In this view, **each element of x specifies how far we should travel in each of these directions**, **with $$x_i$$ specifying how far to move in the direction of column i**。
- **生成子空间（Span）**：`P7-bottom`：区间，跨度，或者叫生成子空间。是由基向量构成的所有可能点的空间。（The **span **of a set of vectors is the set of all points obtainable by linear combination of the original vectors）
    - `P8-top`:因此，求解Ax=b的问题转而变成判断向量**b**是否在矩阵**A**的所有列向量构成的子空间（span）中，在这里，也叫做 **column space** or the range of **A**。
    - **推理：**为了使得对于任意的**b**都有解，则**A**的**列数必须大于等于行数（n>=m）**。 In order for the system Ax = b to have a solution for all values of b ∈ $$R^m$$ , we therefore require that the column space of A be all of $$R^m$$ . If any point in R m is excluded from the **column space**, that point is a potential value of b that has no solution. The requirement that the column space of A be all of R m implies immediately that A must have at least m columns, i.e., n m ≥ . Otherwise,假设consider a 3 × 2 matrix. The target b is 3-D, but x is only 2-D, so modifying the value of x at best allows us to trace out a 2-D plane within $$R^3$$ . The equation has a solution if and only if b lies on that plane.相当于A的每一列提供了一个方向，在这个例子里面就是两个方向（有2列），每一行提供了一个维度，这里是3D（因为有三行），问题就变成了如果让3D空间中的2个点，去生成一个子空间（span）取覆盖整个3D空间（即使得任意的 $$**b** \in R^3$$ 都能有解）。**提供2个方向，怎么走才能到达b点。**
    - 但是，要注意：n$$\geq$$m 只是对于每个$$**b** \in R^m$$有解的一个必要条件（necessary condition）,不是一个充分条件（Having n ≥ m is only a necessary condition for every point to have a solution. It is not a sufficient condition），因为有可能**A**中有一些列是冗余的（redundant）。
        - For example, Consider a 2 ×2 matrix where both of the columns are identical(两列是一样的). This has the** same column space **as a 2 × 1 matrix containing only one copy of the replicated column（**也就是现在的columns space 是跟包含一个columns的 2 × 1 matrix是一样的**）. In other words, the column space is still just a line, and fails to encompass all of $$R^2$$ , even though there are two columns。（**columns space只是一条线，无法完全覆盖2维空间。**）
        - 上面这种冗余情况，称作**线性依赖（linear dependence）**。在一个向量集合里，假如任意一个向量都无法由其他向量**线性组合（linear combination）**，那么这个向量集合是线性无关的（linear independent）（A set of vectors is linearly independent if no vector in the set is a linear combination of the other vectors）。假如增加一个向量到这个集合里，如果这个向量可以由原本集合中的向量线性组合(linear combination)，则这个向量的增加没有增大原本的集合的生成子空间（span）（ If we add a vector to a set that is a linear combination of the other vectors in the set, the new vector does not add any points to the set’s span）。
        - **推论：`P8-center`如果一个矩阵的 columns space 想去覆盖所有的 $$R^m$$,矩阵中必须有m个线性无关的列向量。This means that for the column space of the matrix to encompass all of $$R^m$$ , the matrix must contain at least one set of m linearly independent columns.**这对于**Ax = b**（`P5-center,equation 2.11`）是一个**充分必要条件。**即如果矩阵A的列向量是m个线性无关的向量（注意是刚刚好m个线性无关的列向量，而不是至少m个），那么对于任意的**b**都有**一个解**（have exactly a solution for every value of **b**）。
        - 一个m维空间的向量集合最多只能有m个相互线性无关的向量，但是多于m个列向量的矩阵中，可能有多套相互线性无关的向量。（No set of m-dimensional vectors can have more than m mutually linearly independent columns, but a matrix with more than m columns may have more than one such set。）

- 好，接下来还是回到**一个矩阵是否有逆矩阵**的问题上来。首先必须确保**对于任意的b最多只有一个解（In order for the matrix to have an inverse, we additionally need to ensure that equation 2.11 has at most one solution for each value of b**）,即**矩阵A最多只能m个列**。
    - **对于一套 m个 相互线性无关的向量组，达到任意一个点（或者任意的向量**b**），都只有一种可能，即确定类基向量，空间的点都是可以通过唯一的坐标来确定的**。
    - 如果有多套 线性无关向量组，则会有种解法，每套对应一种。
- 综上，我们可以得到一个结论：矩阵**A**必须是**一个方矩阵（square matrix），即 m==n 切 所有的列（n个）向量之间都是线性无关的**。**这样一个方矩阵（a square matrix with linearly dependent columns）**,叫做**奇异矩阵（singular）**
    - 假如矩阵**A**不是方矩阵或者方矩阵但不奇异，仍然可能有解，但是我们不用矩阵逆的方法来求解。

2.5 Norms（范数）：`P9-top,2.5`向量的大小
- 在机器学习（ML）中，我们通常用一个范数（norm）来衡量一个向量的大小（measure the size of vectors using a function callerd a norm）。表示为 $$L^p$$,公式如下:
    
    $$ ||x||_p = \left( \sum_{i}|x_i|^p \right)^\frac{1}{p} \qquad  p \in R,p \geq 1 $$   

- 范数 就是一个将向量转为非负数的值。可以理解为到原点的距离（On an intuitive level, the norm of a vector x measures the distance from the origin to the point x）。符合这样一些条件：`P9,center`
    - f(**x**) = 0 ⇒ x = 0 
    - f( **x**+**y** ) ≤ f ( **x** )+ f( **y**) (the triangle inequality,两边之和大于第三边)。
    - ∀α ∈ R,f(α**x**) = |α|f(**x**)
    - 一些常用的范数：
        - `P9-center`:p=2时，即$$L^2$$范数，也叫 **欧基里德 范数（Euclidean norm）**，通常表示为||**x**||,省略掉下标2。更常用的是用**$$L^2$$的平方（squared $$L^2$$ norm）**，可以通过** $$x^Tx$$**计算得出。因为：The squared $$L^2$$ norm is more convenient to work with mathematically and computationally than the $$L^2$$ norm itself. 比如：squared $$L^2$$ 的对**x**每个元素的求导（$$2|x_i|$$），只跟x的对应元素有关，而 $$L^2$$ 跟整个**x**有关(the derivatives of the **squared $$L^2$$ norm with respect to each element of x each depend only on the corresponding element of x**, while all of the derivatives of the $$L^2$$ norm depend on the entire vector)。
        - `P10-top`: p = 1时，即 即$$L^1$$范数
        - **Important point**： `P9-bottom to P10-top`： $$L^2$$范数**在原点附近时，增长速度太慢**，区分不出 0 和接近0的small number 的区别，如果对这个比较敏感时，这是就会使用$$L^1$$范数，在所有位置的增长速度都一样。（the squared $$L^2$$ norm may be undesirable because **it increases very slowly near the origin**. In several machine learning applications, it is important to **discriminate between elements that are exactly zero and elements that are small but nonzero**. In these cases, we turn to a function that** grows at the same rate in all location**s, but retains mathematical simplicity: the $$L^1$$ norm）。The **$$L^1$$ norm is commonly used in machine learning when the difference between zero and nonzero elements is very important**. **Every time an element of x moves away from 0 by $$\epsilon$$, the $$L^1$$ norm increases by $$\epsilon$$**.
        - "$$L^0$$": 计算向量中不为0的元素个数，但这是一个不正确的术语，因为这不符合范数的定义，比如向量收放$$\alpha$$倍，都不会改变非0的个数，这不符合范数定义的第三条（∀α ∈ R,f(α**x**) = |α|f(**x**)），我们通常用$$L^1$$范数来代替非0的个数（We sometimes measure the size of the vector by counting its number of nonzero elements. Some authors refer to this function as the “$$L^0$$ norm,” but this is incorrect terminology. The number of non-zero entries in a vector is not a norm, because scaling the vector by α does not change the number of nonzero entries. The L 1 norm is often used as a substitute for the number of nonzero entries）。
        - max norm: $$L^\infty$$，取得绝对值最大的元素，即 $$||x||_\infty = max_i|x_i|$$ 。
    - 对于矩阵的大小，比较常用的是 Frobenius norm ,类似于向量的 $$L^2$$ norm: $$||A||_F=\sqrt(\sum_{i,j}A^2_{i,j})$$
    - 两个向量的点积(dot prodcut)可以写成 范数的 形式： $$x^Ty=||x||_2||y||_2cos\theta$$ ，其中$$\theta$$是x和y之间的角度。
    
2.6 一些特殊矩阵或者向量(Special Kinds of Matrics and Vectors):
- **对角矩阵（Diagonal matrices）**: `P10-bottom``,除了主对角线，其他元素都是0.比如单元矩阵就算是其中一个。通常用 diag(**v**)的形式 来表示对角方矩阵（square diagonal matrix）,其中**v**对应矩阵的对角线（We write diag(v) to denote a square diagonal matrix whose diagonal entries are given by the entries of the vector v）。对角矩阵比较吸引人的地方在于
    - 与一个对角矩阵相乘计算代价非常低，比如计算 diag(**v**)**x**，我们仅仅需要将v和x点乘，即 $$diag(v)x = v \odot x$$(To compute diag(v)x, we only need to scale each element $$x_i$$ by $$v_i$$. In other words, $$diag(v)x = v \odot x$$)。 
        