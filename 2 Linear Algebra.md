# 2 Linear Algebra:

[pdf文档](https://raw.githubusercontent.com/JDwangmo/deepLearningBook/master/book/www.deeplearningbook.org_contents_linear_algebra.pdf) from http://www.deeplearningbook.org/contents/linear_algebra.html
$$x_{-1}$$
2.1 Scalar,Vectors,Matrices and Tensors

- 标量（Scalars）：一个单独的数字（a single number）,可以是实数（real-valued scalar）或自然数（nutural numer scalar）。一般用小写的变量名：比如 “Let s ∈ R be the slope of the line.”
- 向量（Vectors）：数字的有序数组，可以通过索引（index）来访问（identify）每个数字。通常向量的变量名用加粗的小写字母表示（lower case names written in bold typeface）,比如**x**，访问每个元素的写法是斜体字母加下标（identified by writing its name in italic typeface,with a subscript）,比如第一个元素为_$$x_1$$_。$$R^n$$表示元素是实数的任意一个n维向量（ If each element is in R, and the vector has n elements, then the vector lies in the set formed by taking the Cartesian product of R n times, denoted as $$R^n$$ . ）。
    还访问一系列的元素，Sometimes we need to index a set of elements of a vector. In this case, we define a set containing the indices and write the set as a subscript. For example, to access $$x_1$$, $$x_3$$ and $$x_6$$, we define the set S = {1,3,6} and write xS. We use the − sign to index the complement of a set. For example $$x_{-1}$$ is the vector containing all elements of x except for $$x_1$$, and $$x_{S}$$ is the vector containing all of the elements of **x** except for $$x_1$$, $$x_3$$ and $$x_6$$.（补集等）
- 矩阵（Matrices）：二维数组，可以通过行列索引来访问矩阵的元素。我们通常用一个粗体大写变量名来表示矩阵（give matrices upper-case variable names with bold typeface,such as **A**）。类似的$$R^{m*n}$$表示一个买m\*n的任意实数矩阵。访问元素的话，一般是使用一个斜体但不加粗的变量来表示，比如 $$A_{i,j}$$。$$A_{i,:}$$ denotes the horizontal cross section of A with vertical coordinate i（第i行）. This is known as the i-th row of A. Likewise, $$A_{:,i}$$ 表示第i列。 
    - `P3-top`：$$f(A)_{i,j}$$ gives element (i,j) of the matrix computed by applying the function f to **A**. 
- 张量（Tensors）：坐标维度是变量的向量。In some cases we will need an array with **more than two axes**. In the general case, an array of numbers arranged on a regular grid with **a variable number of axes is known as a tensor**. We denote a tensor named “A” with this typeface:** A**. We identify the element of A at coordinates (i,j,k) by writing $$A_{i,j,k}$$ .


- 矩阵操作：
    - 转置（transpose）：$$(A^T)_{i,j}$$ = $$A_{j,i}$$。
        - 注意：向量（Vectors）通常被当作一个**一列的矩阵**。所以向量的转置就变成了一行的矩阵。x = $$[x1,x2,x3]^T$$.
