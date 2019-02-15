# Machine Learning ex2
> 斯坦福大学机器学习课程编程作业2

![image](https://s2.ax1x.com/2019/02/15/kD4rDK.png)

作业两部分思路均为
- 可视化测试集数据
- 实现计算代价函数和相应偏导数的`costFunction`和`costFunctionReg`函数
- 调用`fminunc`
- 可视化决策边界
    > 提供代码，建议阅读

两个较为简单的热身练习分别是实现sigmoid函数和根据预测可能性和阈值给出判断

#### sigmoid.m

```
function g = sigmoid(z)

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================

g = 1./(1 + exp(-z));

% =============================================================

end
```
函数需要可以对标量、向量、矩阵进行运算。需要对矩阵中的每个元素做相应运算

#### costFunction.m

```
function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

J = sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) / m;

grad = (X' * (sigmoid(X*theta) - y))./m;

% =============================================================

end
```
向量化过程中注意维数

#### predict.m

```
function p = predict(theta, X)
%   p = PREDICT(theta, X) computes the predictions for X using a threshold at 0.5

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================

h = sigmoid(X * theta);
for i = 1:m
    if (h(i))>0.5
        p(i) = 1;
    end
end

% =========================================================================

end
```
需要以0.5为阈值对预测结果向量h进行判断，这里采用循环方法
> 没有想到向量化方法

#### costFunctionReg.m

```
function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

J = sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) / m + lambda * sum(theta(2:end).^2) / (2 * m);

grad = (X' * (sigmoid(X*theta) - y))./m;
grad(2:end) = grad(2:end) + lambda * theta(2:end) / m;

% =============================================================

end
```
grad除第一个元素外都要加正则化，先计算通项，再提取后面的向量作相应运算。
