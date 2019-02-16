# Machine Learning ex4

> 斯坦福大学机器学习课程编程作业4

![image](https://s2.ax1x.com/2019/02/16/kskv0P.png)

需要完成的三个文件如下：
- `sigmoidGradient.m` - Compute the gradient of the sigmoid function
    > 实现sigmoid函数的导函数，计5分
- `randInitializeWeights.m` - Randomly initialize weights
    > 代码在作业文档中给出，不计分
- `nnCostFunction.m` - Neural network cost function
    > 实现前向传播、代价函数及其正则化以及反向传播、偏导数的计算及其正则化，计95分


#### nnCostFunction

```matlab
function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% ----------------- Part 1: Feedforward and cost function -----------------

X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');
a2 = [ones(m,1) a2];
h = sigmoid(a2 * Theta2');

temp = zeros(num_labels,1);
for c = 1 : num_labels
    temp(c) = sum((y == c).* log(h(:,c)) + (1 - (y == c)).* log(1-h(:,c)));
end
J = - sum(temp) / m;

% ------------------ Part 2: Regularized Cost Function --------------------

temp1 = Theta1.^2;
temp2 = Theta2.^2;
r = sum(sum(temp1(:,2:end))) + sum(sum(temp2(:,2:end)));
J = J + lambda * r / (2 * m);

% ----------- Part 3: Neural Network Gradient (Backpropagation) -----------

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for t = 1 : m
% ******************* Step 1: Perform a feedforward pass ******************
    a_1 = X(t,:);
    a_1 = a_1(:);
    a_2 = sigmoid(Theta1 * a_1);
    a_2 = [1; a_2];
    a_3 = sigmoid(Theta2 * a_2);
% *********************** Step 2: Compute delta_3 *************************
    logic_y = (1:num_labels)';
    delta_3 = a_3 - (logic_y == y(t));
% *********************** Step 3: Compute delta_2 *************************
    delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; Theta1 * a_1]);
    delta_2 = delta_2(2:end);
% ********************* Step 4: Accumulate the gradient *******************
    Delta1 = Delta1 + delta_2 * a_1';
    Delta2 = Delta2 + delta_3 * a_2';
end

% ************** Step 5: Obtain the (unregularized) gradient **************
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% -------------------- Part 4: Regularized Gradient -----------------------

Theta1_grad(: , (2:end)) = Theta1_grad(: , (2:end))+ lambda * Theta1(: , (2:end)) / m;
Theta2_grad(: , (2:end)) = Theta2_grad(: , (2:end))+ lambda * Theta2(: , (2:end)) / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
```



- Part 1: Feedforward and cost function
    > 前向传播步骤与上次作业类似，计算代价函数时，将k求和交换到m外面，内部m求和可向量化
- Part 2: Regularized Cost Function
    > matlab不支持索引临时变量，要先用两个temp存储中间结果。单独计算出正则项后再加到代价函数上
- Part 3: Neural Network Gradient (Backpropagation)
    - Step 1: Perform a feedforward pass
        > 计算前向传播的激活单元，注意X在Part 1中已经加入了偏置项
    
    - Step 2: Compute delta_3
        > 使用逻辑数组，注意1:n默认为行向量，需要进行转置
    
    - Step 3: Compute delta_2
        > 计算中注意维数

    - Step 4: Accumulate the gradient
        > 对偏导数进行累加
    
    - Step 5: Obtain the (unregularized) gradient
        > 计算得到未经正则化的偏导数
- Part 4: Regularized Gradient
    > 为偏导数加上正则项，注意第一列不需要正则化

#### sigmoidGradient

```matlab
function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================

g = sigmoid(z) .* (1 - sigmoid(z));

% =============================================================

end
```

#### computeNumericalGradient

做GradientChecking的函数已经被提供，建议阅读，代码如下

```matlab
function numgrad = computeNumericalGradient(J, theta)
%COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
%and gives us a numerical estimate of the gradient.
%   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
%   gradient of the function J around theta. Calling y = J(theta) should
%   return the function value at theta.

% Notes: The following code implements numerical gradient checking, and 
%        returns the numerical gradient.It sets numgrad(i) to (a numerical 
%        approximation of) the partial derivative of J with respect to the 
%        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
%        be the (approximately) the partial derivative of J with respect 
%        to theta(i).)
%                

numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

end
```
