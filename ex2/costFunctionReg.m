function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta_summation = 0;
cost_summ =0;
gradient_summ = zeros(size(theta));




% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

 for j = 2:size(theta)
     theta_summation = theta_summation + (theta(j))^2 ;
 end
 
 summation = theta_summation * (lambda/2);
 
 for i = 1:m
     cost_summ = cost_summ  - (y(i)*log(sigmoid((theta.')* (X(i,:).')))) - ((1-y(i))*log(1 - sigmoid( (theta.')* (X(i,:).')))) ;
 end
 
 
 J = (1/m) * (cost_summ + summation);
    
 for i = 1:m
     gradient_summ(1) = gradient_summ(1) + ( sigmoid( (theta.')* (X(i,:).' )) - y(i))* X(i,1);
 end
 
 for j = 2:size(theta)
     for i = 1:m
         
         gradient_summ(j) = gradient_summ(j) +  (sigmoid( (theta.')* (X(i,:).' )) - y(i)) * X(i,j);
    
     end
    
 end
 
 
grad(1) = gradient_summ(1) * (1/m);
 
for j = 2:size(theta)
    grad(j) = (gradient_summ(j) + (theta(j) * lambda)) *(1/m);
end

    



% =============================================================

end
