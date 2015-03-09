function soft_SVM_demo_hw()
% perceptron_demo_wrapper - runs the perceptron model on a separable two 
% class dataset consisting of two dimensional data features. 
% The perceptron is run 3 times with 3 initial points to show the 
% recovery of different separating boundaries.  All points and recovered
% as well as boundaries are then visualized.

%%% load data %%%

M = importdata('beer_data.csv', ',', 1);
[D,b] = load_data();

%%% run perceptron for 3 initial points %%%

% Calculate fixed steplength - via Lipschitz constant (see Chap 9 for 
% explanation) - for use in all three runs
lam = 10^2;        % regularization parameter 
L = 2*norm(diag(b)*D')^2;
alpha = 1/(L + 2*lam);        % step length
x0 = [1;2;3];    % initial point
x = grad_descent_soft_SVM(D,b,x0,alpha,lam);

% Run perceptron second time
lam = 10;
alpha = 1/(L + 2*lam);        % step length
y = grad_descent_soft_SVM(D,b,x0,alpha,lam);

% Run perceptron third time
lam = 10^-2;
alpha = 1/(L + 2*lam);        % step length
z = grad_descent_soft_SVM(D,b,x0,alpha,lam);

%%% plot everything, pts and lines %%%
plot_all(D',b,x,y,z);


%%% gradient descent function for perceptron %%%
function x = grad_descent_soft_SVM(D,b,x0,alpha,lam)
    % Initializations 
    x = x0;
    iter = 1;
    max_its = 3000;
    grad = 1;

    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = find_grad(D, b, x, alpha, lam);           % your code goes here!
        x = x - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(A,b,x,y,z)
    
    % plot points 
    ind = find(b == 1);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(b == -1);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    hold on

    % plot separators
    s =[min(A(:,2)):.01:max(A(:,2))];
    plot (s,(-x(1)-x(2)*s)/x(3),'k','linewidth',2);
    hold on

    plot (s,(-y(1)-y(2)*s)/y(3),'g','linewidth',2);
    hold on

    plot (s,(-z(1)-z(2)*s)/z(3),'m','linewidth',2);
    hold on

    set(gcf,'color','w');
    axis([ (min(A(:,2)) - 1) (max(A(:,2)) + 1) (min(A(:,3)) - 1) (max(A(:,3)) + 1)])
    box off
    
    % graph info labels
    xlabel('a_1','Fontsize',14)
    ylabel('a_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)

end

%%% loads data %%%
function [A,b] = load_data()
    data = load('soft_SVM_data.mat');
    data = data.data;
    A = data(:,1:3);
    A = A';
    b = data(:,4);
end

function grad = find_grad(D, b, x, alpha, lam)
    N = numel(b);
    k = numel(x);
%     u1 = 0;
%     u2 = zeros(1, k);
%     u3 = zeros(k, 1);
%     u4 = eye(k);
    in = ones(N, 1) - diag(b)*D'*x;
    in = max(0, in(:));
    % U = [u1, u2; u3, u4];
    U = eye(k);
    U(1,1) = 0;
    grad = -2*D*diag(b)*in + 2*lam*U*x;
end

end
