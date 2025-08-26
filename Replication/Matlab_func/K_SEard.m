function K = K_SEard(sf,ell,X)
% exponential (SE) covariance kernel function 
% Automatic Relevance Detemination
%k(x,z) = sf^2 * exp(-(x-z)'*inv(P)*(x-z)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
% D=dim(X_i)
n = size(X,1);
K = zeros(n,n);
invP = diag(1./ell.^2);

for i=1:n
    for j=1:n
        K(i,j) = sf^2 * exp((X(i,:)- X(j,:))*invP*(X(i,:)'- X(j,:)')/2);
    end
end


end

