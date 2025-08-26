function [K,dK] = covPScor(mode,hyp, x, z)
% the correction term for the covariance based on estimated propensity score
% "logit" or "probit" refers to the specification of the propensity score,
% it has nothing to do with the outcome equation or its link function
global likLogis bd ps_bd q coef

%if nargin<1, mode = 'Logit'; end  % default values                                  
mode_list = '''Logit'', ''Probit''';
switch mode
  case 'Logit',   ne = '1';
  case 'Probit',  ne = '1';
  otherwise,   error('Parameter mode is either %s.',mode_list)
end

if nargin<3, K = ne; return; end                   % report number of parameters
if nargin<4, z = []; end                          % make sure, z existsxeq
xeqz = isempty(z); dg = strcmp(z,'diag');          % sort out different modes
[n,p_a] = size(x); %ne = eval(ne);                   % dimensions
p = p_a-1; % dimension of covariates

hyp = hyp(:); % make sure coef and hyp are column vectors

if numel(hyp)~=1, error('Wrong number of hyperparameters'), end



% x(:,p+1) is d (treatment status)
switch mode
    case 'Logit',  ps = @(x) likLogis(coef(1)+x(:,1:q)*coef(2:end)); 
                   dps = @(x) likLogis(x).*(1-likLogis(x));
    case 'Probit',  ps = @(x) normcdf(coef(1)+x(:,1:q)*coef(2:end));
                    dps = @(x) normpdf(x);
end


Riesz = @(x) x(:,p+1)./bd(ps(x(:,1:p)),ps_bd) - (1-x(:,p+1))./(1-bd(ps(x(:,1:p)),ps_bd)); % Riesz representer, output: n * 1 column vector

if dg
   K_0 = Riesz(x).^2; % n-dim column vector
   K =  K_0 * hyp^2;  
else
  if xeqz % if z=[]
     z = x;  
  end
  K_0 = Riesz(x) * Riesz(z)';
  K = K_0 * hyp^2;  
end

if nargout > 1
   dK = @(Q) dirder(Q,K_0,Riesz,ps,dps,hyp,x,z,dg,xeqz); % directional derivative of tr(Q'*K)
end
end


function [dhyp, dx] = dirder(Q,K_0,Riesz,ps,dps,hyp,x,z,dg,xeqz)
   global bd ps_bd q coef
   [n,p_a] = size(x);
   p = p_a-1;
   ii = ones(n,1);
   dhyp = Q(:)'* K_0(:) * 2 * hyp; % d(tr(Q'*K))/dhyp
   dgdx = zeros(n,p); %dg/dx
   for j=1:q
   dgdx(:,q) = ii * coef(j+1);
   end
   dpsdx_part_1 = dps(coef(1)+x(:,1:q)*coef(2:end)); % dim: n * 1
   dpsdx = bsxfun(@times,dpsdx_part_1,dgdx); % the derivative of propensity score pi(x) w.r.t. x (covariate); dim: n * p
   psx_trim = bd(ps(x(:,1:p)),ps_bd);
   dRiedx_part_1 = -(x(:,p+1)./psx_trim.^2 - (1-x(:,p+1))./(1-psx_trim).^2);
   dRiedx = bsxfun(@times,dRiedx_part_1,dpsdx); % the derivative of Riesz rep. w.r.t x (covariate); dim: n*p
   dRiedd = 1./ps_trim + 1./(1-ps_trim); % the derivative of Riesz rep. w.r.t d (treatment); dim: n*1
   dRie = [dRiedx,dRiedd]; % the derivative of Riesz rep. w.r.t (x,d); dim: n* (p+1);
   
   if dg
       y = Q; % dim: n*1 
   else
     if xeqz
       y = (Q + Q') * Riesz(x); % dim: n * 1, Q: n*n
     else
       y = Q * Riesz(z); % dim: n * 1, Q: n*t
     end
   end
   dx = bsxfun(@times,y,dRie) * hyp^2; % d(tr(Q'*K))/d(x,d), dim: n * (p+1);
end