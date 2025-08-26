function varargout = covPS(varargin)

% Wrapper for PS correction covariance function covPScor.m. 

varargout = cell(max(1,nargout),1);
[varargout{:}] = covPScor('Probit',varargin{:});
% q=3 the order of poly in the probit regression for the propensity score
end

