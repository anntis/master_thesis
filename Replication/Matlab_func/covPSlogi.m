function varargout = covPSlogi(varargin)

% Wrapper for PS correction covariance function covPScor.m. 

varargout = cell(max(1,nargout),1);
[varargout{:}] = covPScor('Logit',varargin{:});
% q=3 the order of poly in the logit regression for the propensity score
end

