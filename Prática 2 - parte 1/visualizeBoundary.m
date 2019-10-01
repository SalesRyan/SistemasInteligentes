function visualizeBoundary(X, y, model, varargin)
%   essa função plota a fronteira de decisão não-linear 
%   aprendida pelo SVM

% Plota os dados de treinamento sobre a fonteira de decisão
plotData(X, y)
title('Fronteira de decisão não-linear para C=1 e sigma=0.1')
% Realiza as predições de classificação 
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plota a frontreira aprendida pelo SVM
hold on
contour(X1, X2, vals, [0.5 0.5], 'b');
hold off;
print -dpng 'fronteira_decisao_nlinear_C1_sigma01.png' -S640,640
end
