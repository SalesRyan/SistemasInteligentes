function visualizeBoundaryLinear(X, y, model)
% Essa função plota a fronteira de decisão linear 
% definida no treinamento do SVM

% f(x) = wx + b
% x -> amostras de treino
% w -> vetor de coeficientes gerado pelo modelo treinado
% wx -> produto escalar, onde é calculada a soma dos produtos dos componentes vetoriais 
% b -> uma constante que define o viés do modelo

w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
plotData(X, y);
title('Fronteira de decisão para C=1')
hold on;
plot(xp, yp, '-b'); 
hold off
print -dpng 'fronteira_decisao_C1.png' -S640,640
end
