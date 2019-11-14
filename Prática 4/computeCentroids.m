function centroids = computeCentroids_deb(X, idx, K)
% Retorna os novos centróides calculados a partir do ponto médio de todas as
% amostras atribuídas a cada centróide.
%   X -> dataset
%   idx -> vetor com os centróides associados a amostras [1..K]
%   K -> número de centróides

[m n] = size(X);

% Inicializando a matrix com as posições dos centróides
centroids = zeros(K, n);

% ====================== Implemente seu código aqui ======================
% Cada linha do vetor de "centroids" corresponde a um centróide
% e cada coluna corresponde as coordenadas (x,y) respectivamente
%

for i=1:K
  xi = X(idx==i,:);
  ck = size(xi,1);
  centroids(i, :) = sum(xi)/ck;
end

% =============================================================

end

