function idx = findClosestCentroids(X, centroids)
% essa função associa o centróide mais próximo a cada exemplo de treinamento e 
% guarda o valor associado no vetor idx. Portanto, se temos m exemplos de 
% treinamento, o vetor deve possuir a dimensão idx = m x 1 
%

% Definindo o valor de K
K = size(centroids, 1);

% inicializando idx
idx = zeros(size(X,1), 1);

% ====================== Implemente seu código aqui ======================
% idx(i) deve conter o índice do centróide mais próximo do exemplo i. 
% Portanto, deve ser um valor dentro do intervalo de 1 a K
%

m = size(X,1);

for i=1:m
  k = 1;
  dist_min = sum((X(i,:) - centroids(1,:)) .^ 2);
  for j=2:K
      dist = sum((X(i,:) - centroids(j,:)) .^ 2);
      if(dist < dist_min)
        dist_min = dist;
        k = j;
      end
  end
  idx(i) = k;
end

% =============================================================

end

