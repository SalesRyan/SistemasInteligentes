function sim = linearKernel(x1, x2)
%Essa função retorna a combinação linear entre duas características

% Garante que X1 e X2 são vetores coluna
x1 = x1(:); x2 = x2(:);

% Essa combinação é resultado da multiplicação elemento
% a elemento do vetor, resultando no somatório das posições 
% do vetor resultante
sim = x1' * x2; 

end