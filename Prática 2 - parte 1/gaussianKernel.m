function sim = gaussianKernel(x1, x2, sigma)
%   sim = gaussianKernel(x1, x2) retorna a similaridade entre x1 e x2 
%   calculada através da função gaussiana, o valor de retorno é armazenado
%   na variável sim

% Garante que x1 e x2 são vetores coluna
x1 = x1(:); x2 = x2(:);

% Inicializa a variável sim 
sim = 0;

sim = exp(-((sum((x1 - x2).^2))/(2*sigma^2)));
    
end
