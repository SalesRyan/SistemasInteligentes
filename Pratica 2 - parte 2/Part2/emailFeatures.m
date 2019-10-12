function x = emailFeatures(word_indices)

% Número total de palavras do dicionário
n = 1899;

% Inicializando o vetor de características
x = zeros(n, 1);

for i = 1:n
  if(any(word_indices == i) == 1)
    x(i) = 1;
  end
endfor

% Pegue o word_indices e contrua um vetor de característica binário 
% que indica se uma determinada palavra ocorre (1) ou não ocorre (0) no e-mail.

   
endfunction
