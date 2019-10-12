function word_indices = processEmail(email_contents)

% Carrega o vocabulário
vocabList = getVocabList();

% Inicializa o vetor com os índices das palavras encontradas 
word_indices = [];

% Converte todo o email para letras minúsculas
email_contents = lower(email_contents);

% Remove todas as tags HTML, procurando por qualquer expressão que comece com 
% "<" e termine com ">"
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% Substitui todas os números pela expressão 'number'
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% As URLs são substituídas pela expressão “httpaddr”
% Procura por strings que começam com http:// or https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Busca por strings com @ no meio e substitui pela expressão 'emailaddr'
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Busca pelo símbolo $ e substitui pela expressão 'dollar'
email_contents = regexprep(email_contents, '[$]+', 'dollar');

% Tokenizar o email
% Printa o email na tela 
fprintf('\n==== Email ====\n\n');

l = 0;
cont = 0;

while ~isempty(email_contents)
    cont = cont+1;
    % Quebra a string em palavras e remove qualquer pontuação
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
   
    % Remove qualquer caractere que não seja alpha numérico
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    % Pular a palavra se ela for muito curta
    if length(str) < 1
       continue;
    end

    % Buscar a palavra no dicionário e adicionar o índice na variável
    % word_indices, caso exista
     for p = 1:size(vocabList,1)
      if (strcmp(vocabList{p},str))
        word_indices(end + 1) = p;
      end 
    endfor
    % Imprime na saída, garantindo que as linhas não sejam muito longas  
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

fprintf('\n\n=========================\n');

end
