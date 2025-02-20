% ====================== DATASET/BASE DE DADOS ======================
% Imgine que você é o CEO do Mc Donaldss e deseja abrir 
% uma nova loja da franquia e precisa decidir qual cidade você escolherá. 
% Para te ajudar nessa decisão, você tem dados correspondentes ao lucro de 
% cada franquia e o tamanho da população da cidade na qual ela se 
% encontra. 

%% ================ Parte I: Carrhttps://github.com/SalesRyan/Gradiente-Descendente.gitegando os dados ====================

data = load('exdata.txt');
%característica/entrada/feature
X = data(:, 1);
Orig = data(:, 1); % dados que não serão normalizados

%saida/alvo/target
y = data(:, 2);

m= length(y); 


%% ===== Definição da Função Custo para Regressão Linear ===
function J = computeCost(X, y, theta)
 
  % Inicializando variáveis
  m = length(y);  
  J = 0;

  %hipótese
  h = X*theta;

  %função custo
  J = sum((h - y).^2)/(2*m);
 
endfunction

%% ===== Definição da formula de normalização dos dados ===

function F = normalizacao(X)
 
   minX = min(X);
   maxX = max(X);
   
   F = (X - minX)/(maxX - minX );
   
endfunction

%% ===== Definição do Gradiente Descendente p/  Regressão Linear ===
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
 m = length(y); 
 J_history = zeros(num_iters, 1);
 
 for iter = 1:num_iters
   
   t1 = theta(1) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 1));
   t2 = theta(2) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 2));
   
   theta(1) = t1;
   theta(2) = t2;
   
   J_history(iter) = computeCost(X, y, theta);
   
 endfor
endfunction

%% ================ Parte II: Inicializando os parâmetros do gradiente descendente ====================

% Configurando parâmetros do Gradiente Descendente
X = normalizacao(X);
X = [ones(m, 1), X];

theta = zeros(2, 1); 
iterations = 1500;
alpha = 0.01;

% Mostrar o custo inicial
J0 = computeCost(X, y, theta);

fprintf('O custo inicial é %f\n', J0);

%% ================ Parte III: Treinando o gradiente descendente ====================

[theta,J_history] = gradientDescent(X, y, theta, alpha, iterations);

% Plot 1 (Gráfico do modelo com base nos dados de treinamento):
predict1 = X * theta;
plot(X(:, 2), predict1, X(:, 2), y, 'o');
legend({'Modelo'; 'Dados'});
xlabel('Tamanho da população (dados normalizados)');
ylabel('Lucro');
title('Gráfico do modelo com base nos dados de treinamento');

% Plot 2 (Gráfico de Iterações em Função do Custo, Variando Taxa de Erro):
[theta1,J_history1] = gradientDescent(X, y, theta, 0.3, iterations);
[theta2,J_history2] = gradientDescent(X, y, theta, 0.1, iterations);
[theta3,J_history3] = gradientDescent(X, y, theta, 0.03, iterations);
[theta4,J_history4] = gradientDescent(X, y, theta, 0.01, iterations);

figure
hold on
plot(J_history1)
hold on
plot(J_history2)
hold on
plot(J_history3)
hold on
plot(J_history4)

xlabel('Número de Iterações');
ylabel('Taxa de Erro');
legend({'Alpha = 0.3', 'Alpha = 0.1', 'Alpha = 0.03', 'Alpha = 0.01', 'Dados'});
title('Gráfico de Iterações em Função do Custo');

% Exibe os valores dos parâmetros theta1 e theta2 
fprintf('Parâmetros ótimos do modelo: ');
fprintf('%f %f \n', theta(1), theta(2));

%% ================ Parte IV: Testando o gradiente descendente para uma nova amostra ====================

%Normalizando o valor de teste
normal = (10.0 - min(Orig))/(max(Orig)-min(Orig))

%Predizo lucro da franquia, dado um tamanho da população
predict = [1 normal] *theta;
fprintf('Para uma população de 70.000 mil habitantes, o lucro predito foi %f\n',...
    predict*10000');


%%Testando...

