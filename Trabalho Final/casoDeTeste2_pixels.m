%% Inicialização
clear ; close all; clc

%%% Definição da arquitetura da rede
input_layer_size  = 2500;  
hidden_layer_size = 25;   
num_labels = 2;  % as labels foram definidas de 1 a 10, sendo o 10 atribuídas a classe 0 

% Carregando os dados de treinamento
treinoX = csvread('pixelsTreino50x50X.csv');
treinoy = csvread('pixelsTreino50x50y.csv');
X = treinoX(2:end, :);
y = treinoy(2:end, :);
size(X)

% Carregando dados de teste
testeX = csvread('pixelsTeste50x50X.csv');
testey = csvread('pixelsTeste50x50y.csv');
X_ = testeX(2:end, :);
size(X_)
y_ = testey(2:end, :);

m = size(X, 1);

accTrain = [];
accTest = [];
epoch = 0;

for i=1:10
  %Cálcular theta0 e theta1 randomicamente
  epsilon_init = 0.01;
  initial_Theta1 = randn(25, 2501);
  initial_Theta2 = randn(2, 26);

  % transformar as matrizes de pesos em um vetor 
  initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

  %%%%%%%%%%%% Treinando a rede neural  %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  %%  Variar o número máximo de iterações para verificar como o treinamento é 
  %%  influenciado
  options = optimset('MaxIter', 500);

  % Tentar diferentes valores de lambda
  lambda = 0.5;

  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  % Redimensionar Theta1 e Theta2 para as dimensões originais
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

  %fprintf('Aperte enter para cotinuar.\n');
  %pause;

  %%  Depois de treinar a rede neural, você deve utilizar a função predict para 
  %%  predizer as labels do conjunto de treinamento
  predTrain = predict(Theta1, Theta2, X);
  accTrain(i) = mean(double(predTrain == y)) * 100;
  
  predTest = predict(Theta1, Theta2, X_);
  accTest(i) = mean(double(predTest == y_)) * 100;
  epoch += 1
endfor

fprintf('\n\n+------------------ Métricas de Validação ------------------+\n');
fprintf('Acurácia média de Teste: %f\n', mean(accTest));
mediana = median(accTest);
fprintf('Mediana da acurácia de Teste: %f\n', sum(mediana)/length(mediana));
fprintf('Acurácia máxima de Teste: %f\n', max(accTest));
fprintf('Acurácia mínima de Teste: %f\n', min(accTest));
fprintf('Acurácia média de Treinamento: %f\n', mean(accTrain));
fprintf('+-----------------------------------------------------------+\n');

%pred = predict(Theta1, Theta2, X_);
%fprintf('\nAcurácia de Teste: %f\n', mean(double(pred == y_)) * 100);

%% Se eu rodar 10x, por exemplo, a taxa de acurácia será a mesma?
%% Cálcule a média, o máximo e o mínino da taxa de acurácia para 100 repetições 
%% do treinamento?