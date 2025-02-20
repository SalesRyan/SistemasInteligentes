%% Inicialização
clear ; close all; clc

%% Definição da arquitetura da rede
input_layer_size  = 8;  
hidden_layer_size = 25;   
num_labels = 2;  % as labels foram definidas de 1 a 10, sendo o 10 atribuídas a classe 0

% Carregando os dados de treinamento
treinoX = csvread('peleX.csv');
treinoy = csvread('peley.csv');
X = treinoX(2:end, 2:end);
y = treinoy(2:end, 2:end);

%normalizando
X = (X - min(X)/(max(X) - min(X)));
m = size(X, 1);

%Cálcular theta0 e theta1 randomicamente
epsilon_init = 0.12;
initial_Theta1 = randn(25, 9);
initial_Theta2 = randn(2, 26);

% transformar as matrizes de pesos em um vetor 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%%%%%%%%%%%% Treinando a rede neural  %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%  Variar o número máximo de iterações para verificar como o treinamento é 
%%  influenciado
options = optimset('MaxIter', 35);

% Tentar diferentes valores de lambda
lambda = 1;

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

%teste
%testeX = csvread('testeX.csv');
%testey = csvread('testey.csv');
%X_ = testeX(2:end, 2:end);
%y_ = testey(2:end, 2:end);
%X_ = (X_ - min(X_)/(max(X_) - min(X_)));

pred = predict(Theta1, Theta2, X);

fprintf('\nAcurácia de Treinamento: %f\n', mean(double(pred == y)) * 100);

%% Se eu rodar 10x, por exemplo, a taxa de acurácia será a mesma?
%% Cálcule a média, o máximo e o mínino da taxa de acurácia para 100 repetições 
%% do treinamento?