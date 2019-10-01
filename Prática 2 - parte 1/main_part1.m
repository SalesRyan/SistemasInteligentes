%% Inicialização
clear ; close all; clc

%Carregando os dados
load('ex6data1.mat');

% Plotando o dataset 1
% X:matriz de características dos exemplos de treinamento
% Y: alvo (+) positivo (o) negativo
%plotData(X, y);

% Treinando o SVM linear
C = 1;
% @linearKernel -> função kernel
% 1e-3 -> determina o limite de casas para comparar 2 números flutuantes
% 10 -> número de iterações
model1 = svmTrain(X, y, 1, @linearKernel, 1e-3, 10);
model2 = svmTrain(X, y, 10, @linearKernel, 1e-3, 10);
model3 = svmTrain(X, y, 100, @linearKernel, 1e-3, 10);
model4 = svmTrain(X, y, 1000, @linearKernel, 1e-3, 10);

models = [model1, model2, model3, model4]

for i = 1:4
  w = models(i).w;
  b = models(i).b;
  xp = linspace(min(X(:,1)), max(X(:,1)), 100);
  yp = - (w(1)*xp + b)/w(2);
  hold on;
  plot(xp, yp); 
  %hold off
  print -dpng 'fronteira_decisao_C1.png' -S640,640
end

legend({'C = 1', 'C = 10', 'C = 100', 'C = 1000'});
title('Fronteira de decisão linear')
hold on
plotData(X, y);

%Função que plota a fonteira de decisão
%visualizeBoundaryLinear(X, y, model);

% Avaliando a função Kernel que deve ser implementada em gaussianKernel.m

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf('O valor da função gaussiana é ...\n') 
sim

% a fim de validar o resultado, o valor retornado por sim para os parâmetros 
% acima deve 0.32465

% Carregando e vizualizando o dataset2 
load('ex6data2.mat');

% Plotando o dataset2
% X:matriz de características dos exemplos de treinamento
% Y: alvo (+) positivo (o) negativo

%plotData(X, y);
%fprintf('Isso pode levar alguns minutinhos ...\n');

% SVM Parameters
C = 1; sigma = 0.1;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%visualizeBoundary(X, y, model);

