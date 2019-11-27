treinoX = csvread('treinoX.csv');
treinoy = csvread('treinoy.csv');

t = treinoX(2:end, 2:end);
u = treinoy(2:end, 2:end);

size(t)
size(u)
size(t, 1)