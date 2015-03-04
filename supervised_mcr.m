clear all;
close all;
clc;
% Machine-written character recognition using a supervised classification
% approach
% Designed and Developed by: Ahmad Pahlavan Tafti & Mahya Sheikhzadeh
%inputs:
%a
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;1 0 0 1 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %b
% Pattern_Input=[1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %c
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %d
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;1 1 0 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %e
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 1 1 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %f
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 0 1 0 0 0 0 1;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %g
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;1 0 0 0 0 1 0 0 0 0 1 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0]
% %h
% Pattern_Input=[1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %i
% Pattern_Input=[0 0 0 0 1 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 1 0 1 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %j
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 1 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0]
% %k
% Pattern_Input=[1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 1 0 1 1 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %l
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 1;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %m
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 1 0 0 0 0 0;1 0 1 1 0 1 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %n
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 1 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %o
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %p
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 1 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %q
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 0 0 0 0 0 0;1 0 0 1 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0] 
% %r
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %s
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 1 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %t
% Pattern_Input=[1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %u
Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 1 1 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %v
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
% %w
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 1 1 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 1 0 0 0;0 0 1 0 0 0 0 0 1 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %x
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 1 1 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0] 
% %y
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0] 
% %z
% Pattern_Input=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 1 1 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 1 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0]
tic
Pattern_Input= Pattern_Input'

% LVQ1
P1=[0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 0 0 1 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 1];
P1=P1';
TC1=[1 2 3 4];
T1 = ind2vec(TC1);
targets = full(T1);
net1 = newlvq(minmax(P1),4,[.25 .25 .25 .25]);
net1 = train(net1,P1,T1);
Y1 = sim(net1,Pattern_Input(:,1))
Yc1 = vec2ind(Y1);

% LVQ2
P2=[0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 1 0 0 0 0 1 0 0 0 0 1;0 0 0 0 1 0 0 0 0 0 0 0];
P2=P2';
TC2=[1 2 3 4];
T2 = ind2vec(TC2);
targets = full(T2);
net2 = newlvq(minmax(P2),4,[.25 .25 .25 .25]);
net2 = train(net2,P2,T2);
Y2 = sim(net2,Pattern_Input(:,2));
Yc2 = vec2ind(Y2);

% LVQ3
P3=[0 0 1 0 0 0 0 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;0 1 1 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 1 0 0 1 1 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 1 0 0;1 1 0 0 0 1 1 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;0 1 1 0 0 0 1 0 0 0 0 0];
P3=P3';
TC3=[1 2 3 4 5 6 7 8 9 10 11 12];
T3 = ind2vec(TC3);
targets = full(T3);
net3 = newlvq(minmax(P3),12,[1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12]);
net3 = train(net3,P3,T3);
Y3 = sim(net3,Pattern_Input(:,3))
Yc3 = vec2ind(Y3);

% LVQ4
P4=[1 0 0 1 0 1 1 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;1 1 0 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 1 0;1 1 0 0 0 1 1 0 0 0 0 0;1 0 0 0 0 1 0 0 0 0 1 0;1 1 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;1 0 1 1 0 1 0 0 0 0 0 0;1 0 0 1 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 1 1 0 0 1 0 0 0 0 0];
P4=P4';
TC4=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
T4 = ind2vec(TC4);
targets = full(T4);
net4 = newlvq(minmax(P4),15,[1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15 1/15]);
net4 = train(net4,P4,T4);
Y4 = sim(net4,Pattern_Input(:,4))
Yc4 = vec2ind(Y4);

% LVQ5
P5=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0];
P5=P5';
TC5=[1 2 3];
T5 = ind2vec(TC5);
targets = full(T5);
net5 = newlvq(minmax(P5),3,[1/3 1/3 1/3]);
net5 = train(net5,P5,T5);
Y5 = sim(net5,Pattern_Input(:,5))
Yc5 = vec2ind(Y5);

% LVQ6
P6=[0 0 0 1 0 0 0 0 0 0 0 0;1 1 0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 1 1 0 1 1 0 0 0 0 0 0;1 0 0 1 0 1 1 0 0 0 0 0;1 1 0 0 0 0 1 0 0 0 0 0;1 1 0 0 0 1 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;0 0 0 0 0 0 0 0 1 0 0 0;0 0 1 0 0 1 1 0 0 0 0 0;0 0 0 0 0 0 0 0 0 1 0 0];
P6=P6';
TC6=[1 2 3 4 5 6 7 8 9 10 11 12];
T6 = ind2vec(TC6);
targets = full(T6);
net6 = newlvq(minmax(P6),12,[1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12 1/12]);
net6 = train(net6,P6,T6);
Y6 = sim(net6,Pattern_Input(:,6));
Yc6 = vec2ind(Y6);

% LVQ7
P7=[1 0 1 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 1 0 0 0 0 0;1 1 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;1 0 1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 0 0 0 0 0 0;0 0 0 1 0 0 1 0 0 0 0 0;1 0 1 0 0 1 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0 1 0;0 1 0 0 0 0 1 0 0 0 0 0;1 0 1 1 0 1 1 0 0 0 0 0;0 0 1 0 0 0 0 0 1 0 0 0];
P7=P7';
TC7=[1 2 3 4 5 6 7 8 9 10 11 12 13 14];
T7 = ind2vec(TC7);
targets = full(T7);
net7 = newlvq(minmax(P7),14,[1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14 1/14]);
net7.trainParam.epochs = 300;
net7.trainParam.goal = 0.01;
net7 = train(net7,P7,T7);
Y7 = sim(net7,Pattern_Input(:,7));
Yc7 = vec2ind(Y7);

% LVQ8
P8=[0 0 0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0];
P8=P8';
TC8=[1 2 3];
T8 = ind2vec(TC8);
targets = full(T8);
net8 = newlvq(minmax(P8),3,[1/3 1/3 1/3]);
net8 = train(net8,P8,T8);
Y8 = sim(net8,Pattern_Input(:,8));
Yc8 = vec2ind(Y8);

% LVQ9
P9=[0 0 0 0 0 0 0 0 0 0 0 0;0 1 0 0 0 0 1 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0];
P9=P9';
TC9=[1 2 3];
T9 = ind2vec(TC9);
targets = full(T9);
net9 = newlvq(minmax(P9),3,[1/3 1/3 1/3]);
net9 = train(net9,P9,T9);
Y9 = sim(net9,Pattern_Input(:,9));
Yc9 = vec2ind(Y9);

% LVQ10
P10=[0 0 0 0 0 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 1 0 0 0 0 0];
P10=P10';
TC10=[1 2 3];
T10 = ind2vec(TC10);
targets = full(T10);
net10 = newlvq(minmax(P10),3,[1/3 1/3 1/3]);
net10 = train(net10,P10,T10);
Y10 = sim(net10,Pattern_Input(:,10));
Yc10 = vec2ind(Y10);

% makeing AND
W26=[1 1 1 1 1 1 1 1 1 1];
b26=[-10];

I1=[Y1(1,1);Y2(1,1);Y3(1,1);Y4(1,1);Y5(1,1);Y6(1,1);Y7(1,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n1 = W26*I1+b26
if n1>=0 
    disp('Character "a" is detected')
end

I2=[Y1(2,1);Y2(1,1);Y3(2,1);Y4(2,1);Y5(1,1);Y6(2,1);Y7(2,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n2 = W26*I2+b26
if n2>=0 
    disp('Character "b" is detected')
end

I3=[Y1(1,1);Y2(1,1);Y3(1,1);Y4(3,1);Y5(1,1);Y6(1,1);Y7(3,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n3 = W26*I3+b26
if n3>=0 
    disp('Character "c" is detected')
end

I4=[Y1(1,1);Y2(2,1);Y3(1,1);Y4(4,1);Y5(1,1);Y6(1,1);Y7(4,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n4 = W26*I4+b26
if n4>=0 
    disp('"d" is detected')
end

I5=[Y1(1,1);Y2(1,1);Y3(3,1);Y4(5,1);Y5(1,1);Y6(1,1);Y7(3,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n5 = W26*I5+b26
if n5>=0 
    disp('Character "e" is detected')
end

I6=[Y1(1,1);Y2(3,1);Y3(4,1);Y4(6,1);Y5(1,1);Y6(3,1);Y7(5,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n6 = W26*I6+b26
if n6>=0 
    disp('"f" is detected')
end

I7=[Y1(1,1);Y2(1,1);Y3(1,1);Y4(7,1);Y5(1,1);Y6(1,1);Y7(6,1);Y8(1,1);Y9(2,1);Y10(2,1)];
n7 = W26*I7+b26
if n7>=0 
    disp('Character "g" is detected')
end

I8=[Y1(2,1);Y2(1,1);Y3(2,1);Y4(8,1);Y5(1,1);Y6(4,1);Y7(5,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n8 = W26*I8+b26
if n8>=0 
    disp('Character "h" is detected')
end

I9=[Y1(3,1);Y2(1,1);Y3(5,1);Y4(9,1);Y5(1,1);Y6(5,1);Y7(7,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n9 = W26*I9+b26
if n9>=0 
    disp('Character "i" is detected')
end

I10=[Y1(1,1);Y2(4,1);Y3(4,1);Y4(10,1);Y5(1,1);Y6(3,1);Y7(8,1);Y8(1,1);Y9(2,1);Y10(2,1)];
n10 = W26*I10+b26
if n10>=0 
    disp('Character "j" is detected')
end

I11=[Y1(2,1);Y2(1,1);Y3(2,1);Y4(11,1);Y5(1,1);Y6(6,1);Y7(9,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n11 = W26*I11+b26
if n11>=0 
    disp('Character "k" is detected')
end

I12=[Y1(4,1);Y2(1,1);Y3(6,1);Y4(9,1);Y5(1,1);Y6(7,1);Y7(7,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n12 = W26*I12+b26
if n12>=0 
    disp('Character "l" is detected')
end

I13=[Y1(1,1);Y2(1,1);Y3(6,1);Y4(12,1);Y5(2,1);Y6(4,1);Y7(5,1);Y8(2,1);Y9(1,1);Y10(1,1)];
n13 = W26*I13+b26
if n13>=0 
    disp('Character "m" is detected')
end

I14=[Y1(1,1);Y2(1,1);Y3(6,1);Y4(8,1);Y5(1,1);Y6(4,1);Y7(5,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n14 = W26*I14+b26
if n14>=0 
    disp('Character "n" is detected')
end

I15=[Y1(1,1);Y2(1,1);Y3(1,1);Y4(2,1);Y5(1,1);Y6(1,1);Y7(2,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n15 = W26*I15+b26
if n15>=0 
    disp('Character "o" is detected')
end

I16=[Y1(1,1);Y2(1,1);Y3(7,1);Y4(2,1);Y5(1,1);Y6(8,1);Y7(2,1);Y8(1,1);Y9(3,1);Y10(1,1)];
n16 = W26*I16+b26
if n16>=0 
    disp('Character "p" is detected')
end

I17=[Y1(1,1);Y2(1,1);Y3(8,1);Y4(13,1);Y5(1,1);Y6(1,1);Y7(10,1);Y8(1,1);Y9(1,1);Y10(3,1)];
n17 = W26*I17+b26
if n17>=0 
    disp('Character "q" is detected')
end

I18=[Y1(1,1);Y2(1,1);Y3(6,1);Y4(9,1);Y5(1,1);Y6(4,1);Y7(7,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n18 = W26*I18+b26
if n18>=0 
    disp('Character "r" is detected')
end

I19=[Y1(1,1);Y2(1,1);Y3(9,1);Y4(14,1);Y5(1,1);Y6(9,1);Y7(11,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n19 = W26*I19+b26
if n19>=0 
    disp('Character "s" is detected')
end

I20=[Y1(2,1);Y2(1,1);Y3(10,1);Y4(9,1);Y5(1,1);Y6(2,1);Y7(12,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n20 = W26*I20+b26
if n20>=0 
    disp('Character "t" is detected')
end

I21=[Y1(1,1);Y2(1,1);Y3(5,1);Y4(10,1);Y5(1,1);Y6(1,1);Y7(13,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n21 = W26*I21+b26
if n21>=0 
    disp('Character "u" is detected')
end

I22=[Y1(1,1);Y2(1,1);Y3(11,1);Y4(11,1);Y5(1,1);Y6(1,1);Y7(2,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n22 = W26*I22+b26
if n22>=0 
    disp('Character "v" is detected')
end

I23=[Y1(1,1);Y2(1,1);Y3(5,1);Y4(15,1);Y5(3,1);Y6(10,1);Y7(14,1);Y8(3,1);Y9(1,1);Y10(1,1)];
n23 = W26*I23+b26
if n23>=0 
    disp('Character "w" is detected')
end

I24=[Y1(1,1);Y2(1,1);Y3(11,1);Y4(11,1);Y5(1,1);Y6(11,1);Y7(9,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n24 = W26*I24+b26
if n24>=0 
    disp('Character "x" is detected')
end

I25=[Y1(1,1);Y2(1,1);Y3(5,1);Y4(10,1);Y5(1,1);Y6(1,1);Y7(10,1);Y8(1,1);Y9(2,1);Y10(2,1)];
n25 = W26*I25+b26
if n25>=0 
    disp('Character "y" is detected')
end

I26=[Y1(1,1);Y2(1,1);Y3(12,1);Y4(5,1);Y5(1,1);Y6(12,1);Y7(12,1);Y8(1,1);Y9(1,1);Y10(1,1)];
n26 = W26*I26+b26
if n26>=0 
    disp('Character "z" is detected')
end

% if no one of the outputs is not equal, minimum one of the LVQs does not cluster correctly
probout=max([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25,n26]);
if probout==n1
    disp('Character "a" is detected')
end

if probout==n2
    disp('Character "b" is detected')
end

if probout==n3
    disp('Character "c" is detected')
end

if probout==n4
    disp('Character "d" is detected')
end

if probout==n5
    disp('Character "e" is detected')
end

if probout==n6
    disp('Character "f" is detected')
end

if probout==n7
    disp('Character "g" is detected')
end

if probout==n8
    disp('Character "h" is detected')
end

if probout==n9
    disp('Character "i" is detected')
end

if probout==n10
    disp('Character "j" is detected')
end

if probout==n11
    disp('Character "k" is detected')
end

if probout==n12
    disp('Character "l" is detected')
end

if probout==n13
    disp('"m" is detected')
end

if probout==n14
    disp('Character "n" is detected')
end

if probout==n15
    disp('Character "o" is detected')
end

if probout==n16
    disp('Character "p" is detected')
end

if probout==n17
    disp('Character "q" is detected')
end

if probout==n18
    disp('Character "r" is detected')
end

if probout==n19
    disp('"s" is detected')
end

if probout==n20
    disp('Character "t" is detected')
end

if probout==n21
    disp('Character "u" is detected')
end

if probout==n22
    disp('Character "v" is detected')
end

if probout==n23
    disp('Character "w" is detected')
end

if probout==n24
    disp('Character "x" is detected')
end

if probout==n25
    disp('Character "y" is detected')
end

if probout==n26
    disp('Character "z" is detected')
end

fprintf('Total elapsed Time equals %d Sec.\n', toc);


