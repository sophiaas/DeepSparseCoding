"""
%%      Lloyd-Max Algorithm in matlab

function [ sk,lu ] = cent( mint,mfin,sig,k,levels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lu=0;
sk=0;
for j=1:10000
    if(sig(j)<mfin && sig(j)>=mint)
        lu=lu+1;
        sk=sk+sig(j);
    end
end
if lu==0
    sk=(mint+mfin)/2;
    lu=1;
end
% if (lu==0 && k <= levels/2)
%     sk = mint;
%     lu=1;
% elseif (lu == 0 && k > levels/2)
%     sk = mfin;
%     lu=1;
% end

end


clear all
close all
clc

i = 1;
switch i
    case 1 
        load Dat_1.mat
        
    case 2
        load Dat_1.mat
end

N = 10000;
f = 1/N;
iterations = 12;
bits = 3;
levels = 2^bits;
m_max = 3.9;
delta = (2*m_max)/levels;
m = linspace(-m_max, m_max, levels+1);

for i = 1:iterations
    m
    for k = 1:levels
        
        sum=0;
        count=0;
        for j = 1:N
            if m(k) <= X(j) && X(j) < m(k+1)
                count = count + 1;
                sum = sum + X(j);
            end
        end
        
        if count == 0 && k <= levels/2
            v(k) = m(k);
        else
        if count == 0 && k > levels/2
            v(k) = m(k+1);
        else v(k) = sum/count;
        end
        end
        
    end
    
    for k = 2:levels-1
        m(k) = (v(k) + v(k+1))/2;
    end
   v     
end     

function [ mu ] = mew( min,max )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
ts=0.01;
intn = 0;
intd = 0;
for i=min:ts:max
    intn = intn+ts*i*exp(-i*i);
    intd = intd+ts*exp(-i*i);
end
mu = intn/intd;
end

LMAX
clear all;
close all;
for q=1:6%%no. of bits
    n_bits = q;
    q_levels = 2^n_bits;%% no of levels
    minv = -10;%% dynamic range
    maxv = 10;
    len=(-1*minv+maxv)/q_levels;%%length of interval
    m=zeros(1,q_levels+1);
    for i=1:q_levels+1
        m(i)=minv+(i-1)*len;%%initialize
    end
    load('Dat_2.mat');
    sig=X;
    sig=sort(sig);
    lu=zeros(1,q_levels);
    sk=zeros(1,q_levels);
    for i=1:100%%iterations
        for k=1:q_levels
            [sk(k),lu(k)]=cent(m(k),m(k+1),sig,k,q_levels);
            new(k)=sk(k)/lu(k);%%centroid
        end
        for k=2:q_levels
            m(k)=(new(k-1)+new(k))/2;%%new intervals
        end
        for h=1:q_levels%%MSE calculation
            for t=1:10000
                if(X(t)<m(h+1) && X(t)>=m(h))
                    Y(t)=new(h);
                end
            end
        end
        a=X-Y;
        b=a.^2;
        mse1(i)=sum(b)/10000;
    end
    plot(mse1)
    hold on
end
"""