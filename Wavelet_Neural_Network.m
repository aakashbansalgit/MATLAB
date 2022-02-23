clear all
close all
clc
%SET DATA PARAMETERS
P = 6 % Number of rows in the input data
m = 2 % Number of input variables
n = 20 % Number of Hidden Units we want to create in the analysis
N = 1 % Number of output variables

%%%%%%%%%%%%% MATRiX DEFENITION
% a(n) b(n) matrix define the dilation and the translation parameters respectively
% x(P,m) matrix defines the input variable values in P*m shaped matrix
% net(P,n) matrix defines the value of output from input variable
% y(P,N) matrix defines the output variable values obtained after the analysis in P*N shaped matrix
% d(P,N) matrix defines the actual values of the output variable in P*N shaped matrix
% phi(P,n) matrix defines the output of the hidden units in P*n shaped matrix
% W(N,n) matrix defines the weight values in solving the output from the hidden units 
% WW(n,m) matrix defines the weight values in solving the hidden units from the input variables
%%%%%%%%%%%%%

%%%%%%%% INPUT FROM EXCEL
% filename = 'rainfall.xlsx'
% sheet = 'sheet1' [name of the sheet within the excel file]
% xlRange1 = 'A:B'
% xlRange2 = 'C'
% x = xlsread(filename, sheet, xlRange1)
% d = xlsread(filename, sheet, xlRange2)
%%%%%%%%

x=[2,4;4,6;1,4;5,6;5,3;3,8]
d=[3;7;3;4;7;9]
W=rand(N,n)
WW=rand(n,m)
a=ones(1,n)
for j=1:n
b(j)=j*P/n;
end

%%%%%%%%%%%%%% WEIGHT GRADIENTS
% EW(N,n) change in W(N,n) in the next iteration
% EWW(n,m) change in WW(n,m) in the next iteration
% Ea(n) change in a(n) in the next iteration
% Eb(n) change in b(n) in the next iteration
%%%%%%%%%%%%%%

epoch=1; % First iteration (variable to loop through number of iterations)
epo=10000; % Number of iterations
error=0.05; % Initial Error Value (Anything greater than err value) 
err=0.001; % Target Error Value, we stop the analysis if error goes below this value
delta =1; 
lin=0.1; %learning Rate
while (error>=err & epoch<=epo)
     
     u=0;% MIDDLE VARIABLE 
     %%%% Calculating the NET INPUT
     for p=1:P
         for j=1:n
             u=0;
              for k=1:m
                  u=u+WW(j,k)*x(p,k);
              end
              net(p,j)=u;
              
         end
     end
     
     %Have inserted formulas for both morlet and mexican hat wavelets
     %currenty using the mexican hat wavelet
          for p=1:P
              for j=1:n
                  u=net(p,j);
                  u=(u-b(j))/a(j);
                  %phi(p,j)=cos(1.75*u)*exp(-u*u/2); %FOR MORLET WAVELET
                  phi(p,j)=(1-u^2)*exp(-u*u/2); %FOR MEXICAN HAT WAVELET
              end
          end
          
      %calculation of NETWORK OUTPUT
         for p=1:P
             for i=1:N
                 u=0;
                 for j=1:n
                     u=u+W(i,j)*phi(p,j);
                 end
                 y(p,i)=delta*abs(u);
             end
         end
         
       %calculation of OUTPUT ERROR
       u=0;
     for p=1:P
         for i=1:N
            %u=u+abs(d(p,i)*log(y(p,i))+(1-d(p,i)*log(1-y(p,i)))); %FOR MORLET WAVELET
            u=u+(d(p,i)-y(p,i))^2;
         end
     end
     
     %u=u/2 %FOR MORLET WAVELET
     
     error=u;
     
     %calculate of NETWORK GRADIENTS
     for i=1:N
         for j=1:n
             u=0;
             for p=1:P
                 u=u+(d(p,i)-y(p,i))*phi(p,j);
             end
             EW(i,j)=u; %FOR MEXICAN HAT WAVELET
             %EW(i,j)=-u; %FOR MORLET WAVELET
         end
     end
     
     for j=1:n
         for k=1:m
             u=0;
             for p=1:P
                 for i=1:N
                    u=u+(d(p,i)-y(p,i))*W(i,j)*phi(p,j)*x(p,k)/a(j) ;
                 end
             end
             EWW(j,k)=u; %FOR MEXICAN HAT WAVELET
             %EWW(j,k)=u %FOR MORLET WAVELET
         end 
     end 
     
     for j=1:n
             u=0;
             for p=1:P
                 for i=1:N
                    u=u+(d(p,i)-y(p,i))*W(i,j)*phi(p,j)/a(j) ;
                 end
             end
             Eb(j)=u;
     end 
     
     for j=1:n
             u=0;
             for p=1:P
                 for i=1:N
                    u=u+(d(p,i)-y(p,i))*W(i,j)*phi(p,j)*((net(p,j)-b(j))/b(j))/a(j) ;
                 end
             end
             Ea(j)=u;
     end 
     
    %WEIGHT ADJUSTMENT
    WW=WW-lin*EWW;
    W=W-lin*EW;
    a=a-lin*Ea;
    b=b-lin*Eb;
    
     %ITERATION INCREASED BY 1
     epoch=epoch+1;
     
end

plot(x,d)
hold on
plot(x,y,'--')

d
y
