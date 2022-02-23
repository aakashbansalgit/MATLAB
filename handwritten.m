N = 1
P = 4
n = 1
m = 2
d = [3;7;11;17]
y = [-0.0442;-0.1735;0.3480;-0.1717]
W = [0.4]
a = [1]
b = [1]
phi = [-0.1106;-0.4337;0.87;-0.4293]
x = [1,2;4,3;7,4;9,8]
net = [1.1;2.4;3.7;5.9]
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