clear
close('all')
%%Rate media rate equations, quasi 2-level system

%for ASE see Pedersen, B., The design of erbium-doped fiber amplifiers. 1991, Journal of Lightwave Tech. 

%-----plotting
% numPlots = 11;
% 
% m = floor(sqrt(numPlots));      %rows of plots
% n = ceil(numPlots/m);           %columns of plots
% p = 1;                          %current plotting position
% 
% scrsz = get(0,'ScreenSize');
% f = figure('Position',[1 scrsz(4) 2*scrsz(3)/3 2*scrsz(4)/3], 'Visible', 'off');

%color parameters
nCol = 12;
pcol = [1 0 0];
sigcol = [1 0 1];
asefcol = [0 0 1];
asebcol = [0 1 1];
%-----


%constants
h = 6.62606957E-34;     %J*s
c = 299792458;          %m/s

%parameters
%fibre parameters
L = 1.0;                  %m fiber length
alpha_p = 250/(10/log(10));  %absorption coef. m^-1, 250 dB/m from NuFern YSF-HI

%cross sections, from NKT photonics crossection file
s_ap = 3.04306E-24;     %absorption pump, m^2
s_ep = 3.17025E-24;     %emission pump, m^2
s_as = 0.04966E-24;     %abs signal, m^2
s_es = 0.59601E-24;     %emi signal, m^2

N = alpha_p/s_ap;          %number density of doopant atoms, #/m^3

tau_se = 770E-6;        %spontaneous emission lifetime, s, see Barnard 1994 j quant elec.

%wavelengths
l_p = 0.976E-6;         %pump wavelength, 976nm
l_s = 1.028E-6;         %signal wavelength, 1028nm
v_p = c/l_p;            %pump freq, Hz
v_s = c/l_s;            %signal freq, Hz

dl = 50E-9;             %ase bandwidth

%calculated constants (see notebook 1, page , or equation below)
b_p = (s_ap + s_ep)/(h*v_p);
b_s = (s_as + s_es)/(h*v_s);
a_p = s_ap/(h*v_p);
a_s = s_as/(h*v_s);


%With these defined contsants, rate equation looks like
%dn2/dt = a_p*I_p + a_s*I_s - n2*(b_p*I_p + b_s*I_s + 1/tau_se)



%%
%Position-dependent pump intensity
dz = L/200;                %spacial slice, 2mm
z = 0:dz:L;

P_pi = 0.05;        %lowest pump power (W)
P_pf = 0.65;        %highest pump power (W)

d = 7.5E-6;           %core diameter (m), 7.5um MFD from NuFern

I_0p = (P_pi:P_pi:P_pf)/(pi*(d/2)^2);          %Initial pump intensity, 10kW/cm^2 = 1E8 W/m^2 --> 1W 6um core diameter ~3.5E10W/m^2
I_0s = 4.5E8;                %total seed signal intensity


%DE's for light intensity
dI_p = @(z,I_p,n2) (-(s_ap*N*(1-n2) - s_ep*N*n2)).*I_p;
dI_sig = @(z,I_sig,n2) (-(s_as*N*(1-n2) - s_es*N*n2)).*I_sig;
dI_ase = @(z,I_ase, n2) (-(s_as*N*(1-n2) - s_es*N*n2)).*I_ase + n2*h*v_s*N*s_es*(dl*c/l_s^2);
%note ASE has same DE, but z directions are different, this is implemented
%below

    
%Initialize parameters
I_p = zeros(length(I_0p),length(z));        %pump intensity
I_s = zeros(length(I_0p),length(z));        %combind signal intensity (signal, ase)
I_sig = zeros(length(I_0p),length(z));      %singal (of interest) intensity
I_asef = zeros(length(I_0p),length(z));     %forward ase intensity, also initial cond.
I_aseb = zeros(length(I_0p),length(z));     %forward ase intensity, also initial cond.
n_2 = zeros(length(I_0p),length(z));        %excited state population
dg = zeros(length(I_0p),length(z));         %gain contribution
g = zeros(length(I_0p),length(z));          %gain coefficient
G = zeros(length(I_0p),length(z));          %gain  
a = 0;                                      %loss coeff. not currently used

%initial Pump/Signal conditions
I_p(:,1) = I_0p;
I_sig(:,1) = I_0s;
    
    
    %while-loop determinants
j = 0;
cGain = zeros(length(I_0p),1);
errG = 1;
err = [];
    
    
%Main loop
while j < 500 && abs(errG) > 1E-12

pGain = cGain;    

%RK4 method
%single-pass propagation    
for i = 1:length(z)
k = length(z) - i + 1;

    I_s(:,i) = I_sig(:,i) + I_asef(:,i) + I_aseb(:,i);
    n_2(:,i) = (a_p*I_p(:,i) + a_s*I_s(:,i))./(b_p*I_p(:,i) + b_s*I_s(:,i) + 1/tau_se);

    if i<length(z)
        
        %update I_p
        k1 = dI_p(z(i), I_p(:,i), n_2(:,i));
        k2 = dI_p(z(i) + dz/2, I_p(:,i) + k1*dz/2, n_2(:,i));
        k3 = dI_p(z(i) + dz/2, I_p(:,i) + k2*dz/2, n_2(:,i));
        k4 = dI_p(z(i) + dz, I_p(:,i) + k3*dz, n_2(:,i));
        I_p(:,i+1) = I_p(:,i) + (k1+2*k2+2*k3+k4)*dz/6;
        
        %update I_sig
        k1 = dI_sig(z(i), I_sig(:,i), n_2(:,i));
        k2 = dI_sig(z(i) + dz/2, I_sig(:,i) + k1*dz/2, n_2(:,i));
        k3 = dI_sig(z(i) + dz/2, I_sig(:,i) + k2*dz/2, n_2(:,i));
        k4 = dI_sig(z(i) + dz, I_sig(:,i) + k3*dz, n_2(:,i));
        I_sig(:,i+1) = I_sig(:,i) + (k1+2*k2+2*k3+k4)*dz/6;
        
        %update I_asef
        k1 = dI_ase(z(i), I_asef(:,i), n_2(:,i));
        k2 = dI_ase(z(i) + dz/2, I_asef(:,i) + k1*dz/2, n_2(:,i));
        k3 = dI_ase(z(i) + dz/2, I_asef(:,i) + k2*dz/2, n_2(:,i));
        k4 = dI_ase(z(i) + dz, I_asef(:,i) + k3*dz, n_2(:,i));
        I_asef(:,i+1) = I_asef(:,i) + (k1+2*k2+2*k3+k4)*dz/6;
        
        %update I_aseb
        k1 = dI_ase(z(k), I_aseb(:,k), n_2(:,k));
        k2 = dI_ase(z(k) + dz/2, I_aseb(:,k) + k1*dz/2, n_2(:,k));
        k3 = dI_ase(z(k) + dz/2, I_aseb(:,k) + k2*dz/2, n_2(:,k));
        k4 = dI_ase(z(k) + dz, I_aseb(:,k) + k3*dz, n_2(:,k));
        I_aseb(:,k-1) = I_aseb(:,k) + (k1+2*k2+2*k3+k4)*dz/6;
        

    end
    
   %single-pass gain
    g(:,i) = (s_es*N*n_2(:,i) - s_as*N*(1-n_2(:,i))); 
    dg(:,i) = g(:,i)*dz;
    G(:,i) = exp(sum(dg(:,1:i),2));
    
    cGain = G(:,end);
    
end

errG = max(abs((cGain - pGain)./cGain));
err = [err, errG];

j = j+1;

end

%display final loop count and err
display(['Number of loops: ',num2str(j)]);
display(['Gain error: ', num2str(errG)]);


%%
%Plotting
pumpPlot = figure('Name', 'Pump Power');
signalPlot = figure('Name', 'Signal Power');
asePlot = figure('Name', 'ASE Power');
nPlot = figure('Name', 'Popultion Excitation');

col = hsv(length(I_0p));

for i = 1:length(I_0p)
    
figure(pumpPlot)
hold on
plot(z, I_p(i,:), 'Color', col(i,:));

figure(signalPlot)
hold on
plot(z, I_sig(i,:), 'Color', col(i,:));

figure(asePlot)
hold on
plot(z, I_asef(i,:),'-', 'Color', col(i,:));
plot(z, I_aseb(i,:),'-.', 'Color', col(i,:));

figure(nPlot)
hold on
plot(z, n_2(i,:), 'Color', col(i,:));

end




