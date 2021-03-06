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
%crystal parameters
L = 2;                  %m fiber length
%N = 6.1815E25; %30.1*10^24;  %number density of doopant atoms, #/m^3
abs_p = 5.10/(10/log(10));
alpha_p = 0.1;          %additional pump loss

%cross sections, from Nufern file for Gen VIII fiber.
s_ap = 1.7806E-24;     %absorption pump, m^2
s_ep = 1.7623E-24;     %emission pump, m^2
s_as = 0.0562E-24;     %abs signal, m^2
s_es = 0.7634E-24;     %emi signal, m^2

tau_se = 770E-6;        %spontaneous emission lifetime, s, see Barnard 1994 j quant elec.

dCore = 30E-6;           %core diameter (m)
dClad = 250E-6;
MFA = pi*(dCore/2)^2;    %dopant mode field area, ~core size
Ap = pi*(dClad/2)^2;
As = pi*(dCore/2)^2;

Gp = MFA/Ap;    %Pump overlap w 
Gs = MFA/As;    %signal overlap

N = (abs_p/s_ap)/Gp;

%wavelengths
l_p = 0.976E-6;         %pump wavelength, m
l_s = 1.030E-6;         %signal wavelength, m
v_p = c/l_p;            %pump freq, Hz
v_s = c/l_s;            %signal freq, Hz


%calculated constants (see notebook 1, page , or equation below)
b_p = (s_ap + s_ep)/(h*v_p);
b_s = (s_as + s_es)/(h*v_s);
a_p = s_ap/(h*v_p);
a_s = s_as/(h*v_s);


%With these defined contsants, rate equation looks like
%dn2/dt = a_p*I_p + a_s*I_s - n2*(b_p*I_p + b_s*I_s + 1/tau_se)



%%
%Position-dependent pump intensity
dz = 0.002;                %spacial slice, 2mm
z = 0:dz:L;

P_pi = 5;        %lowest pump power (W)
P_pf = 30;        %highest pump power (W)

%s_ap = s_ap*(dCore/dClad)^2;
%s_ep = s_ep*(dCore/dClad)^2;

FSeed = 500E3; %in rep rate s^-1
ESeed = 100E-9; % in J

I_0p = (P_pi:P_pi:P_pf)/Ap;          %Initial pump intensity, 10kW/cm^2 = 1E8 W/m^2 --> 1W 6um core diameter ~3.5E10W/m^2
I_0s = (ESeed*FSeed)/As;                %total seed signal intensity

dv_ase = 53E-9*(v_s/l_s);

%DE's for light intensity
dI_p = @(z,I_p,n2) (-Gp*(s_ap*N*(1-n2) - s_ep*N*n2) - alpha_p).*I_p;
dI_sig = @(z,I_sig,n2) (-Gs*(s_as*N*(1-n2) - s_es*N*n2)).*I_sig;
dI_ase = @(z,I_ase, n2) (-Gs*(s_as*N*(1-n2) - s_es*N*n2)).*I_ase + 2*Gs*n2*h*v_s*N*s_es*dv_ase/As;
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
%backpumped fiber
I_p(:,end) = I_0p;
I_sig(:,1) = I_0s;
    
    
    %while-loop determinants
j = 0;
cGain = zeros(length(I_0p),1);
errG = 1;
err = [];
    
    
%Main loop
while j < 400 && abs(errG) > 1E-12

pGain = cGain;    

%RK4 method
%single-pass propagation    
for i = 1:length(z)
k = length(z) - i + 1;

    I_s(:,i) = I_sig(:,i) + I_asef(:,i) + I_aseb(:,i);
    n_2(:,i) = (a_p*I_p(:,i) + a_s*I_s(:,i))./(b_p*I_p(:,i) + b_s*I_s(:,i) + 1/tau_se);

    if i<length(z)
        
        %update I_p
        k1 = dI_p(z(k), I_p(:,k), n_2(:,k));
        k2 = dI_p(z(k) + dz/2, I_p(:,k) + k1*dz/2, n_2(:,k));
        k3 = dI_p(z(k) + dz/2, I_p(:,k) + k2*dz/2, n_2(:,k));
        k4 = dI_p(z(k) + dz, I_p(:,k) + k3*dz, n_2(:,k));
        I_p(:,k-1) = I_p(:,k) + (k1+2*k2+2*k3+k4)*dz/6;
        
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
%-----plotting
numPlots = 6;

m = floor(sqrt(numPlots));      %rows of plots
n = ceil(numPlots/m);           %columns of plots
p = 1;                          %current plotting position

scrsz = get(0,'ScreenSize');
f = figure('Position',[1 scrsz(4) 2*scrsz(3)/3 2*scrsz(4)/3], 'Visible', 'off');

figure(f)
colors = hsv(length(I_0p));
set(gcf, 'Colormap', colors);


%Pump plot
subplot(m,n,p)
p=p+1;
set(gca, 'ColorOrder', colors);
hold on
plot(z, I_p*Ap)
title('Pump intensity')
xlabel('Position along fiber (m)')
ylabel('Pump Power (W)')
%annotation('textbox', [0.6,0.6,0.1,0.1],...
%           'String', ['I_0 = ' num2str(I_p(1)/(1E7)) ' kW/cm^2']);
hold off


%Signal plot
subplot(m,n,p)
p=p+1;
set(gca, 'ColorOrder', colors);
hold on
plot(z, I_sig*As)
title('Signal intensity')
xlabel('Position along fiber (m)')
ylabel('Signal Power (W)')
%annotation('textbox', [0.6,0.6,0.1,0.1],...
%           'String', ['I_0 = ' num2str(I_p(1)/(1E7)) ' kW/cm^2']);
hold off


%ASE plot
subplot(m,n,p)
p=p+1;
set(gca, 'ColorOrder', colors);
hold on
plot(z, As*I_asef, '-')
plot(z, As*I_aseb, '-.')
title('ASE Signal intensity')
xlabel('Position along fiber (m)')
ylabel('ASE Signal Power (W)')
%annotation('textbox', [0.6,0.6,0.1,0.1],...
%           'String', ['I_0 = ' num2str(I_p(1)/(1E7)) ' kW/cm^2']);
hold off

%Inversion plot
subplot(m,n,p)
p=p+1;
set(gca, 'ColorOrder', colors);
hold on
plot(z, n_2)
title('Inversion')
xlabel('Position along fiber (m)')
ylabel('Upper state population')
%annotation('textbox', [0.6,0.6,0.1,0.1],...
%           'String', ['I_0 = ' num2str(I_p(1)/(1E7)) ' kW/cm^2']);
hold off

%Gain plot
subplot(m,n,p)
p=p+1;
set(gca, 'ColorOrder', colors);
hold on
plot(z, g)
title('Gain')
xlabel('Position along fiber (m)')
ylabel('Gain Coef')
%annotation('textbox', [0.6,0.6,0.1,0.1],...
%           'String', ['I_0 = ' num2str(I_p(1)/(1E7)) ' kW/cm^2']);
hold off

%Output plot
subplot(m,n,p)
p=p+1;
set(gca, 'ColorOrder', colors);
hold on
plot(I_0p*Ap, I_sig(:,end)*As)
title('Pump Efficiency')
xlabel('Pump power (W)')
ylabel('Output power (W)')
%annotation('textbox', [0.6,0.6,0.1,0.1],...
%           'String', ['I_0 = ' num2str(I_p(1)/(1E7)) ' kW/cm^2']);
hold off
%
% %%
% %Analysis
% 
% d = 6E-6;
% %P_t = (I_p(:,end) + I_sig(:,end) + I_asef(:,end) + I_aseb(:,end))*pi*(d/2)^2;
% P_0p = I_0p*pi*(d/2)^2;
% 
% effPlot = figure('Name', 'Total Output Power');
% plot(P_0p,P_t)




