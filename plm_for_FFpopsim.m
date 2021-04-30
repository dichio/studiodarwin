function J = plm_for_FFpopsim(data)
    %addpath('/Users/vito.dichio/Desktop/studiodarwin2/');
    %addpath('/Users/vito.dichio/Desktop/studiodarwin2/PLM-DCA/function');
    %addpath('/Users/vito.dichio/Desktop/studiodarwin2/PLM-DCA/thirdparty');
    %% PLM
    [B, N] = size(data); %input: [0,1]
    MSA = uint8(data); 
    q = 2;  % nucleic acid
    %S = uint8(MSA - 1);   % [1,q] -> [0,q-1]
    S = MSA';
    weights = ones(B,1);

    lambda = 0;
    
    Jr = PLM_DCA_Ising(S,N,B,q,weights, lambda, 4);

    
    J_plm = zeros(N,N);
    for i =1: N 
        x = Jr(:,i);
        if i==1
            J_plm(:, i) = [0; x];
        elseif (i == N)
            J_plm(:, i) = [x; 0];
        else
            J_plm(:, i) = [x(1:i-1); 0; x(i:end)];
        end
    end
    
    %% symmetrilize:
    J = 0.5*(J_plm + J_plm');
    J = J-diag(diag(J));
    
    J = single(J);
    


