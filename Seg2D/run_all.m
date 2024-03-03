%% Run all experiments to verify MS segmentation

% 1) MSSEG16
cd MSSEG16;

verify_adjustContrast_cmd;

disp("MSSEG1 ")

verify_biasField_cmd; 

cd ..;

% 2) ISBI

cd ISBI;

verify_adjustContrast_cmd;

verify_biasField_cmd; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% If we have time

verify_linf_cmd;

cd ..;

cd MSSEG16;

verify_linf_cmd;

cd ..;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3) Finally (probably will not include)

cd UMCL;

verify_adjustContrast_cmd;

verify_biasField_cmd; 

verify_linf_cmd;

