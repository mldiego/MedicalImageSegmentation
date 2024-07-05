clc;clear;close all;
poolobj = gcp('nocreate');
delete(poolobj);

verify_linf_par;

poolobj = gcp('nocreate');
delete(poolobj);

clc;clear;close all;

verify_biasField_par;

poolobj = gcp('nocreate');
delete(poolobj);

verify_adjustContrast_par;

clc;clear;close all;
