% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 1447.095155607248671 ; 1450.220854410630409 ];

%-- Principal point:
cc = [ 896.815318709177518 ; 470.036065018980082 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.053788863391483 ; -0.086229932179440 ; -0.010452273568919 ; -0.004608035571923 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 17.147922010867294 ; 17.335146925417522 ];

%-- Principal point uncertainty:
cc_error = [ 16.903133465471722 ; 13.058886739038355 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.017460838068763 ; 0.033467524014154 ; 0.002786121272724 ; 0.002956166257654 ; 0.000000000000000 ];

%-- Image size:
nx = 1920;
ny = 1080;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 10;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ -2.038095e+00 ; -2.198949e+00 ; 3.191216e-01 ];
Tc_1  = [ -4.290099e+00 ; -9.157219e+01 ; 4.962710e+02 ];
omc_error_1 = [ 7.200930e-03 ; 7.310371e-03 ; 1.264018e-02 ];
Tc_error_1  = [ 5.728148e+00 ; 4.434082e+00 ; 6.115450e+00 ];

%-- Image #2:
omc_2 = [ 2.179940e+00 ; 2.232478e+00 ; -1.907765e-02 ];
Tc_2  = [ 1.569868e+00 ; -8.175599e+01 ; 4.007195e+02 ];
omc_error_2 = [ 6.606363e-03 ; 5.137640e-03 ; 1.125342e-02 ];
Tc_error_2  = [ 4.695240e+00 ; 3.609950e+00 ; 4.808821e+00 ];

%-- Image #3:
omc_3 = [ -2.133591e+00 ; -2.292710e+00 ; 1.078886e-01 ];
Tc_3  = [ 5.303960e+00 ; -5.102490e+01 ; 5.424621e+02 ];
omc_error_3 = [ 6.369028e-03 ; 8.481250e-03 ; 1.540804e-02 ];
Tc_error_3  = [ 6.326269e+00 ; 4.882894e+00 ; 6.494029e+00 ];

%-- Image #4:
omc_4 = [ 2.219839e+00 ; 2.177848e+00 ; -8.876759e-02 ];
Tc_4  = [ -2.873183e+01 ; -7.422930e+01 ; 4.816450e+02 ];
omc_error_4 = [ 7.053266e-03 ; 6.013546e-03 ; 1.265850e-02 ];
Tc_error_4  = [ 5.607513e+00 ; 4.326740e+00 ; 5.777677e+00 ];

%-- Image #5:
omc_5 = [ -2.187441e+00 ; -2.151523e+00 ; 1.920688e-01 ];
Tc_5  = [ 3.088295e+01 ; -3.582597e+01 ; 5.590961e+02 ];
omc_error_5 = [ 7.006091e-03 ; 8.713418e-03 ; 1.611306e-02 ];
Tc_error_5  = [ 6.526275e+00 ; 5.036293e+00 ; 6.726632e+00 ];

%-- Image #6:
omc_6 = [ -2.197356e+00 ; -2.119273e+00 ; 1.570914e-01 ];
Tc_6  = [ 2.602577e+01 ; -5.527655e+01 ; 6.051948e+02 ];
omc_error_6 = [ 8.072444e-03 ; 9.298776e-03 ; 1.760680e-02 ];
Tc_error_6  = [ 7.071006e+00 ; 5.452291e+00 ; 7.420589e+00 ];

%-- Image #7:
omc_7 = [ -1.857801e+00 ; -1.660070e+00 ; 8.987113e-01 ];
Tc_7  = [ 1.634808e+02 ; -4.591142e+01 ; 7.664402e+02 ];
omc_error_7 = [ 9.671438e-03 ; 7.581658e-03 ; 1.308919e-02 ];
Tc_error_7  = [ 9.011875e+00 ; 6.965114e+00 ; 8.680538e+00 ];

%-- Image #8:
omc_8 = [ -1.819359e+00 ; -1.633024e+00 ; 8.919566e-01 ];
Tc_8  = [ 1.056178e+02 ; -9.099813e+01 ; 6.378971e+02 ];
omc_error_8 = [ 9.547887e-03 ; 7.162366e-03 ; 1.230723e-02 ];
Tc_error_8  = [ 7.466010e+00 ; 5.751981e+00 ; 7.126979e+00 ];

%-- Image #9:
omc_9 = [ -1.792316e+00 ; -1.623287e+00 ; 9.145554e-01 ];
Tc_9  = [ 1.282747e+02 ; -8.102737e+01 ; 5.957594e+02 ];
omc_error_9 = [ 9.207793e-03 ; 7.095883e-03 ; 1.192351e-02 ];
Tc_error_9  = [ 6.987707e+00 ; 5.395706e+00 ; 6.661923e+00 ];

%-- Image #10:
omc_10 = [ -1.768979e+00 ; -1.642284e+00 ; 9.592538e-01 ];
Tc_10  = [ 1.390646e+02 ; -5.712947e+01 ; 5.869390e+02 ];
omc_error_10 = [ 9.140281e-03 ; 7.195204e-03 ; 1.165549e-02 ];
Tc_error_10  = [ 6.885384e+00 ; 5.330845e+00 ; 6.539093e+00 ];

