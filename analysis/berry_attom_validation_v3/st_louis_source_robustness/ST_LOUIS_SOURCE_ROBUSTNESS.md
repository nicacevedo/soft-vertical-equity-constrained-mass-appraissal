# St. Louis County local vs ATTOM source robustness

Same transactions, same sale target (local PRICE), same chronology, same LGBM config search.
Local predictors: dwelling snapshot strictly before SALEDT.
ATTOM predictors: HISTORY_STRUCTURAL_CORE-style Assessor History (no prior-sale market history, no tax values).

Common N=476; validation N=38; test unscored until freeze.
Join: normalized PARCELNUMBERRAW/PARID + sale date, 2016-2019 (formatted APN missing).
Price corroboration vs ATTOM TRANSFERAMOUNT: exact=0.9852941176470589; <=1%=0.9873949579831933; <=5%=0.9915966386554622.

                   source selected_lgbm_config  n_common  n_validation  n_test_unscored  test_block_scored  share_exact_price  share_price_le_1pct  share_price_le_5pct  R2_price       R2   R2_log  R2 (log)  OOS_R2_price   OOS R2  OOS_R2_log    RMSE_price          RMSE  RMSE_log     MAE_price           MAE     MAPE    MdAPE  Corr(r,price)  Corr(r,logprice)  Slope(r~logy)  Std ratio  Median ratio  Mean ratio  W. Mean ratio       COD  COV_IAAO       VEI      PRD       PRB      MKI  Cov_log_residual_log_price  Beta_log  Corr_log_residual_log_price  dCor_e_y fairness_ratio_mode  Corr(r,y)_price  FisherZ(r,y)_price  Slope(r~y)_price   Std(r)  val_rows  small_abs_y_share   y_q05    y_q50    y_q95    r_q05    r_q50    r_q95  N  RMSE (log)  MAE (log)  Cov(e,logprice)  Delta_NL
local_historical_dwelling         test_best_r2       476            38               96              False           0.985294             0.987395             0.991597  0.086241 0.086241 0.480005  0.480005      0.110012 0.110012    0.486908 443229.097189 443229.097189  0.540999 140001.126303 140001.126303 0.270198 0.224851      -0.600781         -0.693254      -0.333157   0.360541      0.960882    1.007911       0.732019 27.901898  0.362512 -9.843516 1.376893 -0.222537 0.540475                   -0.294063 -0.522451                    -0.733996  0.641476                 div        -0.600781           -0.694368     -4.671513e-07 0.360541        38                0.0 85000.0 219000.0 887000.0 0.591927 0.960882 1.599640 38    0.540999   0.311666        -0.294063       0.0
   attom_assessor_history         test_best_r2       476            38               96              False           0.985294             0.987395             0.991597  0.273363 0.273363 0.674770  0.674770      0.292265 0.292265    0.679087 395249.470297 395249.470297  0.427851 138596.675226 138596.675226 0.281621 0.197414      -0.496570         -0.575003      -0.316585   0.413065      0.949919    1.022239       0.760982 29.039943  0.409503 -3.753424 1.343316 -0.174255 0.571841                   -0.226873 -0.403077                    -0.713722  0.630048                 div        -0.496570           -0.544743     -4.423705e-07 0.413065        38                0.0 85000.0 219000.0 887000.0 0.554735 0.949919 1.925046 38    0.427851   0.286560        -0.226873       0.0

Berry official assessment ratios are not used here. This is a predictor-source comparison on a common sale target.

