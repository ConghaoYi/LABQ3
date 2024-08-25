import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import scipy.stats
import pickle
import pytest
import xraylib
from PIL import Image

import LABQ3
script_dir = os.path.dirname(os.path.realpath(__file__))
x_domain = np.linspace(0, 0.002, 2000)
####################################################################################################################################
# DO NOT CHANGE THIS SCRIPT. THIS SCRIPT IS MEANT TO TEST WHETHER LABQ3 CAN RUN AS EXPECTED.                                       #
# THE RELEVANT VARIABLES AND VALUES WERE HARD-CODED INTO THE SCRIPT TO TEST IF LABQ3 IS WORKING PROPERLY. ALL TESTS SHOULD PASS.   #
####################################################################################################################################
USE_CACHED_BAYES = True

def get_config_binary(std1="CaCO3", std2="CdCO3", energy="8kev", step=1):
    unique_run_key = f"{std1}__{std2}__{energy}__{step}"
    config = dict(# Below related to the classifier
                  std1_name=std1,
                  std2_name=std2,
                  dir_std1=f"{script_dir}/standards/{std1}_{energy}/txt",
                  dir_std2=f"{script_dir}/standards/{std2}_{energy}/txt",
                  dir_out=f"{script_dir}/test/",
                  step=step,
                  unique_run_key=unique_run_key,
                  use_pre_existing=USE_CACHED_BAYES,
                  # Below relates to the image
                  scan_name='bi_test',
                  dir_scan=f"{script_dir}/test/data/binary",
                  down_sample_scale=False,
                  erosion=False,
                  median_filter=False,
                  scale_erosion=False,
                  scale_median=False,
                  )
    return config
        
def get_config_ternary(std1="CaCO3", std2="CdCO3", std3="ZnCO3", step=1):
    unique_run_key = f"{std1}__{std2}__{std3}__{step}"
    config = dict(# Below relates to the bayes classifier
                  std1_name=std1,
                  std2_name=std2,
                  std3_name=std3,
                  dir_std1_8kev=f"{script_dir}/standards/{std1}_8kev/txt",
                  dir_std2_8kev=f"{script_dir}/standards/{std2}_8kev/txt",
                  dir_std3_8kev=f"{script_dir}/standards/{std3}_8kev/txt",
                  dir_std1_10kev=f"{script_dir}/standards/{std1}_10kev/txt",
                  dir_std2_10kev=f"{script_dir}/standards/{std2}_10kev/txt",
                  dir_std3_10kev=f"{script_dir}/standards/{std3}_10kev/txt",
                  dir_out=f"{script_dir}/test/",
                  step=step,
                  use_pre_existing=USE_CACHED_BAYES,
                  unique_run_key=unique_run_key,
                  # Below relates to the image
                  scan_name='ter_test',                  
                  dir_scan8=f"{script_dir}/test/data/ternary/8kev",
                  dir_scan10=f"{script_dir}/test/data/ternary/10kev",
                  down_sample_scale=False,
                  erosion=False,
                  median_filter=False,
                  scale_erosion=False,
                  scale_median=False,
                  )
    return config


LABQ3.main_bi(**get_config_binary("CaCO3", "CdCO3","8kev"))
LABQ3.main_ter(**get_config_ternary())


## The constant parameters are from the reconstruction algorithm
ca_8kev = xraylib.Refractive_Index_Im('CaCO3',8,2.71) * 1522
cd_8kev = xraylib.Refractive_Index_Im('CdCO3',8,4.26)*1500
zn_8kev = xraylib.Refractive_Index_Im('ZnCO3',8,4.434) * 1522
ca_10kev = xraylib.Refractive_Index_Im('CaCO3',9.675,2.71) * 1522
cd_10kev = xraylib.Refractive_Index_Im('CdCO3',9.68,4.26)*1500
zn_10kev = xraylib.Refractive_Index_Im('ZnCO3',9.7,4.434) * 1522

## Molar volumes of the standard minerals
caco3_vm = 36.93357934
cdco3_vm = 34.27833002
znco3_vm = 28.30474041

def avg_content_bi(r_core1, m1, m2):
    m1 = np.nan_to_num(m1)
    m2 = np.nan_to_num(m2)
    m1_core = 0
    m2_core = 0
    m1_rim = 0
    m2_rim = 0
    m1_core_list = []
    m2_core_list = []
    m1_rim_list = []
    m2_rim_list = []
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            #if is pore, discard
            if m1[i,j] ==0 and m2[i,j] == 0:
                continue
            if (i-512)**2 + (j-512)**2 < r_core1**2:
                m1_core = m1_core+m1[i,j]
                m2_core = m2_core+m2[i,j]
                m1_core_list.append(m1[i,j])
                m2_core_list.append(m2[i,j])
            if (i-512)**2 + (j-512)**2 >= r_core1**2 :
                m1_rim = m1_rim+m1[i,j]
                m2_rim = m2_rim+m2[i,j]
                m1_rim_list.append(m1[i,j])
                m2_rim_list.append(m2[i,j])
    m2_core_ratio = m2_core/(m1_core+m2_core)
    m2_rim_ratio = m2_rim/(m2_rim+m1_rim)
    return(m2_core_ratio,m2_rim_ratio)

def avg_content_ter(r_core1, m1, m2,m3):
    m1 = np.nan_to_num(m1)
    m2 = np.nan_to_num(m2)
    m3 = np.nan_to_num(m3)
    m1_core = 0
    m2_core = 0
    m1_rim = 0
    m2_rim = 0
    m3_core = 0
    m3_rim = 0
    m1_core_list = []
    m2_core_list = []
    m1_rim_list = []
    m2_rim_list = []
    m3_core_list = []
    m3_rim_list = []
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            #if is pore, discard
            if m1[i,j] ==0 and m2[i,j] == 0 and m3[i,j] == 0:
                continue
            if (i-700)**2 + (j-670)**2 < r_core1**2:
                m1_core = m1_core+m1[i,j]
                m2_core = m2_core+m2[i,j]
                m3_core = m3_core+m3[i,j]
                m1_core_list.append(m1[i,j])
                m2_core_list.append(m2[i,j])
                m3_core_list.append(m3[i,j])
            if (i-700)**2 + (j-670)**2 >= r_core1**2 :
                # print('rim')
                m1_rim = m1_rim+m1[i,j]
                m2_rim = m2_rim+m2[i,j]
                m3_rim = m3_rim+m3[i,j]
                m1_rim_list.append(m1[i,j])
                m2_rim_list.append(m2[i,j])
                m3_rim_list.append(m3[i,j])
    return(np.array(m1_core_list),np.array(m2_core_list),np.array(m1_rim_list),np.array(m2_rim_list),np.array(m3_core_list),np.array(m3_rim_list))

def error_binary(ca_theory, cd_theory):
    """
    Calculate the l_infty norm between the theoritical value and the predicted value
    """
    combo = np.load(f'{script_dir}/test/CaCO3__CdCO3__8kev__1/compositions/binary_step1.npy')/100
    table_bi = np.load(f'{script_dir}/test/CaCO3__CdCO3__8kev__1/lookup_table/binary_with_101_classes.npy')
    l_infty = []
    for i in range(combo.shape[0]):
        ca_true_composition = combo[i,0]
        cd_true_composition = combo[i,1]
        mu_theory = ca_true_composition*ca_theory + cd_true_composition*cd_theory

        
        mu_theory_index = np.digitize(mu_theory,x_domain) - 1
        index = table_bi[
                :, mu_theory_index
            ].argmax()
        ca_hat = combo[index, 0]
        cd_hat = combo[index, 1]
        l = max(np.abs(ca_true_composition - ca_hat), np.abs(cd_true_composition - cd_hat))
        l_infty.append(l)
        print(f'{i+1} out of {combo.shape[0]} done. True composition is Ca = {ca_true_composition} Cd = {cd_true_composition}. Predicted compositions is Ca = {ca_hat} Cd = {cd_hat}')
    l_inf = np.array(l_infty)
    return l_inf

def error_ternary(ca_theory_8,cd_theory_8,zn_theory_8,
                  ca_theory_10, cd_theory_10,zn_theory_10):
    combo = np.load(f'{script_dir}/test/CaCO3__CdCO3__ZnCO3__1/compositions/ternary_step1.npy')/100
    with open(f'{script_dir}/test/CaCO3__CdCO3__ZnCO3__1/lookup_table/joint_table_step1.cpkl','rb') as file:
        table_joint = pickle.load(file)
    file.close()
    l_infty = []
    for i in range(combo.shape[0]):
        ca_true_composition = combo[i,0]
        cd_true_composition = combo[i,1]
        zn_true_composition = combo[i,2]
        mu_theory_8 = ca_true_composition*ca_theory_8 + cd_true_composition*cd_theory_8 + zn_true_composition * zn_theory_8
        mu_theory_10 = ca_true_composition * ca_theory_10 + cd_true_composition * cd_theory_10 + zn_true_composition *zn_theory_10

        mu_theory_index_8 = np.digitize(mu_theory_8,x_domain)
        mu_theory_index_10 = np.digitize(mu_theory_10,x_domain)
        
        ca_hat = combo[table_joint[(mu_theory_index_8,mu_theory_index_10)], 0]
        cd_hat = combo[table_joint[(mu_theory_index_8,mu_theory_index_10)], 1]
        zn_hat = combo[table_joint[(mu_theory_index_8,mu_theory_index_10)], 2]

        print(f'{i+1} out of {combo.shape[0]} done. True composition is Ca = {ca_true_composition} Cd = {cd_true_composition} Zn = {zn_true_composition}. Predicted compositions is Ca = {ca_hat} Cd = {cd_hat} Zn = {zn_hat}')
        l = max(np.abs(ca_true_composition - ca_hat), np.abs(cd_true_composition - cd_hat),np.abs(zn_true_composition - zn_hat))
        l_infty.append(l)
    l_inf = np.array(l_infty)
    return l_inf


def test_error_bi():
    error_bi = error_binary(ca_8kev,cd_8kev)
    assert np.max(error_bi) == pytest.approx(0.050000,abs = 1e-6)
    assert np.average(error_bi) == pytest.approx(0.022475,abs = 1e-6)


def test_error_ter():
    error_ter = error_ternary(ca_8kev,cd_8kev,zn_8kev,
                          ca_10kev,cd_10kev,zn_10kev)
    assert np.max(error_ter) == pytest.approx(0.189999,abs = 1e-6)
    assert np.average(error_ter) == pytest.approx(0.079390,abs = 1e-6)

core_bi = avg_content_bi(110,np.load(f'{script_dir}/test/CaCO3__CdCO3__8kev__1/bi_test/std1/0001.npy'),
                         np.load(f'{script_dir}/test/CaCO3__CdCO3__8kev__1/bi_test/std2/0001.npy'))

core_ter = avg_content_ter(320,np.load(f'{script_dir}/test/CaCO3__CdCO3__ZnCO3__1/ter_test/std1/0001.npy'),
                           np.load(f'{script_dir}/test/CaCO3__CdCO3__ZnCO3__1/ter_test/std2/0001.npy'),
                           np.load(f'{script_dir}/test/CaCO3__CdCO3__ZnCO3__1/ter_test/std3/0001.npy'))

def test_bi_core_content():
    """
    Test for results reported in section 4.3
    """
    cd_core = (round(core_bi[0],2)/cdco3_vm)/((round(core_bi[0],2)/cdco3_vm)+((1-round(core_bi[0],2))/caco3_vm))
    cd_rim = (round(core_bi[1],2)/cdco3_vm)/((round(core_bi[1],2)/cdco3_vm)+((1-round(core_bi[1],2))/caco3_vm))
    assert cd_core == pytest.approx(0.346699,abs = 1e-6)
    assert cd_rim == pytest.approx(0.222645,abs = 1e-6)

def test_ter_core_content():
    """
    Test for results reported in section 4.4
    """
    ca_core_v = np.average(core_ter[0])
    cd_core_v = np.average(core_ter[1])
    zn_core_v = np.average(core_ter[4])
    ca_rim_v = np.average(core_ter[2])
    cd_rim_v = np.average(core_ter[3])
    zn_rim_v = np.average(core_ter[5])

    ca_core_x = (round(ca_core_v,2)/caco3_vm)/((round(cd_core_v,2)/cdco3_vm)+((round(ca_core_v,2))/caco3_vm)+((round(zn_core_v,2))/znco3_vm))
    cd_core_x = (round(cd_core_v,2)/cdco3_vm)/((round(cd_core_v,2)/cdco3_vm)+((round(ca_core_v,2))/caco3_vm)+((round(zn_core_v,2))/znco3_vm))
    zn_core_x = (round(zn_core_v,2)/znco3_vm)/((round(cd_core_v,2)/cdco3_vm)+((round(ca_core_v,2))/caco3_vm)+((round(zn_core_v,2))/znco3_vm))
    ca_rim_x = (round(ca_rim_v,2)/caco3_vm)/((round(cd_rim_v,2)/cdco3_vm)+((round(ca_rim_v,2))/caco3_vm)+((round(zn_rim_v,2))/znco3_vm))
    cd_rim_x = (round(cd_rim_v,2)/cdco3_vm)/((round(cd_rim_v,2)/cdco3_vm)+((round(ca_rim_v,2))/caco3_vm)+((round(zn_rim_v,2))/znco3_vm))
    zn_rim_x = (round(zn_rim_v,2)/znco3_vm)/((round(cd_rim_v,2)/cdco3_vm)+((round(ca_rim_v,2))/caco3_vm)+((round(zn_rim_v,2))/znco3_vm))
    assert ca_core_x == pytest.approx(0.634638,abs = 1e-6)
    assert cd_core_x == pytest.approx(0.142883,abs = 1e-6)
    assert zn_core_x == pytest.approx(0.222477,abs = 1e-6)
    assert ca_rim_x == pytest.approx(0.146577,abs = 1e-6)
    assert cd_rim_x == pytest.approx(0.380893,abs = 1e-6)
    assert zn_rim_x == pytest.approx(0.472529,abs = 1e-6)

def test_table_1():
    """
    Test for results reported in Table 1.
    """
    #Helper function
    def get_mean_variance_skew_kurtosis(pmf, values):
        values = values * 10**7 #turn unit into nm**-1
        mean = np.sum(values*pmf) 
        variance  = np.sum(((values - mean) ** 2) * pmf)
        skew = scipy.stats.skew(pmf)
        kurto = scipy.stats.kurtosis(pmf)
        mod = values[np.argmax(pmf)]

        return(mean, mod, variance, skew, kurto)
    #CaCO3 at 8keV
    angles = 1522
    ##Mean
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_8kev/txt',1)[0],x_domain/angles)[0] == pytest.approx(2.519429,abs=1e-6)
    ##Mode
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_8kev/txt',1)[0],x_domain/angles)[1] == pytest.approx(2.465095,abs=1e-6)
    ##Variance
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_8kev/txt',1)[0],x_domain/angles)[2] == pytest.approx(0.858160,abs=1e-6)
    ##Skewness
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_8kev/txt',1)[0],x_domain/angles)[3] == pytest.approx(1.606280,abs=1e-6)
    ##Kurtosis
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_8kev/txt',1)[0],x_domain/angles)[4] == pytest.approx(1.101241,abs=1e-6)
    
    #CdCO3 at 8keV
    angles = 1500
    ##Mean
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_8kev/txt',1)[0],x_domain/angles)[0] == pytest.approx(7.459327,abs=1e-6)
    ##Mode
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_8kev/txt',1)[0],x_domain/angles)[1] == pytest.approx(8.377522,abs=1e-6)
    ##Variance
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_8kev/txt',1)[0],x_domain/angles)[2] == pytest.approx(2.928827,abs=1e-6)
    ##Skewness
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_8kev/txt',1)[0],x_domain/angles)[3] == pytest.approx(0.966990,abs=1e-6)
    ##Kurtosis
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_8kev/txt',1)[0],x_domain/angles)[4] == pytest.approx(-0.420821,abs=1e-6)

    #ZnCO3 at 8keV
    angles = 1522
    ##Mean
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_8kev/txt',1)[0],x_domain/angles)[0] == pytest.approx(2.004829,abs=1e-6)
    ##Mode
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_8kev/txt',1)[0],x_domain/angles)[1] == pytest.approx(2.096974,abs=1e-6)
    ##Variance
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_8kev/txt',1)[0],x_domain/angles)[2] == pytest.approx(0.949605,abs=1e-6)
    ##Skewness
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_8kev/txt',1)[0],x_domain/angles)[3] == pytest.approx(1.429952,abs=1e-6)
    ##Kurtosis
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_8kev/txt',1)[0],x_domain/angles)[4] == pytest.approx(0.498543,abs=1e-6)

    #CaCO3 at 10keV
    angles = 1522
    ##Mean
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_10kev/txt',1)[0],x_domain/angles)[0] == pytest.approx(2.031432,abs=1e-6)
    ##Mode
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_10kev/txt',1)[0],x_domain/angles)[1] == pytest.approx(1.222687,abs=1e-6)
    ##Variance
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_10kev/txt',1)[0],x_domain/angles)[2] == pytest.approx(1.623296,abs=1e-6)
    ##Skewness
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_10kev/txt',1)[0],x_domain/angles)[3] == pytest.approx(1.083185,abs=1e-6)
    ##Kurtosis
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CaCO3_10kev/txt',1)[0],x_domain/angles)[4] == pytest.approx(-0.531809,abs=1e-6)

    #CdCO3 at 10keV
    angles = 1500
    ##Mean
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_10kev/txt',1)[0],x_domain/angles)[0] == pytest.approx(4.267153,abs=1e-6)
    ##Mode
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_10kev/txt',1)[0],x_domain/angles)[1] == pytest.approx(3.795230,abs=1e-6)
    ##Variance
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_10kev/txt',1)[0],x_domain/angles)[2] == pytest.approx(3.063721,abs=1e-6)
    ##Skewness
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_10kev/txt',1)[0],x_domain/angles)[3] == pytest.approx(0.831445,abs=1e-6)
    ##Kurtosis
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/CdCO3_10kev/txt',1)[0],x_domain/angles)[4] == pytest.approx(-0.669936,abs=1e-6)

    #ZnCO3 at 10keV
    angles = 1522
    ##Mean
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_10kev/txt',1)[0],x_domain/angles)[0] == pytest.approx(5.633864,abs=1e-6)
    ##Mode
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_10kev/txt',1)[0],x_domain/angles)[1] == pytest.approx(5.995113,abs=1e-6)
    ##Variance
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_10kev/txt',1)[0],x_domain/angles)[2] == pytest.approx(3.465534,abs=1e-6)
    ##Skewness
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_10kev/txt',1)[0],x_domain/angles)[3] == pytest.approx(0.603640,abs=1e-6)
    ##Kurtosis
    assert get_mean_variance_skew_kurtosis(LABQ3.gen_avg_std_his(f'{script_dir}/standards/ZnCO3_10kev/txt',1)[0],x_domain/angles)[4] == pytest.approx(-1.170647,abs=1e-6)

