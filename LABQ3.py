import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
import scipy
import pickle
import functools

# Domain for histograms used for densities
x_domain = np.linspace(0, 0.002, 2000)

def init_ternary(step, config_info):
    """    initialize all classes 
    5151 different combinations
    
    from left to right, the numbers represent ratio of Ca, Cd, Zn, respectively."""

    combo_dir = os.path.join(config_info['unique_run_dir'], 'compositions')
    if config_info['use_pre_existing'] and os.path.exists(os.path.join(combo_dir, f'ternary_step{step}.npy')):
        print(f'loading existing ternary  simplex data from {combo_dir}')
        combination = np.load(os.path.join(combo_dir, f'ternary_step{step}.npy'))
    else:
        print(f"Recreating ternary simplex data")
        combination = np.zeros((0, 3))
        for ca in range(0, 101, step):
            for cd in range(0, 101, step):
                zn = 100 - ca - cd
                if zn < 0: continue
                combination = np.concatenate((combination, np.array([[ca, cd, zn]])))
        os.makedirs(combo_dir, exist_ok=True)
        np.save(os.path.join(combo_dir, f'ternary_step{step}'), combination)
    return combination / 100  # convert percentages to decimal

def init_binary(step,config_info):
    combo_dir = os.path.join(config_info['unique_run_dir'], 'compositions')
    if config_info['use_pre_existing'] and os.path.exists(os.path.join(combo_dir, f'binary_step{step}.npy')):
        print(f'loading existing binary simplex data from from {combo_dir}')
        combination = np.load(os.path.join(combo_dir, f'binary_step{step}.npy'))
    else:
        print(f"Recreating binary simplex data")
        combination = np.zeros((0, 2))
        for a in range(0, 101, step):
            b = 100 - a
            if b < 0: continue
            combination = np.concatenate((combination, np.array([[a, b]])))
        os.makedirs(combo_dir, exist_ok=True)
        np.save(os.path.join(combo_dir, f'binary_step{step}'), combination)
    return combination / 100  # convert percentages to decimal

@functools.cache
def _loadtxt(dir):
    hhh = np.loadtxt(dir).ravel()
    hhh[hhh < 1e-5] = np.nan
    # NaN processing is very expensive
    hhh = hhh[~np.isnan(hhh)]
    return hhh
@functools.cache
def load_std_with_para(dir, para):
    hhh = _loadtxt(dir)
    hist_temp = para * hhh
    pre_hist = np.digitize(hist_temp, x_domain)
    return (
            np.histogram(
                    pre_hist, np.arange(2001), weights=np.ones(len(pre_hist)) / len(pre_hist)
            )[0],
            pre_hist,
    )
def syn_bi(dir1, dir2, para1, para2):
    his1 = gen_avg_std_his(dir1, para1)[0]
    his2 = gen_avg_std_his(dir2, para2)[0]
    return np.convolve(his1, his2)[: x_domain.size]

def syn_ter(dir1, dir2, dir3, para1, para2, para3):
    his1 = gen_avg_std_his(dir1, para1)[0]
    his2 = gen_avg_std_his(dir2, para2)[0]
    his3 = gen_avg_std_his(dir3, para3)[0]
    his_temp = np.convolve(his1, his2)
    return np.convolve(his_temp, his3)[: x_domain.size]

def gen_avg_std_his(his_dir, para):
    filelist = os.listdir(his_dir)
    std_all = np.zeros((len(filelist), 2000))
    iter = 0
    for i in filelist:
        filename = his_dir + "/" + i
        std_all[iter, :] = load_std_with_para(filename, para)[0]

        iter = iter + 1
    return (np.mean(std_all, axis=0), np.std(std_all, axis=0))

def find_comp_bayes_bi(sample, lookup_table, combo):
    sample = np.where(sample <= 0, np.nan, sample)
    sample_index = (
            np.digitize(sample, x_domain) - 1
    )  # np.digitize numbers bins starting from 1 NOT 0
    std1 = np.zeros((sample.shape[0], sample.shape[1]))
    std2 = np.zeros((sample.shape[0], sample.shape[1]))
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            if not np.isnan(sample[i, j]):
                index = lookup_table[
                        :, sample_index[i, j]
                        ].argmax()  # find the indices that corresponds to the max posteriori probability
                std1[i, j] = combo[index, 0]
                std2[i, j] = combo[index, 1]
            else:
                std1[i, j] = np.nan
                std2[i, j] = np.nan

    return (std1, std2)

def find_comp_bayes_ter(sample8, sample10, joint_table, combo):
    # only consider positive data
    mask = (sample8 > 0) & (sample10 > 0)

    # Initialize arrays with NaNs to avoid default zeros in unmasked regions
    std1 = np.full(sample8.shape, np.nan)
    std2 = np.full(sample8.shape, np.nan)
    std3 = np.full(sample8.shape, np.nan)
    sample_index_8 = np.digitize(sample8[mask], x_domain) - 1
    sample_index_10 = np.digitize(sample10[mask], x_domain) - 1

    # Indices of masked elements
    masked_indices = np.where(mask)

    for i, (row, col) in enumerate(zip(*masked_indices)):
        mu8_index = sample_index_8[i]
        mu10_index = sample_index_10[i]
        # the +1 here is because the key for joint_table is based on digitize index
        std1[row, col] = combo[joint_table[(mu8_index + 1, mu10_index + 1)], 0]
        std2[row, col] = combo[joint_table[(mu8_index + 1, mu10_index + 1)], 1]
        std3[row, col] = combo[joint_table[(mu8_index + 1, mu10_index + 1)], 2]

    return (std1, std2, std3)

def gen_table_bi(dir1, dir2, step, config_info):
    table_dir = os.path.join(config_info['unique_run_dir'], 'lookup_table')
    combo = init_binary(step,config_info)
    if config_info['use_pre_existing'] and os.path.exists(os.path.join(table_dir, f'binary_with_{combo.shape[0]}_classes.npy')):
        print(f'loading existing binary lookup table data from {table_dir}')
        table = np.load(os.path.join(table_dir, f'binary_with_{combo.shape[0]}_classes.npy'))
    else:
        row = 0
        table = np.zeros((0, x_domain.shape[0]))
        while row < combo.shape[0]:
            table = np.concatenate(
                    [table, syn_bi(dir1, dir2, combo[row, 0], combo[row, 1])[np.newaxis, :]]
            )
            print(f'calculated row {row} of total {combo.shape[0]} rows  combo={combo[row]}')
            row = row + 1
        os.makedirs(table_dir, exist_ok=True)
        name = os.path.join(table_dir, f'binary_with_{combo.shape[0]}_classes')
        np.save(name, table)
    return table

def gen_joint_table(dir1_8, dir2_8, dir3_8, dir1_10, dir2_10, dir3_10, step, config_info):
    table_dir = os.path.join(config_info['unique_run_dir'], 'lookup_table')
    if config_info['use_pre_existing'] and os.path.exists(os.path.join(table_dir, f'joint_table_step{step}.cpkl')):
        print(f'loading existing ternary lookup table data from {table_dir}')
        with open(os.path.join(table_dir, f'joint_table_step{step}.cpkl'), 'rb') as f1:
            joint_table = pickle.load(f1)
        f1.close()
    else:
        print('Generating joint lookup table, takes time.')
        table8 = gen_table_ter(dir1_8, dir2_8, dir3_8, step,"8kev", config_info)
        table10 = gen_table_ter(dir1_10, dir2_10, dir3_10, step,"10kev",config_info)

        mu8 = np.digitize(x_domain, x_domain)
        mu10 = np.digitize(x_domain, x_domain)

        joint_table = {}
        # TBD - could be vectorized
        for i in mu8:
            print(" Merging posteriors ... ", i)
            for j in mu10:
                p_mu8_w = table8[:, i - 1]
                p_mu10_w = table10[:, j - 1]

                p_mu_joint = p_mu8_w * p_mu10_w
                index = p_mu_joint.argmax()
                joint_table[(i, j)] = index

        with open(os.path.join(table_dir, f'joint_table_step{step}.cpkl'), 'wb') as file:
            pickle.dump(joint_table, file)
        file.close()
    return joint_table

def gen_table_ter(dir1, dir2, dir3, step, energy, config_info):
    table_dir = os.path.join(config_info['unique_run_dir'], 'lookup_table')
    combo = init_ternary(step,config_info)
    if config_info['use_pre_existing'] and os.path.exists(os.path.join(table_dir, f'ternary_with_{combo.shape[0]}_classes_{energy}.npy')):
        print(f'loading existing ternary lookup table data for energy {energy} from {table_dir}')
        table = np.load(os.path.join(table_dir, f'ternary_with_{combo.shape[0]}_classes_{energy}.npy'))
    else:
        row = 0
        table = np.zeros((0, x_domain.shape[0]))
        while row < combo.shape[0]:
            table = np.concatenate(
                    [
                            table,
                            syn_ter(dir1, dir2, dir3, combo[row, 0], combo[row, 1], combo[row, 2])[
                            np.newaxis, :
                            ],
                    ]
            )
            print(f'calculated row {row + 1} of total {combo.shape[0]} rows  combo={combo[row]} for {energy}')
            row = row + 1
        os.makedirs(table_dir, exist_ok=True)
        name = os.path.join(table_dir, f'ternary_with_{combo.shape[0]}_classes_{energy}')
        np.save(name, table)
    return table


def get_total_ratio_bi(dir_file1, dir_file2):
    sum1 = 0
    sum2 = 0
    for file1 in os.listdir(dir_file1):
        temp1 = np.load(os.path.join(dir_file1, str(file1)))
        sum1 = sum1 + np.nansum(temp1)
    print("total mineral standard 1 = \n", sum1)
    for file2 in os.listdir(dir_file2):
        temp2 = np.load(os.path.join(dir_file2, str(file2)))
        sum2 = sum2 + np.nansum(temp2)
    print("total mineral standard 2 = \n", sum2)
    ratio1 = sum1 / (sum1 + sum2)
    ratio2 = 1-ratio1
    return (ratio1,ratio2)

def get_total_ratio_ter(dir_file1, dir_file2, dir_file3):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for file1 in os.listdir(dir_file1):
        temp1 = np.load(os.path.join(dir_file1, str(file1)))
        sum1 = sum1 + np.nansum(temp1)
    print("total mineral standard 1 = \n", sum1)
    for file2 in os.listdir(dir_file2):
        temp2 = np.load(os.path.join(dir_file2, str(file2)))
        sum2 = sum2 + np.nansum(temp2)
    print("total mineral standard 2 = \n", sum2)
    for file3 in os.listdir(dir_file3):
        temp3 = np.load(os.path.join(dir_file3, str(file3)))
        sum3 = sum3 + np.nansum(temp3)
    print("total mineral standard 3 = ", sum3)
    ratio1 = sum1 / (sum1 + sum2 + sum3)
    ratio2 = sum2 / (sum1 + sum2 + sum3)
    ratio3 = 1 - (ratio1 + ratio2)
    return (ratio1, ratio2, ratio3)

def main_bi(
        std1_name,
        std2_name,
        dir_scan,
        dir_out,
        dir_std1,
        dir_std2,
        step=1,
        down_sample_scale=False,
        erosion=False,
        median_filter=False,
        scale_erosion=False,
        scale_median=False,
        use_pre_existing=False,
        unique_run_key="",
        scan_name='my_particle'):

    config_info = dict(run_dir=dir_out,
                       unique_run_dir=os.path.join(dir_out,unique_run_key),
                       img_out_dir=os.path.join(dir_out,unique_run_key,scan_name),
                       use_pre_existing=use_pre_existing)

    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(config_info["unique_run_dir"], exist_ok=True)
    os.makedirs(config_info["img_out_dir"], exist_ok=True)

    combination = init_binary(step,config_info)
    print("\n Generating lookup table \n")
    table = gen_table_bi(dir_std1, dir_std2, step, config_info)
    print("\n Finished generating lookup table\n")

    slice_num = 0
    filelist = sorted(os.listdir(dir_scan))
    for file in filelist:
        print(file)
        slice_num = slice_num + 1
        if '.npy' in file:
            sample1 = np.load(dir_scan + "/" +str(file))
        if '.txt' in file:
            sample1 = np.loadtxt(dir_scan + "/" + str(file))
        if down_sample_scale:
            print("down sampling by", down_sample_scale)
            sample = skimage.measure.block_reduce(
                    sample1, (down_sample_scale, down_sample_scale), np.mean
            )
        else:
            print("no down sampling")
            sample = sample1
        if erosion:
            print("applying erosion filter with size ", str(scale_erosion))
            sample = sample * scipy.ndimage.morphology.binary_erosion(
                    sample, structure=np.ones(scale_erosion)
            )
        if median_filter:
            print("applying median filter with size of ", str(scale_median))
            sample = scipy.signal.medfilt2d(sample, scale_median)

        comp = find_comp_bayes_bi(sample, table, combination)
        os.makedirs(config_info["img_out_dir"]+"/std1", exist_ok=True)
        os.makedirs(config_info["img_out_dir"]+"/std2", exist_ok=True)                
        filename1 = os.path.join(config_info['img_out_dir'],'std1/' + str(slice_num).zfill(4))
        filename2 = os.path.join(config_info['img_out_dir'],'std2/' + str(slice_num).zfill(4))
        np.save(filename1, comp[0])
        np.save(filename2, comp[1])

    print("Computing final ratios")
    a = get_total_ratio_bi(
            os.path.join(config_info['img_out_dir'], 'std1'), os.path.join(config_info['img_out_dir'], 'std2')
    )
    ratio = open(config_info['img_out_dir']+ "/ratio.txt", mode="w")
    para = (
            "Mineral standard 1 = "
            + str(std1_name)
            + ", "
            + "Mineral standard 2 ="
            + str(std2_name)
            + ", "
            + ", resolution of classes = "
            + str(step)
            + ", down sampling factor = "
            + str(down_sample_scale)
            + "\n erosion filter "
            + str(erosion)
            + " by factor "
            + str(scale_erosion)
            + "\n median filter "
            + str(median_filter)
            + " by factor "
            + str(scale_median)
    )
    print(f"ratio is [{std1_name},{std2_name}] =", a)
    ratio.writelines(str(a))
    ratio.writelines(para)

def main_ter(
        std1_name,
        std2_name,
        std3_name,
        dir_scan8,
        dir_scan10,
        dir_out,
        dir_std1_8kev,
        dir_std2_8kev,
        dir_std3_8kev,
        dir_std1_10kev,
        dir_std2_10kev,
        dir_std3_10kev,
        step=1,
        down_sample_scale=False,
        erosion=False,
        median_filter=False,
        scale_erosion=False,
        scale_median=False,
        use_pre_existing=True,
        unique_run_key="",
        scan_name='my_particle'):

    config_info = dict(run_dir=dir_out,
                       unique_run_dir=os.path.join(dir_out,unique_run_key),
                       img_out_dir=os.path.join(dir_out,unique_run_key,scan_name),
                       use_pre_existing=use_pre_existing)

    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(config_info["unique_run_dir"], exist_ok=True)
    os.makedirs(config_info["img_out_dir"], exist_ok=True)

    combination = init_ternary(step,config_info)
    print("\n Generating lookup table \n")
    table_joint = gen_joint_table(dir_std1_8kev, dir_std2_8kev, dir_std3_8kev,
                                  dir_std1_10kev, dir_std2_10kev, dir_std3_10kev,
                                  step, config_info)
    print("\n Finished generating lookup table\n")

    slice_num = 0
    filelist_8kev = sorted(os.listdir(dir_scan8))
    filelist_10kev = sorted(os.listdir(dir_scan10))
    # the file names in the 8kev and 10kev folders need to be the same for the code to work
    if filelist_10kev != filelist_8kev:
        print('Quantification stopped because the file names in both scan folders are not the same.\nThey need to be the same for the quantification to proceed.')
        return
    for file in filelist_8kev:
        slice_num = slice_num + 1
        if '.npy' in file:
            sample1_8kev = np.load(os.path.join(dir_scan8, str(file)))
            sample1_10kev = np.load(os.path.join(dir_scan10, str(file)))
        if '.txt' in file:
            sample1_8kev = np.loadtxt(os.path.join(dir_scan8, str(file)))
            sample1_10kev = np.loadtxt(os.path.join(dir_scan10, str(file)))
        print(f'Quantifying {os.path.join(dir_scan8, file)} and {os.path.join(dir_scan10, file)}')
        if down_sample_scale:
            print("down sampling by", down_sample_scale)
            sample_8kev = skimage.measure.block_reduce(
                    sample1_8kev, (down_sample_scale, down_sample_scale), np.mean
            )
            sample_10kev = skimage.measure.block_reduce(
                    sample1_10kev, (down_sample_scale, down_sample_scale), np.mean
            )
        else:
            print("no down sampling")
            sample_8kev = sample1_8kev
            sample_10kev = sample1_10kev
        if erosion:
            print("applying erosion filter with size ", str(scale_erosion))
            sample_8kev = sample_8kev * scipy.ndimage.morphology.binary_erosion(
                    sample_8kev, structure=np.ones(scale_erosion)
            )
            sample_10kev = sample_10kev * scipy.ndimage.morphology.binary_erosion(
                    sample_10kev, structure=np.ones(scale_erosion)
            )
        if median_filter:
            print("applying median filter with size of ", str(scale_median))
            sample_8kev = scipy.signal.medfilt2d(sample_8kev, scale_median)
            sample_10kev = scipy.signal.medfilt2d(sample_10kev, scale_median)
        comp = find_comp_bayes_ter(sample_8kev, sample_10kev, table_joint, combination)
        ddir = config_info['img_out_dir']
        os.makedirs(config_info["img_out_dir"]+"/std1", exist_ok=True)                        
        os.makedirs(config_info["img_out_dir"]+"/std2", exist_ok=True)
        os.makedirs(config_info["img_out_dir"]+"/std3", exist_ok=True)                                
        filename1 = os.path.join(ddir, 'std1/' + str(slice_num).zfill(4))
        filename2 = os.path.join(ddir, 'std2/' + str(slice_num).zfill(4))
        filename3 = os.path.join(ddir, 'std3/' + str(slice_num).zfill(4))

        np.save(filename1, comp[0])
        np.save(filename2, comp[1])
        np.save(filename3, comp[2])

    print("Writing sample results")
    a = get_total_ratio_ter(
            os.path.join(ddir, 'std1'), os.path.join(ddir, 'std2'), os.path.join(ddir, 'std3')
    )
    ratio = open(ddir + "/ratio.txt", mode="w")
    para = (
            "Mineral standard 1 = "
            + str(std1_name)
            + ", "
            + "Mineral standard 2 ="
            + str(std2_name)
            + ", "
            + "Mineral standard 3 ="
            + str(std3_name)
            + ", "
            + ", resolution of classes = "
            + str(step)
            + ", down sampling factor = "
            + str(down_sample_scale)
            + "\n erosion filter "
            + str(erosion)
            + " by factor "
            + str(scale_erosion)
            + "\n median filter "
            + str(median_filter)
            + " by factor "
            + str(scale_median)
    )
    print(f"ratio is [{std1_name},{std2_name},{std3_name}] =", a)
    ratio.writelines(str(a))
    ratio.writelines(para)
