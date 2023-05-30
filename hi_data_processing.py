from functions import data_reduction, running_difference, make_jplot, download_files, reduced_pngs, check_calfiles
from itertools import repeat
import numpy as np
import datetime
from time import time as timer
import os

start_t = timer()


def main():

    config_path = 'config.txt'
    file = open(config_path, 'r')

    config = file.readlines()

    path = config[0].splitlines()[0]
    save_path = config[1].splitlines()[0]
    datpath = config[2].splitlines()[0]
    ftpsc = config[3].splitlines()[0]
    instrument = config[4].splitlines()[0]
    bflag = config[5].splitlines()[0]
    start = config[6].splitlines()[0]
    mode = config[7].splitlines()[0]
    task = config[8].splitlines()[0]
    save_img = config[9].splitlines()[0]
    silent = config[10].splitlines()[0]

    if save_img == 'save_rdif_img':
        save_img = True

    else:
        save_img = False

    if silent == 'silent':
        silent = True

    else:
        silent = False

    if bflag == 'science':
        path_flg = 'L0'

    if bflag == 'beacon':
        path_flg = bflag

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    date = datetime.datetime.strptime(start, '%Y%m%d')

    if mode == 'week':
        duration = 8

    if mode == 'month':
        duration = 32
        
    interv = np.arange(duration)

    datelist = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv]

    check_calfiles(datpath)

    if task == 'download':

        download_files(start, duration, save_path, ftpsc, instrument, bflag, silent)

        print('\n')

        print('Files saved to:', save_path + path_flg + '/' + sc[0] + '/img/hi_1/')

        print('--------------------------------------------------------------------------------')

        print('Files saved to:', save_path + path_flg + '/' + sc[0] + '/img/hi_2/')

    if task == 'reduction':
        
        for i in range(len(datelist)):
            data_reduction(datelist[i], path, datpath, ftpsc, instrument, bflag, silent, save_path, path_flg)

        print('\n')

        print('Files saved to:', path + 'reduced/chosen_dates/' + sc[0].upper() + '/' + bflag + '/hi_1/')

        print('--------------------------------------------------------------------------------')

        print('Files saved to:', path + 'reduced/chosen_dates/' + sc[0].upper() + '/' + bflag + '/hi_2/')

    if task == 'difference':

        for i in range(len(datelist)):
            running_difference(datelist[i], path, datpath, ftpsc, instrument, bflag, silent, save_img)

        print('\n')

        print('Pickle files saved to:', path + 'running_difference/data/' + sc[0].upper() + '/' + bflag + '/hi_1/chosen_dates/')

        print('--------------------------------------------------------------------------------')

        print('Pickle files saved to:', path + 'running_difference/data/' + sc[0].upper() + '/' + bflag + '/hi_2/chosen_dates/')

        print('--------------------------------------------------------------------------------')

        if save_img:
            print('jpeg/png files saved to:', path + 'running_difference/pngs/' + sc[0].upper() + '/' + bflag + '/hi_1/chosen_dates/')

            print('--------------------------------------------------------------------------------')

            print('jpeg/png files saved to:', path + 'running_difference/pngs/' + sc[0].upper() + '/' + bflag + '/hi_2/chosen_dates/')

    if task == 'jplot':

        make_jplot(start, duration, path, datpath, ftpsc, instrument, bflag, silent)

        print('\n')

        print('Jplots saved to:', path + 'jplot/' + sc[0].upper() + '/' + bflag + '/hi_1/' + str(start[0:4]) + '/')

        print('--------------------------------------------------------------------------------')

        print('Jplots saved to:', path + 'jplot/' + sc[0].upper() + '/' + bflag + '/hi_2/' + str(start[0:4]) + '/')

        print('--------------------------------------------------------------------------------')

    if task == 'all':

        download_files(start, duration, save_path, ftpsc, instrument, bflag, silent)

        for i in range(len(datelist)):
            data_reduction(datelist[i], path, datpath, ftpsc, instrument, bflag, silent, save_path, path_flg)

        for i in range(len(datelist)):
            running_difference(datelist[i], path, datpath, ftpsc, instrument, bflag, silent, save_img)

        make_jplot(start, duration, path, datpath, ftpsc, instrument, bflag, silent)

    if task == 'reduced_pngs':

        for i in range(len(datelist)):
            reduced_pngs(datelist[i], path, bflag, silent)


    print('\n')

    print('Done.')

    hours, rem = divmod(timer() - start_t, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Elapsed Time: {} minutes {} seconds".format(int(minutes), int(seconds)))


if __name__ == '__main__':
    main()
