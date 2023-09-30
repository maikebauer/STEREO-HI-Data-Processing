from functions import data_reduction, running_difference, make_jplot, download_files, reduced_pngs, check_calfiles, check_pointfiles, get_bkgd
from itertools import repeat
import numpy as np
import datetime
from time import time as timer
import os
import subprocess
import sys

def main():

    if os.path.isfile('config.txt'):
        config_path = 'config.txt'

    else:
        config_path = 'sample_config.txt'

    file = open(config_path, 'r')

    config = file.readlines()

    path = config[0].splitlines()[0]
    save_path = config[1].splitlines()[0]
    datpath = config[2].splitlines()[0]
    ftpsc = config[3].splitlines()[0].split(',')
    instrument = config[4].splitlines()[0]
    bflag = config[5].splitlines()[0].split(',')
    start = config[6].splitlines()[0].split(',')
    mode = config[7].splitlines()[0]
    task = config[8].splitlines()[0]
    save_img = config[9].splitlines()[0]
    silent = config[10].splitlines()[0]

    list_len = len(ftpsc)

    if any(len(lst) != list_len for lst in [ftpsc, bflag, start]):
        print('Number of specified spacecraft, dates, and/or science/beacon arguments does not match. Exiting...')
        sys.exit()

    if save_img == 'save_rdif_img':
        save_img = True

    else:
        save_img = False

    if silent == 'silent':
        silent = True

    else:
        silent = False

    if mode == 'week':
        duration = 7

    if mode == 'month':
        duration = 31

    for num in range(list_len):

        start_t = timer()

        if bflag[num] == 'science':
            path_flg = 'L0'

        if bflag[num] == 'beacon':
            path_flg = 'beacon'

        if ftpsc[num] == 'A':
            sc = 'ahead'

        if ftpsc[num] == 'B':
            sc = 'behind'

        bg_dur = 7
        date = datetime.datetime.strptime(start[num], '%Y%m%d')
        date_red = datetime.datetime.strptime(start[num], '%Y%m%d') - datetime.timedelta(days=bg_dur+1) 
        
        interv = np.arange(duration)
        interv_red = np.arange(duration+bg_dur+1)

        datelist = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv]
        datelist_red = [datetime.datetime.strftime(date_red + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv_red]
        
        check_calfiles(datpath)
        check_pointfiles(datpath)
        print('Starting processing for event ' + start[num] + ' (SC: ' + ftpsc[num] + ', mode: ' + bflag[num] + ')' + '...')
        
        if task == 'download':
         
            download_files(start[num], duration, save_path, ftpsc[num], instrument, bflag[num], silent)

            print('\n')

            print('Files saved to:', save_path + 'stereo' + ftpsc[num][0].lower() + '/')

        if task == 'reduction':
            #start one day earlier for reduction only
            for i in range(len(datelist_red)):
                data_reduction(datelist_red[i], path, datpath, ftpsc[num], instrument, bflag[num], silent, save_path, path_flg)

            print('\n')

            print('Files saved to:', path + 'reduced/chosen_dates/' + sc[0].upper() + '/' + bflag[num] + '/')

        if task == 'difference':

            for i in range(len(datelist)):
                bkgd = get_bkgd(path, ftpsc[num], datelist[i], bflag[num], instrument)
                running_difference(datelist[i], bkgd, path, datpath, ftpsc[num], instrument, bflag[num], silent, save_img)

            print('\n')

            print('Fits files saved to:', path + 'running_difference/data/' + sc[0].upper() + '/' + bflag[num] + '/')

            if save_img:
                print('jpeg/png files saved to:', path + 'running_difference/pngs/' + sc[0].upper() + '/' + bflag[num] + '/')

        if task == 'jplot':

            make_jplot(start[num], duration, path, datpath, ftpsc[num], instrument, bflag[num], save_path, path_flg, silent)

            print('\n')

            print('Jplots saved to:', path + 'jplot/' + sc[0].upper() + '/' + bflag[num] + '/')

        if task == 'all':

            download_files(start[num], duration, save_path, ftpsc[num], instrument, bflag[num], silent)
            
            for i in range(len(datelist_red)):
                data_reduction(datelist_red[i], path, datpath, ftpsc[num], instrument, bflag[num], silent, save_path, path_flg)
            
            for i in range(len(datelist)):
                bkgd = get_bkgd(path, ftpsc[num], datelist[i], bflag[num], instrument)
                running_difference(datelist[i], bkgd, path, datpath, ftpsc[num], instrument, bflag[num], silent, save_img)

            make_jplot(start[num], duration, path, datpath, ftpsc[num], instrument, bflag[num], save_path, path_flg, silent)

        if task == 'reduced_pngs':

            for i in range(len(datelist)):
                reduced_pngs(datelist[i], path, bflag[num], silent)


        print('\n')

        print('Done.')

        hours, rem = divmod(timer() - start_t, 3600)
        minutes, seconds = divmod(rem, 60)
        subprocess.call(['chmod', '-R', '775', path])
        
        print("Elapsed Time: {} minutes {} seconds".format(int(minutes), int(seconds)))


if __name__ == '__main__':
    main()
