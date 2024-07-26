from functions import data_reduction, running_difference, make_jplot, download_files, reduced_pngs, check_calfiles, check_pointfiles, get_bkgd, reduced_nobg, parse_yml
import numpy as np
import datetime
from time import time as timer
import os
import subprocess
import sys


def main():

    if os.path.isfile('config.yml'):
        config_path = 'config.yml'

    else:
        config_path = 'sample_config.yml'

    config = parse_yml(config_path)

    if config['task'] == 'jplot' or config['task'] == 'all':
        jplot_type = config['task_spec']['jplot_spec']
    
    if config['task'] == 'difference' or config['task'] == 'all':
        save_img = config['task_spec']['save_img']
    
    list_len = len(config['spacecraft'])

    if any(len(lst) != list_len for lst in [config['spacecraft'], config['data_type'], config['start_date']]):
        print('Number of specified spacecraft, dates, and/or science/beacon arguments does not match. Exiting...')
        sys.exit()
        
    if config['instrument'] == 'hi1hi2':
        ins_list = ['hi_1', 'hi_2']
    
    if config['instrument'] == 'hi_1':
        ins_list = ['hi_1']
    
    if config['instrument'] == 'hi_2':
        ins_list = ['hi_2']
                    
    for num in range(list_len):

        start_t = timer()

        if config['data_type'][num] == 'science':
            path_flg = 'L0'

        if config['data_type'][num] == 'beacon':
            path_flg = 'beacon'

        if config['spacecraft'][num] == 'A':
            sc = 'ahead'

        if config['spacecraft'][num] == 'B':
            sc = 'behind'

        date = datetime.datetime.strptime(config['start_date'][num], '%Y%m%d')
        date_end = datetime.datetime.strptime(config['end_date'][num], '%Y%m%d')

        date_red = datetime.datetime.strptime(config['start_date'][num], '%Y%m%d') - datetime.timedelta(days=config['background_length']) 
        
        datelist = np.arange(date, date_end + datetime.timedelta(days=1), datetime.timedelta(days=1)).astype(datetime.datetime)
        datelist = [dat.strftime('%Y%m%d') for dat in datelist]

        datelist_red = np.arange(date_red, date_end + datetime.timedelta(days=1), datetime.timedelta(days=1)).astype(datetime.datetime)
        datelist_red = [dat.strftime('%Y%m%d') for dat in datelist_red]
        
        check_calfiles(config['solarsoft_directory'])
        check_pointfiles(config['solarsoft_directory'])
        
        print('Starting processing for event ' + config['start_date'][num] + ' (SC: ' + config['spacecraft'][num] + ', mode: ' + config['data_type'][num] + ')' + '...')
        
        if config['task'] == 'download':
         
            download_files(datelist_red, config['data_directory'], config['spacecraft'][num], ins_list, config['data_type'][num], config['silent_mode'])

            print('\n')

            print('Files saved to:', config['data_directory'] + 'stereo' + config['spacecraft'][num][0].lower() + '/')

        if config['task'] == 'reduction':

            for i in range(len(datelist_red)):
                data_reduction(datelist_red[i], config['output_directory'], config['solarsoft_directory'], config['spacecraft'][num], ins_list, config['data_type'][num], config['silent_mode'], config['data_directory'], path_flg)

            print('\n')

            print('Files saved to:', config['output_directory'] + 'reduced/chosen_dates/' + sc[0].upper() + '/' + config['data_type'][num] + '/')

        if config['task'] == 'difference':

            for i in range(len(datelist)):
                for ins in ins_list:
                  bkgd = get_bkgd(config['output_directory'], config['spacecraft'][num], datelist[i], config['data_type'][num], ins, config['background_length'])
                  running_difference(datelist[i], bkgd, config['output_directory'], config['solarsoft_directory'], config['spacecraft'][num], ins, config['data_type'][num], config['silent_mode'], save_img)

            print('\n')

            print('Fits files saved to:', config['output_directory'] + 'running_difference/data/' + sc[0].upper() + '/' + config['data_type'][num] + '/')

            if save_img:
                print('jpeg/png files saved to:', config['output_directory'] + 'running_difference/pngs/' + sc[0].upper() + '/' + config['data_type'][num] + '/')

        if config['task'] == 'jplot':

            make_jplot(datelist, config['output_directory'], config['spacecraft'][num], config['instrument'], config['data_type'][num], config['data_directory'], config['silent_mode'], jplot_type)

            print('\n')

            print('Jplots saved to:', config['output_directory'] + 'jplot/' + sc[0].upper() + '/' + config['data_type'][num] + '/')

        if config['task'] == 'all':

            download_files(datelist_red, config['data_directory'], config['spacecraft'][num], ins_list, config['data_type'][num], config['silent_mode'])
            
            for i in range(len(datelist_red)):
                data_reduction(datelist_red[i], config['output_directory'], config['solarsoft_directory'], config['spacecraft'][num], ins_list, config['data_type'][num], config['silent_mode'], config['data_directory'], path_flg)
            
            for i in range(len(datelist)):
                for ins in ins_list:
                    bkgd = get_bkgd(config['output_directory'], config['spacecraft'][num], datelist[i], config['data_type'][num], ins, config['background_length'])
                    running_difference(datelist[i], bkgd, config['output_directory'], config['solarsoft_directory'], config['spacecraft'][num], ins, config['data_type'][num], config['silent_mode'], save_img)

            make_jplot(datelist, config['output_directory'], config['spacecraft'][num], config['instrument'], config['data_type'][num], config['data_directory'], config['silent_mode'], jplot_type)

        if config['task'] == 'reduced_pngs':

            for i in range(len(datelist)):
                reduced_pngs(datelist[i], config['output_directory'], config['data_type'][num], config['silent_mode'])
                
        if config['task'] == 'reduced_nobg':
            
            for i in range(len(datelist)):
                for ins in ins_list:
                    bkgd = get_bkgd(config['output_directory'], config['spacecraft'][num], datelist[i], config['data_type'][num], ins, config['background_length'])
                    reduced_nobg(datelist[i], bkgd, config['output_directory'], config['solarsoft_directory'], config['spacecraft'][num], ins, config['data_type'][num], config['silent_mode'])

        print('\n')

        print('Done.')

        hours, rem = divmod(timer() - start_t, 3600)
        minutes, seconds = divmod(rem, 60)
        subprocess.call(['chmod', '-R', '775', config['output_directory']])
        
        print("Elapsed Time: {} minutes {} seconds".format(int(minutes), int(seconds)))


if __name__ == '__main__':
    main()
