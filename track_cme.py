import matplotlib
from pandas.plotting import register_matplotlib_converters
import os
import sys
from functions import parse_yml, load_jplot, track_jplot

#matplotlib.use("Qt5Agg")

register_matplotlib_converters()

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

jplot_type = config['task_spec']['jplot_spec']
jp_name = '' if jplot_type == 'median' else '_no_median'

if any(len(lst) != list_len for lst in [config['spacecraft'], config['data_type'], config['start_date']]):
    print('Number of specified spacecraft, dates, and/or science/beacon arguments does not match. Exiting...')
    sys.exit()

for num in range(1):

    if config['instrument'] == 'hi1hi2':

        img_rescale_h1, orig_h1, t_beg_h1, t_end_h1, e_beg_h1, e_end_h1 = load_jplot(config['output_directory'], ins_list[0], config['spacecraft'][num], config['data_type'][num], config['start_date'][num], config['end_date'][num], jp_name)
        img_rescale_h2, orig_h2, t_beg_h2, t_end_h2, e_beg_h2, e_end_h2 = load_jplot(config['output_directory'], ins_list[1], config['spacecraft'][num], config['data_type'][num], config['start_date'][num], config['end_date'][num], jp_name)

        img_rescale = [img_rescale_h1, img_rescale_h2]
        orig = [orig_h1, orig_h2]
        t_beg = [t_beg_h1, t_beg_h2]
        t_end = [t_end_h1, t_end_h2]
        e_beg = [e_beg_h1, e_beg_h2]
        e_end = [e_end_h1, e_end_h2]
    
    elif config['instrument'] == 'hi_1':

        img_rescale, orig, t_beg, t_end, e_beg, e_end = load_jplot(config['output_directory'], ins_list[0], config['spacecraft'][num], config['data_type'][num], config['start_date'][num], config['end_date'][num], jp_name)
        
        img_rescale = [img_rescale]
        orig = [orig]
        t_beg = [t_beg]
        t_end = [t_end]
        e_beg = [e_beg]
        e_end = [e_end]


    elif config['instrument'] =='hi_2':

        img_rescale, orig, t_beg, t_end, e_beg, e_end = load_jplot(config['output_directory'], ins_list[0], config['spacecraft'][num], config['data_type'][num], config['start_date'][num], config['end_date'][num], jp_name)

        img_rescale = [img_rescale]
        orig = [orig]
        t_beg = [t_beg]
        t_end = [t_end]
        e_beg = [e_beg]
        e_end = [e_end]
    
    track_jplot(img_rescale, orig, t_beg, t_end, e_beg, e_end, config['spacecraft'][num], config['data_type'][num], config['start_date'][num], config['end_date'][num], jp_name, config['instrument'], config['output_directory'])