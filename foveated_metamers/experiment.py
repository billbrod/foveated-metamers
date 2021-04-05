#!/usr/bin/python
"""psychopy script for metamer experiment, run from the command line
"""

import argparse
import h5py
import os
import re
import os.path as op
import datetime
import numpy as np
import pandas as pd
from psychopy import visual, core, event, clock
from psychopy.tools import imagetools
import analysis


def clear_events(win):
    """clear all keys from psychopy

    I've been having issues with keys remaining in psychopy's buffer,
    being pulled in at a later time (thus, pressing the spacebar once
    gets parsed as pressing it twice, in rapid succession). This takes a
    brief pause, .1 seconds, to rapidly flip the window several times
    (should look like just a gray screen), calling ``event.getKeys()``
    each time in order to make sure no keys are left in the buffer

    Parameters
    ----------
    win : Psychophy.visual.Window
        window to flip

    """
    for i in range(10):
        win.flip()
        core.wait(.01)
        event.getKeys()


def calc_pct_correct(raw_behavioral_path, idx, stim_df):
    """Calculate percent correct, grouped by scaling.

    This is only intended for use during the training sessions, to give
    participant a sense of how well they did.

    Parameters
    ----------
    raw_behavioral_path : str
        The str to the hdf5 file that contains the behavioral results,
        as saved by the experiment.py script
    idx : np.array
        The n_trials by 3 array containing the stimuli presentation
        indices for the run being analyzed.
    stim_df : pd.DataFrame
        The metamer information dataframe, as created by
        stimuli.create_metamer_df

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the columns 'scaling' and 'pct_correct'

    """
    trials = analysis.summarize_trials(raw_behavioral_path)
    df = analysis.create_experiment_df_split(stim_df, idx)
    df = analysis.add_response_info(df, trials, 'training', 'training', 'training')
    df = df.groupby('scaling').hit_or_miss_numeric.mean()
    return df.reset_index().rename(columns={'hit_or_miss_numeric': 'pct_correct'})


def _convert_str(list_of_strs):
    """convert strs to hdf5-savable format

    python 3 strings are more complicated than python 2, see
    http://docs.h5py.org/en/latest/strings.html and https://github.com/h5py/h5py/issues/892
    """
    list_of_strs = np.array(list_of_strs)
    saveable_list = []
    for x in list_of_strs:
        try:
            x = x.encode()
        except AttributeError:
            try:
                # then this is not a string but another list of strings
                x = [i.encode() for i in x]
            except AttributeError:
                # in this case, then it's numpy.bytes or something else
                # that's fine
                pass
        saveable_list.append(x)
    return saveable_list


def _insert_into_hdf5(f, key, value):
    """insert value into hdf5 file with given key

    if you try to set a dataset that already exists, it raises an
    exception, so this tries to create a dataset and, if it raises that
    exception, pop off the corresponding key and set it anew

    """
    try:
        f.create_dataset(key, data=value)
    except (RuntimeError, OSError):
        # then it already exists
        f.pop(key)
        f.create_dataset(key, data=value)


def save(file_path, stimuli_path, idx_path, keys, timings, expt_params, idx, **kwargs):
    dataset_names = ['button_presses', 'timing_data', 'stimuli_path', 'idx_path',
                     'shuffled_indices']
    with h5py.File(file_path, 'a') as f:
        for k, d in zip(dataset_names, [_convert_str(keys), _convert_str(timings),
                                        stimuli_path.encode(), idx_path.encode(), idx]):
            _insert_into_hdf5(f, k, d)
        for k, v in expt_params.items():
            _insert_into_hdf5(f, "%s" % k, v)
        # also note differences from default options
        for k, v in kwargs.items():
            if v is None:
                _insert_into_hdf5(f, "%s" % k, str(v))
            else:
                _insert_into_hdf5(f, "%s" % k, v)


def _set_params(stimuli_path, idx_path, size=[1920, 1080], monitor='CBI-prisma-projector',
                units='pix', fullscr=True, screen=[0], color=128, colorSpace='rgb255',
                allowGUI=False, **monitor_kwargs):
    """set the various experiment parameters
    """
    stimuli = np.load(stimuli_path)
    idx = np.load(idx_path)
    expt_params = {}
    # final two dimensions are always the size of the stimuli
    expt_params['stimuli_size'] = stimuli.shape[-2:]
    if idx.ndim == 2:
        # we do some nifty indexing here. idx will be a 2d array, i x 3
        # (where i is the number of trials we want), so when we do the
        # following, we go from n x *img_size array (where n is the max
        # number of images) to i x 3 x *img_size array
        stimuli = stimuli[idx, :, :]
        # the first dimension of stimuli (retrieved by len) is how many
        # trials we have. the next one shows the number of stimuli per
        # trial (3).
        expt_params['trial_num'] = stimuli.shape[0]
        expt_params['stimuli_per_trial'] = stimuli.shape[1]
    elif idx.ndim == 3:
        left_stimuli = stimuli[idx[0], ..., :1300]
        right_stimuli = stimuli[idx[1], ..., 1300:]
        stimuli = np.array([left_stimuli, right_stimuli])
        # the first dimension of stimuli is left/right the second is how many
        # trials we have, and the next one shows the number of stimuli per
        # trial (2)
        expt_params['trial_num'] = stimuli.shape[1]
        expt_params['stimuli_per_trial'] = stimuli.shape[2]
    # array size and stimulus size are backwards of each other, so
    # need to reverse this
    expt_params['stimuli_size'] = expt_params['stimuli_size'][::-1]

    # these are all a variety of kwargs used by monitor
    monitor_kwargs.update({'size': size, 'monitor': monitor, 'units': units, 'fullscr': fullscr,
                           'screen': screen, 'color': color, 'colorSpace': colorSpace,
                           'allowGUI': allowGUI, "gamma": 1})
    return stimuli, idx, expt_params, monitor_kwargs


def check_for_keys(all_keys, keys_to_check=['q', 'esc', 'escape']):
    all_keys = [k[0] for k in all_keys]
    return any([k in all_keys for k in keys_to_check])


def countdown(win, text_height=50):
    countdown_text = visual.TextStim(win, '3', height=text_height)
    for i in range(3)[::-1]:
        countdown_text.text = str(i+1)
        countdown_text.draw()
        win.flip()
        core.wait(1)


def pause(current_i, total_imgs, win, expt_clock, text_height=50):
    pause_text = visual.TextStim(win, f"{current_i}/{total_imgs}\nspace to resume\nq or esc to quit",
                                 height=text_height)
    pause_text.draw()
    win.flip()
    all_keys = event.waitKeys(keyList=['return', 'space', 'q', 'escape', 'esc'], timeStamped=expt_clock)
    clear_events(win)
    return [(key[0], key[1]) for key in all_keys]


def _create_bar_mask(bar_height, bar_width=200, fringe_proportion=.5):
    """Create central bar with raised-cosine edges.

    """
    x = np.linspace(-bar_width//2, bar_width//2, bar_width)
    fringe_width = fringe_proportion * x.max()
    def raised_cos(x, start_x, end_x):
        x = (x-start_x) / (end_x - start_x)
        return .5*(1+np.cos(np.pi*x))
    mask = np.piecewise(x, [x < -fringe_width, (x > -fringe_width) & (x < fringe_width), fringe_width < x],
                       [lambda x: raised_cos(x, -fringe_width, x.min()), 1,
                        lambda x: raised_cos(x, fringe_width, x.max())])
    # for psychopy, 1 mean fully opaque and -1 means fully transparent
    mask = mask * 2 - 1
    mask = np.repeat(np.expand_dims(mask, 0), bar_height, 0)
    return mask


def _setup_run(stimuli_path, idx_path, fix_deg_size=.25, screen_size_deg=73.45,
               take_break=True, timings=[], start_from_stim=0,
               **monitor_kwargs):
    """Setup the run.

    Does the thigns that are constant across different tasks.

    """
    stimuli, idx, expt_params, monitor_kwargs = _set_params(stimuli_path, idx_path,
                                                            **monitor_kwargs)
    if idx.ndim == 2:
        stimuli = stimuli[start_from_stim:]
        total_trials = stimuli.shape[0]
    elif idx.ndim == 3:
        stimuli = stimuli[:, start_from_stim:]
        total_trials = stimuli.shape[1]
    print("Starting from stimulus %s" % start_from_stim)

    if len(monitor_kwargs['screen']) == 1:
        screen = monitor_kwargs.pop('screen')[0]
        print('Running on screen %s' % screen)
        win = visual.Window(winType='pyglet', screen=screen, **monitor_kwargs)
    else:
        raise Exception("Can't handle %s screens!" % len(monitor_kwargs['screen']))

    break_time = total_trials // 2
    if take_break:
        print("%s total trials, will take break after number %s" %
              (total_trials, break_time))

    fix_pix_size = fix_deg_size * (monitor_kwargs['size'][0] / screen_size_deg)
    # two fixation dots: one red and one green. Only show green one during
    # training to show response was correct, else show red.
    fixation = {'default': visual.GratingStim(win, size=fix_pix_size, sf=0, color='red',
                                              mask='circle'),
                'correct': visual.GratingStim(win, size=fix_pix_size, sf=0, color='green',
                                              mask='circle')}

    timer = clock.StaticPeriod(screenHz=60)
    expt_clock = clock.Clock()
    if timings:
        # if we've been passed something, make sure everything happens
        # after it. for some reason, need to use the negative value of
        # it in order to add that to the expt_clock...
        expt_clock.reset(-float(timings[-1][-1]))
    return (stimuli, idx, expt_params, monitor_kwargs, win, break_time,
            fixation, timer, expt_clock, screen)


def _explain_task(win, expt_clock, comparison, fixation, text_height=50,
                  train_flag=False):
    """Draw some text explaining the task
    """
    if comparison == 'met':
        comp_text = "On this run, you'll be comparing two synthesized images."
    elif comparison == 'ref':
        comp_text = ("On this run, you'll be comparing natural and synthesized images."
                     " The first image can be either natural or synthesized, so pay attention!")
    if train_flag:
        train_text = "For this training run, there will only be two natural images and "
        feedback_text = ("Because this is training run, you will receive feedback "
                         "after each trial (fixation dot will turn green if you "
                         "were correct) and we will show you your performance "
                         "at the end of the run. You should get 100% on ")
        if train_flag == 'noise':
            train_text += 'two noise patches.\n\n'
            comp_text = comp_text.replace('synthesized images', 'noise patches')
            feedback_text += 'this run.'
        elif train_flag == 'model':
            train_text += "two possible synthesized image for each: one easy and one hard.\n\n"
            feedback_text += 'the easy trials, but will do worse on the hard ones.'
        duration_text = 'one or two minutes'
    else:
        train_text = ""
        duration_text = "twelve minutes"
        feedback_text = "You will receive no feedback, either during or after the run."
    text = ("In this experiment, you'll be performing a Two-Alternative Forced Choice task: "
            "you'll view an image, split in half, and then, after a brief delay, a second "
            "image, also split in half. One half of the second image will be the same as the "
            "first, but the other half will have changed. Your task is to press the left or "
            "right button to say which half you think changed. You have as much time as you "
            "need, but respond as quickly as you can. All the images will be presented for a "
            "very brief period of time, so pay attention. Sometimes the two images will be "
            "very similar; sometimes they'll be very different. For the similar images, we "
            f"expect the task to be hard. Just do your best!\n\n{comp_text}\n\n{train_text}"
            "Fixate your eyes on the center of the image (there will be a fixation dot)"
            " and try not to move them.\n\n"
            f'{feedback_text}\n\n'
            f"The run will last for about {duration_text} and there will be a break halfway "
            "through. When you've finished the run, take a brief break before beginning the"
            " next one.\n\nPress space to continue")
    explain_text = visual.TextStim(win, text, height=text_height, wrapWidth=2000)

    explain_text.draw()
    fixation['default'].draw()
    win.flip()
    all_keys = event.waitKeys(keyList=['return', 'space', ], timeStamped=expt_clock)
    clear_events(win)
    return [(key[0], key[1]) for key in all_keys]


def _end_run(win, timings, text_height, expt_clock, train_flag=False):
    """End the run.

    Do the things that are shared across task types.

    """
    if not train_flag:
        visual.TextStim(win, "Run over.", height=text_height,
                        wrapWidth=2000).draw()
    else:
        visual.TextStim(win, "Run over\n\nWait a sec while we compute your performance...",
                        height=text_height, wrapWidth=2000).draw()
    win.flip()
    timings.append(("run_end", '', expt_clock.getTime()))
    all_keys = event.getKeys(timeStamped=expt_clock)
    core.wait(4)
    if not train_flag:
        win.close()
    return all_keys


def run_split(stimuli_path, idx_path, save_path, comparison,
              on_msec_length=200, off_msec_length=(500, 500), fix_deg_size=.25,
              screen_size_deg=73.45, take_break=True, keys_pressed=[],
              timings=[], start_from_stim=0, text_height=50, bar_deg_size=2,
              train_flag=False, **monitor_kwargs):
    r"""Run one run of the split task.

    stimuli_path specifies the path of the unshuffled experiment stimuli, while
    idx_path specifies the path of the shuffled indices to use for this run.
    This function will load in the stimuli at stimuli_path and rearrange them
    using the indices found at idx_path, then simply go through those stimuli
    in order, showing each stimuli for ``on_msec_length`` msecs and then a
    blank screen for ``off_msec_length[i]`` msecs (or as close as possible,
    given the monitor's refresh rate; ``i`` depends on whether this was between
    the two stimuli or between trials, though the actual time between trials
    will be the sum of this number, the length of the pause, and the subject's
    response time).

    For fixation, we show a simple red dot whose size (in degrees) is
    specified by ``fix_deg_size``

    We load the arrays saved at stimuli_path and idx_path, and show the stimuli
    whose indices are found in the index array. So if `idx=np.array([0, 4,
    2])`, we would only show the 1st, 3rd, and 5th elements. This can be used
    to show a subset of all stimuli

    Arguments
    ============
    stimuli_path : string
        path to .npy file where stimuli are stored (as 3d array)
    idx_path : string
        path to .npy file where shuffled indices are stored (as 2d
        array)
    save_path : string
        path to .hdf5 file where we'll store the outputs of this
        experiment (we save every trial)
    comparison : {'ref', 'met'}
        whether we're comparing two metamers or a metamer and a reference image
    on_msec_length : int
        length of the ON blocks in milliseconds; that is, the length of
        time to display each stimulus
    off_msec_length : tuple
        2-tuple of ints specifying the length of the length of the OFF blocks
        in milliseconds. The first number gives the length of time between
        stimuli, while the second gives the length of the pause between trials
        (so the actual time between one trial's second stimulus going off and
        the next trial's first one going on is the sum of this and the subjet's
        response time).
    fix_deg_size : int
        the size of the fixation digits, in degrees.
    screen_size_deg : int or float.
        the max visual angle (in degrees) of the full screen.
    take_break : bool
        Whether to take a break half-way through the experiment or not.
    keys_pressed : list
        keys pressed. should be empty list unless resuming a run.
    timings : list
        timing of events. should be empty list unless resuming a run.
    start_from_stim : int
        the first stimulus to show. should be 0 unless resuming a run.
    text_height : int
        The text height in pixels.
    bar_deg_size : float
        The width of the central bar dividing left and right half of stimulus,
        in degrees
    train_flag : bool
        Whether this is a training run or not. If so, the instruction text has
        some extra words and we show the percent correct at the end.
    monitor_kwargs :
        passed to visual.Window

    """
    setup_args = _setup_run(stimuli_path, idx_path, fix_deg_size,
                            screen_size_deg, take_break, timings,
                            start_from_stim, **monitor_kwargs)

    (stimuli, idx, expt_params, monitor_kwargs, win, break_time,
     fixation, timer, expt_clock, screen) = setup_args

    if train_flag:
        correct_responses = np.load(idx_path.replace('idx_', '').replace('.npy', '_correct-responses.npy'))
    else:
        correct_responses = np.zeros_like(idx)

    bar_pix_size = int(bar_deg_size * (monitor_kwargs['size'][0] / screen_size_deg))
    bar_size = [bar_pix_size, expt_params['stimuli_size'][1]]
    center_mask = visual.GratingStim(win, size=bar_size, sf=0,
                                     color=monitor_kwargs['color'],
                                     mask=_create_bar_mask(*bar_size[::-1]),
                                     colorSpace=monitor_kwargs['colorSpace'])

    stim_size = [expt_params['stimuli_size'][0]/2, expt_params['stimuli_size'][1]]
    left_stimuli = stimuli[0]
    left_img = visual.ImageStim(win, image=imagetools.array2image(left_stimuli[0, 0]),
                                size=stim_size, pos=(-stim_size[0]/2, 0))

    right_stimuli = stimuli[1]
    right_img = visual.ImageStim(win, image=imagetools.array2image(right_stimuli[0, 0]),
                                 size=stim_size, pos=(stim_size[0]/2, 0))
    del stimuli

    _explain_task(win, expt_clock, comparison, fixation, text_height,
                  train_flag=train_flag)

    wait_text = visual.TextStim(win, ("Press space to start\nq or esc will quit\nspace to pause"),
                                height=text_height)
    query_text = visual.TextStim(win, "Did image change on left or right?",
                                 height=text_height)
    wait_text.draw()
    win.flip()

    all_keys = event.waitKeys(keyList=['return', 'space', 'q', 'escape', 'esc'], timeStamped=expt_clock)
    clear_events(win)

    if check_for_keys(all_keys):
        win.close()
        return all_keys, [], expt_params, idx
    countdown(win, text_height)

    timings.append(("start", "off", expt_clock.getTime()))

    # this outer for loop is per trial
    for i, (l_stim, r_stim) in enumerate(zip(left_stimuli, right_stimuli)):
        # and this one is for the two stimuli in each trial
        all_keys = []
        for j in range(len(l_stim)):
            left_img.draw()
            right_img.draw()
            center_mask.draw()
            fixation['default'].draw()
            win.flip()
            timings.append(("stimulus_%d-%d" % (i+start_from_stim, j), "on", expt_clock.getTime()))
            # convert to sec
            core.wait(on_msec_length / 1000)
            if j == 1:
                query_text.draw()
            else:
                fixation['default'].draw()
            win.flip()
            timings.append(("stimulus_%d-%d" % (i+start_from_stim, j), "off",
                            expt_clock.getTime()))
            if j != 1:
                timer.start(off_msec_length[j] / 1000)
            # off msec lengths are always longer than on msec length, so
            # we preload the next image here
            try:
                # we either load the next one in the set of three for
                # this trial...
                left_img.image = imagetools.array2image(l_stim[j+1])
                right_img.image = imagetools.array2image(r_stim[j+1])
            except IndexError:
                # or, if we've gone through all those, we load the first
                # image for the next trial. if i+1==len(stimuli), then we've
                # gone through all images and will quit out
                if i+1 < len(left_stimuli):
                    left_img.image = imagetools.array2image(left_stimuli[i+1][0])
                    right_img.image = imagetools.array2image(right_stimuli[i+1][0])
            if j == 1:
                response_keys = event.waitKeys(keyList=['q', 'escape', 'esc', '1', '2'],
                                               timeStamped=expt_clock)
                all_keys.extend(response_keys)
            else:
                timer.complete()
                all_keys.extend(event.getKeys(timeStamped=expt_clock))
            # we need this double break because we have two for loops
            if check_for_keys(all_keys):
                break
        if all_keys:
            keys_pressed.extend([(key[0], key[1]) for key in all_keys])
        # python is 0-indexed, so add 1 to i in order to determine which trial
        # we're on
        if check_for_keys(all_keys, ['return', 'space']) or (take_break and i+1 == break_time):
            timings.append(('pause', 'start', expt_clock.getTime()))
            if take_break and i == break_time:
                break_text = visual.TextStim(win, "Break time!", height=text_height)
                break_text.draw()
                win.flip()
                core.wait(2)
            paused_keys = pause(i+1, len(left_stimuli), win, expt_clock)
            timings.append(('pause', 'stop', expt_clock.getTime()))
            keys_pressed.extend(paused_keys)
            if not check_for_keys(paused_keys):
                countdown(win, text_height)
        else:
            paused_keys = []
        if not check_for_keys(all_keys+paused_keys):
            timings.append(('post-stimulus_%d' % (i+start_from_stim), 'on', expt_clock.getTime()))
            # keys are stored as strs
            if keys_pressed[-1][0] == str(correct_responses[i]):
                fixation['correct'].draw()
            else:
                fixation['default'].draw()
            win.flip()
            core.wait(off_msec_length[1] / 1000)
        save(save_path, stimuli_path, idx_path, keys_pressed, timings, expt_params, idx,
             screen=screen, screen_size_deg=screen_size_deg,
             last_trial=i+start_from_stim, **monitor_kwargs)
        if check_for_keys(all_keys+paused_keys):
            break
    all_keys = _end_run(win, timings, text_height, expt_clock,
                        train_flag)
    if all_keys:
        keys_pressed.extend([(key[0], key[1]) for key in all_keys])
    return keys_pressed, timings, expt_params, idx, win


def expt(stimuli_path, subj_name, sess_num, run_num, comparison,
         output_dir="data/raw_behavioral", screen_size_pix=[3840, 2160],
         screen_size_deg=73.45, take_break=True, text_height=50, screen=[0],
         train_flag=False, **kwargs):
    """run a full experiment

    this just sets up the various paths, calls ``run``, and then saves
    the output

    """
    model_name = re.findall("/((?:RGC|V1|training).*?)/", stimuli_path)[0]
    if not (model_name.startswith('RGC') or model_name.startswith('V1') or model_name.startswith('training')):
        raise Exception(f"Can't find model_name from stimuli_path {stimuli_path}! "
                        f"Found {model_name} when trying to do so")
    output_dir = op.join(output_dir, model_name, f'task-split_comp-{comparison}', subj_name)
    if not op.exists(op.join(output_dir)):
        os.makedirs(op.join(output_dir))
    kwargs_str = ""
    for k, v in kwargs.items():
        kwargs_str += "_{}-{}".format(k, v)
    save_path = op.join(output_dir, "%s_%s_task-split_comp-%s_sess-{sess:02d}_run-{run:02d}%s.hdf5" %
                        (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name, comparison, kwargs_str))
    idx_path = op.join(op.dirname(stimuli_path), f'task-split_comp-{comparison}', subj_name,
                       f'{subj_name}_task-split_comp-{comparison}_idx_sess-{sess_num:02d}_run-{run_num:02d}.npy')
    save_path = save_path.format(sess=sess_num, run=run_num)
    # don't want to load in existing save data for training subject
    if not 'training' in subj_name and os.path.isfile(save_path):
        print("Existing save data %s found! Will load in and append results" % save_path)
        f = h5py.File(save_path)
        saved_stim_path = f['stimuli_path'][()].decode()
        if stimuli_path != saved_stim_path:
            raise Exception("Stimuli path not same in existing save data and this run! "
                            "from save: %s, this run: %s" % (saved_stim_path, stimuli_path))
        saved_idx_path = f['idx_path'][()].decode()
        if idx_path != saved_idx_path:
            raise Exception("Index path not same in existing save data and this run! "
                            "from save: %s, this run: %s" % (saved_idx_path, idx_path))
        keys = list(f.pop('button_presses')[()])
        timings = list(f.pop('timing_data')[()])
        try:
            start_from_stim = f['last_trial'][()]
        except KeyError:
            # in this case, the previous one was quit before it started
            start_from_stim = 0
        f.close()
    else:
        keys = []
        timings = []
        start_from_stim = 0
    if not os.path.isfile(idx_path):
        raise IOError("Unable to find array of stimulus indices %s!" % idx_path)
    if subj_name not in idx_path:
        raise Exception("subj_name %s should be in idx_path %s, are you sure they correspond?" %
                        (subj_name, idx_path))
    print("Will use the following stimulus:")
    print("\t%s" % stimuli_path)
    print("Will use the following index:")
    print("\t%s" % idx_path)
    print("Will save at the following location:\n\t%s" % save_path)
    keys, timings, expt_params, idx, win = run_split(
        stimuli_path, idx_path, save_path, comparison,
        size=screen_size_pix, take_break=take_break,
        screen_size_deg=screen_size_deg, start_from_stim=start_from_stim,
        keys_pressed=keys, timings=timings,
        text_height=text_height, screen=screen, train_flag=train_flag,
        **kwargs)
    save(save_path, stimuli_path, idx_path, keys, timings, expt_params, idx, **kwargs)
    if train_flag:
        stim_df = pd.read_csv(stimuli_path.replace('stimuli_', 'stimuli_description_').replace('.npy', '.csv'))
        pct_correct = calc_pct_correct(save_path, idx, stim_df)
        if len(pct_correct) == 1:
            pct_correct = f'{int(pct_correct.iloc[0].pct_correct * 100)}%'
        else:
            # once we've sorted by scaling, the first and last entries will be
            # the lowest scaling / hardest and highest scaling / easiest,
            # respectively
            pct_correct = pct_correct.sort_values('scaling')
            pct_correct = (f'\nEasy trials: {int(pct_correct.iloc[-1].pct_correct*100)}%'
                           f'\nHard trials: {int(pct_correct.iloc[0].pct_correct*100)}%')
        visual.TextStim(win, f"Percent correct: {pct_correct}\n\nPress space to finish.",
                        height=text_height, wrapWidth=2000).draw()
        win.flip()
        all_keys = event.waitKeys(keyList=['return', 'space', 'q', 'escape', 'esc'])
        clear_events(win)
        win.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run a psychophysical experiment to investigate metamers! Specify the location of the "
                     "stimuli, the session number, image set, and the subject name, and we'll handle the "
                     "rest. Structure of experiment depends on comparison."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stimuli_path", help="Path to your unshuffled stimuli.")
    parser.add_argument("subj_name", help="Name of the subject")
    parser.add_argument("sess_num", help=("Session number"), type=int)
    parser.add_argument('-r', "--run_num",  type=int, default=None, nargs='+',
                        help=("Run number. If unset, we do runs 0 through 4 "
                              "(inclusive), one after the other."))
    parser.add_argument("--output_dir", '-o', help="directory to place output in",
                        default=op.expanduser("~/Desktop/metamers/raw_behavioral"))
    parser.add_argument("--screen", '-s', default=[0], type=int, nargs=1,
                        help=("Screen number to display experiment on."))
    parser.add_argument("--screen_size_pix", '-p', nargs=2, help="Size of the screen (in pixels)",
                        default=[3840, 2160], type=float)
    parser.add_argument("--screen_size_deg", '-d', default=73.45, type=float,
                        help="Size of longest screen side (in degrees)")
    parser.add_argument('--no_break', '-n', action='store_true',
                        help=("If passed, we do not take a break at the half-way point"))
    parser.add_argument("--comparison", '-c', default='ref',
                        help=("{ref, met}. Whether this run is comparing metamers against "
                              "reference images or other metamers."))
    args = vars(parser.parse_args())
    runs = args.pop('run_num')
    if runs is None:
        if 'training' not in args['subj_name']:
            runs = range(5)
        else:
            runs = range(1)
    elif not hasattr(runs, '__iter__'):
        runs = [runs]
    take_break = not args.pop('no_break')
    if 'training_noise' in args['stimuli_path']:
        train_flag = 'noise'
    elif 'training' in args['stimuli_path']:
        train_flag = 'model'
    else:
        train_flag = False
    print(f"Running {len(runs)} runs.\n")
    for run in runs:
        expt(run_num=run, take_break=take_break, train_flag=train_flag, **args)
        print()
