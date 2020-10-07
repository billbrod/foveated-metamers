#!/usr/bin/python
"""psychopy script for ABX experiment, run from the command line
"""

import argparse
import h5py
import os
import os.path as op
import datetime
import warnings
import numpy as np
import pandas as pd
from ipd_calibration import csv_to_binocular_offset, clear_events
from psychopy import visual, core, event, clock
from psychopy.tools import imagetools
try:
    import pylink
except ImportError:
    warnings.warn("Unable to find pylink, will not be able to collect eye-tracking data")


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


def _setup_eyelink(win_size):
    """set up the eyelink eye-tracking
    """

    # Connect to eyelink
    eyetracker = pylink.EyeLink('192.168.1.5')
    pylink.openGraphics()

    # Set content of edf file
    eyetracker.sendCommand('link_sample_data=LEFT,RIGHT,GAZE,AREA')
    eyetracker.sendCommand('file_sample_data=LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS')
    eyetracker.sendCommand('file_event_filter=LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON')

    # Set coords
    eyetracker.sendCommand('screen_pixel_coords=0 0 {} {}'.format(*win_size))
    eyetracker.sendMessage('DISPLAY_COORDS 0 0 {} {}'.format(*win_size))

    # Calibrate
    eyetracker.setCalibrationType('HV5')
    eyetracker.doTrackerSetup(win_size)
    pylink.closeGraphics()

    return eyetracker


def _set_params(stimuli_path, idx_path, size=[1920, 1080], monitor='CBI-prisma-projector',
                units='pix', fullscr=True, screen=0, color=128, colorSpace='rgb255',
                allowGUI=False, **monitor_kwargs):
    """set the various experiment parameters
    """
    stimuli = np.load(stimuli_path)
    idx = np.load(idx_path)
    # we do some nifty indexing here. idx will be a 2d array, i x 3
    # (where i is the number of trials we want), so when we do the
    # following, we go from n x *img_size array (where n is the max
    # number of images) to i x 3 x *img_size array
    stimuli = stimuli[idx, :, :]
    expt_params = {}
    # the first dimension of stimuli (retrieved by len) is how many
    # trials we have. the next one shows the number of stimuli per
    # trial (3), and the final two are the size of the stimuli
    expt_params['trial_num'] = stimuli.shape[0]
    expt_params['stimuli_per_trial'] = stimuli.shape[1]
    expt_params['stimuli_size'] = stimuli.shape[2::]
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


def pause(current_i, total_imgs, win, img_pos, expt_clock, flip_text=True):
    pause_text = [visual.TextStim(w, f"{current_i}/{total_imgs}\nspace to resume\nq or esc to quit",
                                  pos=p, flipHoriz=flip_text) for w, p in zip(win, img_pos)]
    all_keys = []
    while not all_keys:
        [text.draw() for text in pause_text]
        [w.flip() for w in win]
        core.wait(.1)
        all_keys = event.getKeys(keyList=['space', 'q', 'escape', 'esc'], timeStamped=expt_clock)
    clear_events(win)
    return [(key[0], key[1]) for key in all_keys]


def run(stimuli_path, idx_path, save_path, on_msec_length=200, off_msec_length=(500, 1000, 2000),
        fix_deg_size=.25, screen_size_deg=60, eyetracker=None, edf_path=None, save_frames=None,
        binocular_offset=[0, 0], take_break=True, keys_pressed=[], timings=[], start_from_stim=0,
        flip_text=True, **monitor_kwargs):
    """run one run of the experiment

    stimuli_path specifies the path of the unshuffled experiment
    stimuli, while idx_path specifies the path of the shuffled indices
    to use for this run. This function will load in the stimuli at
    stimuli_path and rearrange them using the indices found at idx_path,
    then simply go through those stimuli in order, showing each stimuli
    for ``on_msec_length`` msecs and then a blank screen for
    ``off_msec_length[i]`` msecs (or as close as possible, given the
    monitor's refresh rate; ``i`` depends on which of the stimulus just
    shown was A, B, or X in our ABX design).

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
    on_msec_length : int
        length of the ON blocks in milliseconds; that is, the length of
        time to display each stimulus
    off_msec_length : tuple
        3-tuple of ints specifying the length of the length of the OFF
        blocks in milliseconds. This is an ABX experiment, so the 3 ints
        correspond to the number of milliseconds between A and B, B and
        X, and X and the A of the next trial
    fix_deg_size : int
        the size of the fixation digits, in degrees.
    eyetracker : EyeLink object or None
        if None, will not collect eyetracking data. if not None, will
        gather it. the EyeLink object must already be initialized (by
        calling the _setup_eyelink function, as is done in the expt
        function). if this is set, must also specify edf_path
    edf_path : str or None
        if eyetracker is not None, this must be a string, which is where
        we will save the output of the eyetracker
    screen_size_deg : int or float.
        the max visual angle (in degrees) of the full screen.
    save_frames : None or str
        if not None, this should be the filename you wish to save frames
        at (one image will be made for each frame). WARNING: typically a
        large number of files will be saved (depends on the length of
        your session), which means this may make the end of the run
        (with the screen completely blank) take a while
    binocular_offset : list
        list of 2 ints, specifying the horizontal, vertical offset
        between the stimuli (in pixels) presented on the two monitors in
        order to allow the user to successfully fuse the image. This
        should come from the calibration, run before this experiment.
    flip_text : bool
        Whether to flip the text horizontally or not

    """
    stimuli, idx, expt_params, monitor_kwargs = _set_params(stimuli_path, idx_path,
                                                            **monitor_kwargs)
    stimuli = stimuli[start_from_stim:]
    print("Starting from stimulus %s" % start_from_stim)

    if len(monitor_kwargs['screen']) == 1:
        screen = monitor_kwargs.pop('screen')[0]
        print('Doing single-monitor mode on screen %s' % screen)
        win = [visual.Window(winType='glfw', screen=screen, **monitor_kwargs)]
        img_pos = [(0, 0)]
    elif len(monitor_kwargs['screen']) == 2:
        screen = monitor_kwargs.pop('screen')
        # want these to be in increasing order
        screen.sort()
        print("Doing binocular mode on screens %s" % screen)
        img_pos = [[int(-o // 2) for o in binocular_offset],
                   [int(o // 2) for o in binocular_offset]]
        print("Using binocular offsets: %s" % img_pos)
        win = [visual.Window(winType='glfw', screen=screen[0], swapInterval=1, **monitor_kwargs)]
        # see here for the explanation of swapInterval and share args
        # (basically, in order to make glfw correctly update the two
        # monitors together):
        # https://discourse.psychopy.org/t/strange-behavior-with-retina-displays-external-monitors-in-1-90-2-py2/5485/5
        win.append(visual.Window(winType='glfw', screen=screen[1], swapInterval=0, share=win[0],
                                 **monitor_kwargs))
    else:
        raise Exception("Can't handle %s screens!" % len(monitor_kwargs['screen']))

    break_time = len(stimuli) // 2
    if take_break:
        print("%s total trials, will take break after number %s" % (len(stimuli), break_time))

    fix_pix_size = fix_deg_size * (monitor_kwargs['size'][0] / screen_size_deg)
    fixation = [visual.GratingStim(w, size=fix_pix_size, pos=p, sf=0, color='red',
                                   mask='circle') for w, p in zip(win, img_pos)]
    # first one is special: we preload it, but we still want to include it in the iterator so the
    # numbers all match up (we don't draw or wait during the on part of the first iteration)
    img = [visual.ImageStim(w, image=imagetools.array2image(stimuli[0, 0]), pos=p,
                            size=expt_params['stimuli_size']) for w, p in zip(win, img_pos)]

    if eyetracker is not None:
        assert edf_path is not None, "edf_path must be set so we can save the eyetracker output!"
        eyetracker.openDataFile('temp.EDF')
        pylink.flushGetkeyQueue()
        eyetracker.startRecording(1, 1, 1, 1)

    timer = clock.StaticPeriod(screenHz=60)
    expt_clock = clock.Clock()
    if timings:
        # if we've been passed something, make sure everything happens
        # after it. for some reason, need to use the negative value of
        # it in order to add that to the expt_clock...
        expt_clock.reset(-float(timings[-1][-1]))
    wait_text = [visual.TextStim(w, ("Press space to start\nq or esc will quit\nspace to pause"),
                                 pos=p, flipHoriz=flip_text) for w, p in zip(win, img_pos)]
    query_text = [visual.TextStim(w, "Same as 1 or 2?", pos=p, flipHoriz=flip_text)
                  for w, p in zip(win, img_pos)]
    [text.draw() for text in wait_text]
    [w.flip() for w in win]

    # we should be able to use waitKeys for this, but something weird
    # has happened, where we don't record those button presses for some
    # reason, so instead we do this while loop with a win.flip() and
    # core.wait() (the issue seems to be that we only grab keys
    # successfully pretty quickly after a win.flip()?)
    # all_keys = event.waitKeys(keyList=['space', 'q', 'escape', 'esc'], timeStamped=expt_clock)
    all_keys = []
    while not all_keys:
        [text.draw() for text in wait_text]
        [w.flip() for w in win]
        core.wait(.1)
        all_keys = event.getKeys(keyList=['space', 'q', 'escape', 'esc'], timeStamped=expt_clock)
    clear_events(win)
    if save_frames is not None:
        [w.getMovieFrame() for w in win]

    if check_for_keys(all_keys):
        [w.close() for w in win]
        return all_keys, [], expt_params, idx

    timings.append(("start", "off", expt_clock.getTime()))

    # this outer for loop is per trial
    for i, stim in enumerate(stimuli):
        # and this one is for the three stimuli in each trial
        all_keys = []
        for j in range(len(stim)):
            for im, f in zip(img, fixation):
                im.draw()
                f.draw()
            for w in win:
                w.flip()
            timings.append(("stimulus_%d-%d" % (i+start_from_stim, j), "on", expt_clock.getTime()))
            # convert to sec
            core.wait(on_msec_length / 1000)
            if eyetracker is not None:
                eyetracker.sendMessage("TRIALID %02d" % i)
            if save_frames is not None:
                [w.getMovieFrame() for w in win]
            if j == 2:
                [q.draw() for q in query_text]
            else:
                [f.draw() for f in fixation]
            [w.flip() for w in win]
            timings.append(("stimulus_%d-%d" % (i+start_from_stim, j), "off",
                            expt_clock.getTime()))
            timer.start(off_msec_length[j] / 1000)
            for im in img:
                # off msec lengths are always longer than on msec length, so
                # we preload the next image here
                try:
                    # we either load the next one in the set of three for
                    # this trial...
                    im.image = imagetools.array2image(stim[j+1])
                except IndexError:
                    # or, if we've gone through all those, we load the first
                    # image for the next trial. if i+1==len(stimuli), then we've
                    # gone through all images and will quit out
                    if i+1 < len(stimuli):
                        im.image = imagetools.array2image(stimuli[i+1][0])
            if save_frames is not None:
                [w.getMovieFrame() for w in win]
            timer.complete()
            all_keys.extend(event.getKeys(timeStamped=expt_clock))
            # we need this double break because we have two for loops
            if check_for_keys(all_keys):
                break
        if all_keys:
            keys_pressed.extend([(key[0], key[1]) for key in all_keys])
        # python is 0-indexed, so add 1 to i in order to determine which trial
        # we're on
        if check_for_keys(all_keys, ['space']) or (take_break and i+1 == break_time):
            timings.append(('pause', 'start', expt_clock.getTime()))
            if take_break and i == break_time:
                break_text = [visual.TextStim(w, "Break time!", pos=p, flipHoriz=flip_text)
                              for w, p in zip(win, img_pos)]
                [text.draw() for text in break_text]
                [w.flip() for w in win]
                core.wait(2)
            paused_keys = pause(i+1, len(stimuli), win, img_pos, expt_clock, flip_text)
            timings.append(('pause', 'stop', expt_clock.getTime()))
            keys_pressed.extend(paused_keys)
        else:
            paused_keys = []
        save(save_path, stimuli_path, idx_path, keys_pressed, timings, expt_params, idx,
             screen=screen, edf_path=edf_path, screen_size_deg=screen_size_deg,
             last_trial=i+start_from_stim, **monitor_kwargs)
        if check_for_keys(all_keys+paused_keys):
            break
    [visual.TextStim(w, "Run over", pos=p, flipHoriz=flip_text).draw() for w, p in zip(win, img_pos)]
    [w.flip() for w in win]
    timings.append(("run_end", '', expt_clock.getTime()))
    all_keys = event.getKeys(timeStamped=expt_clock)
    if all_keys:
        keys_pressed.extend([(key[0], key[1]) for key in all_keys])
    core.wait(4)
    if eyetracker is not None:
        eyetracker.stopRecording()
        eyetracker.closeDataFile()
        eyetracker.receiveDataFile('temp.EDF', edf_path)
    if save_frames is not None:
        [w.saveMovieFrames(save_frames) for w in win]
    [w.close() for w in win]
    return keys_pressed, timings, expt_params, idx


def expt(stimuli_path, subj_name, sess_num, im_num, output_dir="data/raw_behavioral", eyetrack=False,
         screen_size_pix=[1920, 1080], screen_size_deg=60, take_break=True, ipd_csv=None,
         flip_text=True, **kwargs):
    """run a full experiment

    this just sets up the various paths, calls ``run``, and then saves
    the output

    """
    if ipd_csv is not None:
        binocular_offset = csv_to_binocular_offset(ipd_csv, subj_name)
    else:
        binocular_offset = [0, 0]
    model_name = op.split(op.dirname(stimuli_path))[-1]
    if not op.exists(op.join(output_dir, model_name)):
        os.makedirs(op.join(output_dir, model_name))
    save_path = op.join(output_dir, model_name, "%s_%s_sess-{sess:02d}_im-{im:02d}.hdf5" %
                        (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name))
    edf_path = op.join(output_dir, model_name, "%s_%s_sess-{sess:02d}_im-{im:02d}.EDF" %
                       (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name))
    idx_path = stimuli_path.replace('stimuli.npy', '%s_idx_sess-%02d_im-%02d.npy' % (subj_name, sess_num, im_num))
    save_path = save_path.format(sess=sess_num, im=im_num)
    if os.path.isfile(save_path):
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
    if eyetrack:
        eyetracker = _setup_eyelink(screen_size_pix)
    else:
        eyetracker = None
        # we pass through the same edf_path even if we're not using the eyetracker because it
        # doesn't get used (and if set this to None or something, then the edf_path.format call
        # several lines down will fail)
    print("Running 1 run, with the following stimulus:")
    print("\t%s" % stimuli_path)
    print("Will use the following index:")
    print("\t%s" % idx_path)
    print("Will save at the following location:\n\t%s" % save_path)
    keys, timings, expt_params, idx = run(stimuli_path, idx_path, save_path, size=screen_size_pix,
                                          eyetracker=eyetracker, take_break=take_break,
                                          screen_size_deg=screen_size_deg,
                                          start_from_stim=start_from_stim, flip_text=flip_text,
                                          binocular_offset=binocular_offset,
                                          edf_path=edf_path.format(sess=sess_num, im=im_num),
                                          keys_pressed=keys, timings=timings, **kwargs)
    save(save_path, stimuli_path, idx_path, keys, timings, expt_params, idx, **kwargs)
    if eyetracker is not None:
        eyetracker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run an ABX experiment to investigate metamers! Specify the location of the "
                     "stimuli, the location of the (already-computed and randomized) indices, and"
                     " the subject name, and we'll handle the rest. Each trial will consist of "
                     "three stimuli, shown briefly, with blank screens in between, with a pause at"
                     " the end of the trial, at which point they must specify whether the third "
                     "stimulus was identical to the first or the second. This continues until we'"
                     "ve gone through all the trials in the index array, at which point we save "
                     "responses, stimulus timing, and exit out."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stimuli_path", help="Path to your unshuffled stimuli.")
    parser.add_argument("subj_name", help="Name of the subject")
    parser.add_argument("sess_num", help=("Session number"), type=int)
    parser.add_argument("im_num", help=("Image set number"), type=int)
    parser.add_argument("--ipd_csv", '-i', help="Path to the csv containing ipd correction info",
                        default=op.expanduser('~/Desktop/metamers/ipd/ipd_correction.csv'))
    parser.add_argument("--output_dir", '-o', help="directory to place output in",
                        default=op.expanduser("~/Desktop/metamers/raw_behavioral"))
    parser.add_argument("--eyetrack", '-e', action="store_true",
                        help=("Pass this flag to tell the script to gather eye-tracking data. If"
                              " pylink is not installed, this is impossible and will throw an "
                              "exception"))
    parser.add_argument("--screen", '-s', default=[1, 2], type=int, nargs='+',
                        help=("Screen number to display experiment on"))
    parser.add_argument("--screen_size_pix", '-p', nargs=2, help="Size of the screen (in pixels)",
                        default=[4096, 2160], type=float)
    parser.add_argument("--screen_size_deg", '-d', default=95, type=float,
                        help="Size of longest screen side (in degrees)")
    parser.add_argument('--no_break', '-n', action='store_true',
                        help=("If passed, we do not take a break at the half-way point"))
    parser.add_argument("--no_flip", '-f', action='store_true',
                        help=("This script is meant to be run on the haploscope. Therefore, we "
                              "left-right flip all text by default. Use this option to disable"
                              " that"))
    args = vars(parser.parse_args())
    take_break = not args.pop('no_break')
    flip = not args.pop('no_flip')
    ipd_csv = args.pop('ipd_csv')
    if op.exists(ipd_csv):
        ipd_csv = pd.read_csv(ipd_csv)
    else:
        warnings.warn("Can't find ipd_csv, using zero binocular offset!")
        ipd_csv = None
    expt(ipd_csv=ipd_csv, take_break=take_break, flip_text=flip, **args)
