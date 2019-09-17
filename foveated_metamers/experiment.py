#!/usr/bin/python
"""psychopy script for ABX experiment, run from the command line
"""

import argparse
import h5py
import glob
import os
import os.path as op
import datetime
import warnings
import numpy as np
from psychopy import visual, core, event
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
            # then this is not a string but another list of strings
            x = [i.encode() for i in x]
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
    except RuntimeError:
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
                **monitor_kwargs):
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
    expt_params['stimuli_size'] = stimuli.shape[2:]

    # these are all a variety of kwargs used by monitor
    monitor_kwargs.update({'size': size, 'monitor': monitor, 'units': units, 'fullscr': fullscr,
                           'screen': screen, 'color': color, 'colorSpace': colorSpace})
    return stimuli, idx, expt_params, monitor_kwargs


def run(stimuli_path, idx_path, save_path, on_msec_length=200, off_msec_length=(500, 1000, 2000),
        fix_deg_size=.25, screen_size_deg=60, eyetracker=None, edf_path=None, save_frames=None,
        binocular_offset=[0, 0], **monitor_kwargs):
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

    All stimuli loaded in from stimuli_path will be shown.

    Arguments
    ============
    stimuli_path: string
        path to .npy file where stimuli are stored (as 3d array)
    idx_path: string
        path to .npy file where shuffled indices are stored (as 2d
        array)
    save_path: string
        path to .hdf5 file where we'll store the outputs of this
        experiment (we save every trial)
    on_msec_length: int
        length of the ON blocks in milliseconds; that is, the length of
        time to display each stimulus
    off_msec_length: tuple
        3-tuple of ints specifying the length of the length of the OFF
        blocks in milliseconds. This is an ABX experiment, so the 3 ints
        correspond to the number of milliseconds between A and B, B and
        X, and X and the A of the next trial
    fix_deg_size: int
        the size of the fixation digits, in degrees.
    eyetracker: EyeLink object or None
        if None, will not collect eyetracking data. if not None, will
        gather it. the EyeLink object must already be initialized (by
        calling the _setup_eyelink function, as is done in the expt
        function). if this is set, must also specify edf_path
    edf_path: str or None
        if eyetracker is not None, this must be a string, which is where
        we will save the output of the eyetracker
    screen_size_deg: int or float.
        the max visual angle (in degrees) of the full screen.
    save_frames: None or str
        if not None, this should be the filename you wish to save frames
        at (one image will be made for each frame). WARNING: typically a
        large number of files will be saved (depends on the length of
        your session), which means this may make the end of the run
        (with the screen completely blank) take a while
    binocular_offset: list
        list of 2 ints, specifying the horizontal, vertical offset
        between the stimuli (in pixels) presented on the two monitors in
        order to allow the user to successfully fuse the image. This
        should come from the calibration, run before this experiment.

    """
    stimuli, idx, expt_params, monitor_kwargs = _set_params(stimuli_path, idx_path,
                                                            **monitor_kwargs)

    if len(monitor_kwargs['screen']) == 1:
        screen = monitor_kwargs.pop('screen')[0]
        print('Doing single-monitor mode on screen %s' % screen)
        win = [visual.Window(winType='glfw', screen=screen, **monitor_kwargs)]
        img_pos = [(0, 0)]
    elif len(monitor_kwargs['screen']) == 2:
        screen = monitor_kwargs.pop('screen')
        print("Doing binocular mode on screens %s" % screen)
        img_pos = [[-o // 2 for o in binocular_offset], [o // 2 for o in binocular_offset]]
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
    for w in win:
        w.mouseVisible = False
        # linear gamma ramp
        w.gammaRamp = np.tile(np.linspace(0, 1, 256), (3, 1))

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

    clock = core.Clock()
    wait_text = [visual.TextStim(w, ("Press 5 to start\nq or escape will quit"), pos=p)
                 for w, p in zip(win, img_pos)]
    query_text = [visual.TextStim(w, "1 or 2?", pos=p) for w, p in zip(win, img_pos)]
    [text.draw() for text in wait_text]
    [w.flip() for w in win]

    # we should be able to use waitKeys for this, but something weird
    # has happened, where we don't record those button presses for some
    # reason, so instead we do this while loop with a win.flip() and
    # core.wait() (the issue seems to be that we only grab keys
    # successfully pretty quickly after a win.flip()?)
    # all_keys = event.waitKeys(keyList=['5', 'q', 'escape', 'esc'], timeStamped=clock)
    all_keys = []
    while not all_keys:
        [text.draw() for text in wait_text]
        [w.flip() for w in win]
        core.wait(.1)
        all_keys = event.getKeys(keyList=['5', 'q', 'escape', 'esc'], timeStamped=clock)
    if save_frames is not None:
        [w.getMovieFrame() for w in win]

    # wait until receive 5, which is the scanner trigger
    if ('q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys] or
        'esc' in [k[0] for k in all_keys]):
        [w.close() for w in win]
        return all_keys, [], expt_params, idx

    # keys_pressed = [(key[0], key[1]) for key in all_keys]
    keys_pressed = []
    timings = [("start", "off", clock.getTime())]
    # this outer for loop is per trial
    for i, stim in enumerate(stimuli):
        # and this one is for the three stimuli in each trial
        for j, s in enumerate(stim):
            for im, f, w in zip(img, fixation, win):
                im.image = imagetools.array2image(s)
                im.draw()
                f.draw()
                w.flip()
            timings.append(("stimulus_%d-%d" % (i, j), "on", clock.getTime()))
            next_stim_time = ((i*3*on_msec_length + (j+1)*on_msec_length +
                               i*np.sum(off_msec_length) + np.sum(off_msec_length[:j]) - 2)
                              / 1000.)
            core.wait(abs(clock.getTime() - timings[0][2] - next_stim_time))
            if eyetracker is not None:
                eyetracker.sendMessage("TRIALID %02d" % i)
            if save_frames is not None:
                [w.getMovieFrame() for w in win]
            if j == 2:
                [q.draw() for q in query_text]
            else:
                [f.draw() for f in fixation]
            [w.flip() for w in win]
            timings.append(("stimulus_%d-%d" % (i, j), "off", clock.getTime()))
            next_stim_time = ((i*3*on_msec_length + (j+1)*on_msec_length +
                               i*np.sum(off_msec_length) + np.sum(off_msec_length[:j+1]) - 1)
                              / 1000.)
            core.wait(abs(clock.getTime() - timings[0][2] - next_stim_time))
            if save_frames is not None:
                [w.getMovieFrame() for w in win]
            all_keys = event.getKeys(timeStamped=clock)
            if all_keys:
                keys_pressed.extend([(key[0], key[1]) for key in all_keys])
            # we need this double break because we have two for loops
            if ('q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys] or
                'esc' in [k[0] for k in all_keys]):
                break
        save(save_path, stimuli_path, idx_path, keys_pressed, timings, expt_params, idx,
             screen=screen, edf_path=edf_path, screen_size_deg=screen_size_deg, **monitor_kwargs)
        if ('q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys] or
            'esc' in [k[0] for k in all_keys]):
            break
    [visual.TextStim(w, "Run over", pos=p).draw() for w, p in zip(win, img_pos)]
    [w.flip() for w in win]
    timings.append(("run_end", '', clock.getTime()))
    all_keys = event.getKeys(timeStamped=clock)
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


def expt(stimuli_path, subj_name, idx_path, output_dir="data/raw_behavioral", eyetrack=False,
         screen_size_pix=[1920, 1080], screen_size_deg=60, **kwargs):
    """run a full experiment

    this just sets up the various paths, calls ``run``, and then saves
    the output

    """
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    save_path = op.join(output_dir, "%s_%s_sess{sess:02d}.hdf5" %
                        (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name))
    edf_path = op.join(output_dir, "%s_%s_sess{sess:02d}.EDF" %
                       (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name))
    sess_num = 0
    while glob.glob(save_path.format(sess=sess_num)):
        sess_num += 1
    save_path = save_path.format(sess=sess_num)
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
                                          eyetracker=eyetracker,
                                          screen_size_deg=screen_size_deg,
                                          edf_path=edf_path.format(sess=sess_num),
                                          **kwargs)
    save(save_path, stimuli_path, idx_path, keys, timings, expt_params, idx,
         **kwargs)
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
    parser.add_argument("idx_path", help=("Path to the shuffled presentation indices"))
    parser.add_argument("subj_name", help="Name of the subject")
    parser.add_argument("--output_dir", '-o', help="directory to place output in",
                        default=op.expanduser("~/Desktop/metamers/raw_behavioral"))
    parser.add_argument("--eyetrack", '-e', action="store_true",
                        help=("Pass this flag to tell the script to gather eye-tracking data. If"
                              " pylink is not installed, this is impossible and will throw an "
                              "exception"))
    parser.add_argument("--screen", '-s', default=0, type=int, nargs='+',
                        help=("Screen number to display experiment on"))
    parser.add_argument("--screen_size_pix", '-p', nargs=2, help="Size of the screen (in pixels)",
                        default=[1920, 1080], type=float)
    parser.add_argument("--screen_size_deg", '-d', default=60, type=float,
                        help="Size of longest screen side (in degrees)")
    args = vars(parser.parse_args())
    expt(**args)
