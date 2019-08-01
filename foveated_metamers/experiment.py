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


def _set_params(stim_path, idx_path, size=[1920, 1080], monitor='CBI-prisma-projector',
                units='pix', fullscr=True, screen=0, color=128, colorSpace='rgb255',
                **monitor_kwargs):
    """set the various experiment parameters
    """
    stimuli = np.load(stim_path)
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


def run(stim_path, idx_path, on_msec_length=200, off_msec_length=(500, 1000, 2000),
        fix_deg_size=.25, screen_size_deg=60, eyetracker=None, edf_path=None, save_frames=None,
        **monitor_kwargs):
    """run one run of the experiment

    stim_path specifies the path of the unshuffled experiment stimuli, while idx_path specifies the
    path of the shuffled indices to use for this run. This function will load in the stimuli at
    stim_path and rearrange them using the indices found at idx_path, then simply go through those
    stimuli in order, showing each stimuli for `on_msec_length` msecs and then a blank screen for
    `off_msec_length` msecs (or as close as possible, given the monitor's refresh rate).

    For fixation, we show a stream of digits whose colors alternate between black and white, with a
    `fix_button_prob` chance of repeating. Digits are presented with alternating stimuli ON and OFF
    blocks, so that a digit will be shown for on_msec_length+off_msec_length msecs and then there
    will be nothing at fixation for the next on_msec_length+off_msec_length msecs. For now, you
    can't change this.

    All stimuli loaded in from stim_path will be shown.


    Arguments
    ============

    stim_path: string, path to .npy file where stimuli are stored (as 3d array)

    idx_path: string, path to .npy file where shuffled indices are stored (as 2d array)

    on_msec_length: int, length of the ON blocks in milliseconds; that is, the length of time to
    display each stimulus before moving on

    off_msec_length: int, length of the OFF blocks in milliseconds; that is, the length of time to
    between stimuli

    fix_pix_size: int, the size of the fixation digits, in pixels.

    fix_button_prob: float. the probability that the fixation digit will repeat or the fixation dot
    will change color (will never repeat more than once in a row). For fixation digit, this
    probability is relative to each stimulus presentation / ON block starting; for fixation dot,
    it's each stimulus change (stimulus ON or OFF block starting).

    eyetracker: EyeLink object or None. if None, will not collect eyetracking data. if not None,
    will gather it. the EyeLink object must already be initialized (by calling the _setup_eyelink
    function, as is done in the expt function). if this is set, must also specify edf_path

    edf_path: str or None. if eyetracker is not None, this must be a string, which is where we
    will save the output of the eyetracker

    screen_size_deg: int or float. the max visual angle (in degrees) of the full screen.

    save_frames: None or str. if not None, this should be the filename you wish to save frames at
    (one image will be made for each frame). WARNING: typically a large number of files will be
    saved (depends on the length of your session), which means this may make the end of the run
    (with the screen completely blank) take a while
    """
    stimuli, idx, expt_params, monitor_kwargs = _set_params(stim_path, idx_path, **monitor_kwargs)

    win = visual.Window(winType='glfw', **monitor_kwargs)
    win.mouseVisible = False
    # linear gamma ramp
    win.gammaRamp = np.tile(np.linspace(0, 1, 256), (3, 1))

    fix_pix_size = fix_deg_size * (monitor_kwargs['size'][0] / screen_size_deg)
    fixation = visual.GratingStim(win, size=fix_pix_size, pos=[0, 0], sf=0, color='red',
                                  mask='circle')
    # first one is special: we preload it, but we still want to include it in the iterator so the
    # numbers all match up (we don't draw or wait during the on part of the first iteration)
    img = visual.ImageStim(win, image=imagetools.array2image(stimuli[0, 0]),
                           size=expt_params['stimuli_size'])

    if eyetracker is not None:
        assert edf_path is not None, "edf_path must be set so we can save the eyetracker output!"
        eyetracker.openDataFile('temp.EDF')
        pylink.flushGetkeyQueue()
        eyetracker.startRecording(1, 1, 1, 1)

    clock = core.Clock()
    wait_text = visual.TextStim(win, ("Press 5 to start\nq or escape will quit"))
    query_text = visual.TextStim(win, "1 or 2?")
    wait_text.draw()
    win.flip()

    # we should be able to use waitKeys for this, but something weird
    # has happened, where we don't record those button presses for some
    # reason, so instead we do this while loop with a win.flip() and
    # core.wait() (the issue seems to be that we only grab keys
    # successfully pretty quickly after a win.flip()?)
    # all_keys = event.waitKeys(keyList=['5', 'q', 'escape', 'esc'], timeStamped=clock)
    all_keys = []
    while not all_keys:
        wait_text.draw()
        win.flip()
        core.wait(.1)
        all_keys = event.getKeys(keyList=['5', 'q', 'escape', 'esc'], timeStamped=clock)
    if save_frames is not None:
        win.getMovieFrame()

    # wait until receive 5, which is the scanner trigger
    if ('q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys] or
        'esc' in [k[0] for k in all_keys]):
        win.close()
        return all_keys, [], expt_params, idx

    # keys_pressed = [(key[0], key[1]) for key in all_keys]
    keys_pressed = []
    timings = [("start", "off", clock.getTime())]
    # this outer for loop is per trial
    for i, stim in enumerate(stimuli):
        # and this one is for the three stimuli in each trial
        for j, s in enumerate(stim):
            img.image = imagetools.array2image(s)
            img.draw()
            fixation.draw()
            win.flip()
            timings.append(("stimulus_%d-%d" % (i, j), "on", clock.getTime()))
            next_stim_time = ((i*3*on_msec_length + (j+1)*on_msec_length +
                               i*np.sum(off_msec_length) + np.sum(off_msec_length[:j]) - 2)
                              / 1000.)
            core.wait(abs(clock.getTime() - timings[0][2] - next_stim_time))
            if eyetracker is not None:
                eyetracker.sendMessage("TRIALID %02d" % i)
            if save_frames is not None:
                win.getMovieFrame()
            if j == 2:
                query_text.draw()
            else:
                fixation.draw()
            win.flip()
            timings.append(("stimulus_%d-%d" % (i, j), "off", clock.getTime()))
            next_stim_time = ((i*3*on_msec_length + (j+1)*on_msec_length +
                               i*np.sum(off_msec_length) + np.sum(off_msec_length[:j+1]) - 1)
                              / 1000.)
            core.wait(abs(clock.getTime() - timings[0][2] - next_stim_time))
            if save_frames is not None:
                win.getMovieFrame()
            all_keys = event.getKeys(timeStamped=clock)
            if all_keys:
                keys_pressed.extend([(key[0], key[1]) for key in all_keys])
            # we need this double break because we have two for loops
            if ('q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys] or
                'esc' in [k[0] for k in all_keys]):
                break
        if ('q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys] or
            'esc' in [k[0] for k in all_keys]):
            break
    visual.TextStim(win, "Run over").draw()
    win.flip()
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
        win.saveMovieFrames(save_frames)
    win.close()
    return keys_pressed, timings, expt_params, idx


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


def expt(stimuli_path, subj_name, idx_path, output_dir="data/raw_behavioral", eyetrack=False,
         screen_size_pix=[1920, 1080], screen_size_deg=60, **kwargs):
    """run a full experiment

    this just loops through the specified stims_path, passing each one to the run function in
    turn. any other kwargs are sent directly to run as well. it then saves the returned
    keys_pressed and frame intervals
    """
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    file_path = op.join(output_dir, "%s_%s_sess{sess:02d}.hdf5" %
                        (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name))
    edf_path = op.join(output_dir, "%s_%s_sess{sess:02d}.EDF" %
                       (datetime.datetime.now().strftime("%Y-%b-%d"), subj_name))
    sess_num = 0
    while glob.glob(file_path.format(sess=sess_num)):
        sess_num += 1
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
    print("Will save at the following location:\n\t%s" % file_path.format(sess=sess_num))
    keys, timings, expt_params, idx = run(stimuli_path, idx_path, size=screen_size_pix,
                                          eyetracker=eyetracker,
                                          screen_size_deg=screen_size_deg,
                                          edf_path=edf_path.format(sess=sess_num),
                                          **kwargs)
    with h5py.File(file_path.format(sess=sess_num), 'a') as f:
        f.create_dataset("button_presses", data=_convert_str(keys))
        f.create_dataset("timing_data", data=_convert_str(timings))
        f.create_dataset("stim_path", data=stimuli_path.encode())
        f.create_dataset("idx_path", data=idx_path.encode())
        f.create_dataset("shuffled_indices", data=idx)
        for k, v in expt_params.items():
            f.create_dataset("%s" % k, data=v)
        # also note differences from default options
        for k, v in kwargs.items():
            if v is None:
                f.create_dataset("%s" % k, data=str(v))
            else:
                f.create_dataset("%s" % k, data=v)
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
    parser.add_argument("--screen", '-s', default=0, type=int,
                        help=("Screen number to display experiment on"))
    parser.add_argument("--screen_size_pix", '-p', nargs=2, help="Size of the screen (in pixels)",
                        default=[1920, 1080], type=float)
    parser.add_argument("--screen_size_deg", '-d', default=60, type=float,
                        help="Size of longest screen side (in degrees)")
    args = vars(parser.parse_args())
    expt(**args)
