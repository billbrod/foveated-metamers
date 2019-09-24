#!/usr/bin/python
"""psychopy script for running IPD calibration, run from command-line
"""
import argparse
import os
import numpy as np
import pandas as pd
import os.path as op
from psychopy import visual, event, core


def csv_to_binocular_offset(ipd_csv, subject_name, units='pix'):
    """
    """
    if isinstance(ipd_csv, str):
        ipd_csv = pd.read_csv(ipd_csv)
    ipd_csv = ipd_csv.query("subject_name==@subject_name")
    binocular_offset = [ipd_csv["ipd_correction_%s_horizontal" % units].mean(),
                        ipd_csv["ipd_correction_%s_vertical" % units].mean()]
    return binocular_offset


def calc_monocular_convergence_angle(binocular_ipd, fixation_distance=42.):
    """calculate monocular convergence angle, using trig

    The monocular convergence angle is just ``arctan((binocular_ipd/2) /
    fixation_distance)`` (``binocular_ipd/2`` is the monocular ipd).

    This is a first-pass approximation, you should then use the
    ipd_calibration function in this script in order to allow the
    subject to perceptually line things up. This should get you close.

    All units need to be in cm. Default value for fixation_distance is
    for the FancyPants v1 haploscope used for the foveated metamers
    experiment.

    Parameters
    ---------
    binocular_ipd : float
        The subject's inter-pupillary distance (IPD), that is, the
        distance between the subject's eyes, in *cm*
    fixation_distance : float
        The fixation distance of the monitor, in cm. Default value is
        for the FancyPants v1 haploscope

    Returns
    -------
    monocular_verg_angle : float
        The monocular convergence angle, in degrees
    """
    monocular_ipd = binocular_ipd / 2
    # arctan returns the answer in radians, so convert to degrees
    verg_angle = np.arctan(monocular_ipd / fixation_distance) * (180 / np.pi)
    return verg_angle


def calc_pix_per_deg(fixation_distance=42., monitor_pix_width=4096, monitor_cm_width=69.8):
    """
    """
    # this follows from the trigonomery (picture an image of size 1cm,
    # fixation_distance away from an observer who's staring at its
    # center); this then gets us the number of degrees per cm. To get a
    # more detailed explanation for why this works, see
    # http://elvers.us/perception/visualAngle/va.html
    deg_per_cm = 2 * np.arctan(1 / (2*fixation_distance))
    # convert from radians to degrees
    deg_per_cm *= (180 / np.pi)
    # this gets us the number of cm per pixel
    cm_per_pix = monitor_cm_width / monitor_pix_width
    # so when we multiply them togeter, we get the number of degrees per
    # pixel, so we take their inverse to get pixels per degree
    return 1 / (deg_per_cm * cm_per_pix)


def clear_events(win):
    """
    """
    for i in range(10):
        [w.flip() for w in win]
        core.wait(.01)
        event.getKeys()


def run_calibration(win, img_pos, stim):
    """
    """
    clear_events(win)

    start_text = (u"Press space to begin\nUse arrow keys to adjust the location of the \u25a1 and "
                  "+ until they overlap\nThen press space")
    start_text = [visual.TextStim(w, start_text, pos=p, wrapWidth=1000)
                  for w, p in zip(win, img_pos)]
    [text.draw() for text in start_text]
    [w.flip() for w in win]

    keys = []
    while not keys:
        [text.draw() for text in start_text]
        [w.flip() for w in win]
        core.wait(.1)
        keys = event.getKeys(keyList=['space'])

    clear_events(win)

    while True:
        [s.draw() for s in stim]
        [w.flip() for w in win]
        keys = event.getKeys()
        if 'space' in keys:
            break
        if 'up' in keys:
            img_pos[1][1] += 1
        if 'down' in keys:
            img_pos[1][1] -= 1
        if 'left' in keys:
            img_pos[1][0] -= 1
        if 'right' in keys:
            img_pos[1][0] += 1
        stim[1].pos = img_pos[1]

    print("Final offset: %s" % img_pos[1])
    return img_pos[1]


def ipd_calibration(subject_name, binocular_ipd, output_dir, screen=[0], size=[4096, 2160],
                    fixation_distance=42, monitor_cm_width=69.8, num_runs=3, **monitor_kwargs):
    """
    """
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    monocular_verg_angle = calc_monocular_convergence_angle(binocular_ipd, fixation_distance)
    default_monitor = {'units': 'pix', 'fullscr': True, 'color': 128, 'colorSpace': 'rgb255'}
    for k, v in default_monitor.items():
        monitor_kwargs.setdefault(k, v)
    neutral_verg_angle = calc_monocular_convergence_angle(62)
    pix_per_deg = calc_pix_per_deg(fixation_distance, size[0], monitor_cm_width)
    # guess what the initial offset should be; vertical starts at 0,
    # horizontal is based on difference from neutral_verg_angle
    offset = [int((monocular_verg_angle - neutral_verg_angle) * pix_per_deg), 0]
    # these pairs are horizontal, vertical
    img_pos = [[0, 0], offset]
    print("Using initial binocular offsets: %s" % img_pos)
    if len(screen) == 1:
        print('Doing single-monitor mode on screen %s' % screen)
        win = [visual.Window(winType='glfw', screen=screen[0], size=size, **monitor_kwargs)]
        # Show target icons (□ and +)
        stim = [visual.TextStim(win[0], text=[u'\u25a1', '+'][i], font="consolas", units='pix',
                                pos=img_pos[i], height=[1.5, 3][i]*pix_per_deg, color=(1, 1, 1),
                                colorSpace='rgb') for i in range(2)]
    elif len(screen) == 2:
        # want these to be in increasing order
        screen.sort()
        print("Doing binocular mode on screens %s" % screen)
        win = [visual.Window(winType='glfw', screen=screen[0], swapInterval=1, size=size,
                             **monitor_kwargs)]
        # see here for the explanation of swapInterval and share args
        # (basically, in order to make glfw correctly update the two
        # monitors together):
        # https://discourse.psychopy.org/t/strange-behavior-with-retina-displays-external-monitors-in-1-90-2-py2/5485/5
        win.append(visual.Window(winType='glfw', screen=screen[1], swapInterval=0, share=win[0],
                                 size=size, **monitor_kwargs))
        stim = [visual.TextStim(win[i], text=[u'\u25a1', '+'][i], font="consolas", units='pix',
                                pos=img_pos[i], height=[1.5, 3][i]*pix_per_deg, color=(1, 1, 1),
                                colorSpace='rgb') for i in range(2)]
    else:
        raise Exception("Can't handle %s screens!" % len(monitor_kwargs['screen']))

    calibrated = []
    for i in range(num_runs):
        stim[1].pos = img_pos[1]
        # need to make sure to do this full copy so the img_pos object
        # doesn't get modified in the other function. we also add a bit
        # of random noise so it's not the same each time
        calibrated.append(run_calibration(win, [i.copy()+np.random.randint(-5, 5)
                                                for i in img_pos], stim))
    df = pd.DataFrame({'subject_name': subject_name, 'binocular_ipd': binocular_ipd,
                       'run': list(range(num_runs)), 'screen_width_pix': size[0],
                       'screen_width_cm': monitor_cm_width,
                       'ipd_correction_pix_horizontal': [c[0] for c in calibrated],
                       'ipd_correction_pix_vertical': [c[1] for c in calibrated],
                       'ipd_correction_deg_horizontal': [c[0] / pix_per_deg for c in calibrated],
                       'ipd_correction_deg_vertical': [c[1] / pix_per_deg for c in calibrated],
                       'monocular_vergence_angle': monocular_verg_angle,
                       'fixation_distance_cm': fixation_distance})
    if op.exists(op.join(output_dir, 'ipd_correction.csv')):
        old_df = pd.read_csv(op.join(output_dir, 'ipd_correction.csv'))
        if subject_name not in old_df.subject_name.unique():
            df['session'] = 0
        else:
            df['session'] = old_df.query('subject_name==@subject_name').session.max() + 1
        df = pd.concat([old_df, df])
    else:
        df['session'] = 0
    df.to_csv(op.join(output_dir, 'ipd_correction.csv'), index=False)


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
    parser.add_argument("subject_name", help="Name of the subject")
    parser.add_argument("binocular_ipd", type=float,
                        help=("Subject's binocular IPD, i.e., the distance between their eyes "
                              "(in cm)"))
    parser.add_argument("--output_dir", '-o', help="directory to place output in",
                        default=op.expanduser("~/Desktop/metamers/ipd"))
    parser.add_argument("--screen", '-s', default=[0], type=int, nargs='+',
                        help=("Screen number to display experiment on"))
    parser.add_argument('--fixation_distance', '-f', default=42, type=float,
                        help="Fixation distance (in cm) of the display")
    parser.add_argument("--size", '-p', nargs=2, help="Size of the screen (in pixels)",
                        default=[4096, 2160], type=float)
    parser.add_argument("--monitor_cm_width", '-w', nargs=2, help="Width of the screen in cm",
                        default=69.8, type=float)
    parser.add_argument("--num_runs", "-n", type=int, default=3,
                        help="Number of times to run the calibration")
    args = vars(parser.parse_args())
    ipd_calibration(**args)
