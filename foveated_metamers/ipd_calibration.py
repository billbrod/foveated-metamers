#!/usr/bin/python
"""psychopy script for running IPD calibration, run from command-line
"""
import argparse
import warnings
import os
import numpy as np
import pandas as pd
import os.path as op
from psychopy import visual, event, core


def csv_to_binocular_offset(ipd_csv, subject_name, units='pix'):
    """Compute the binocular offset from the ipd_correction.csv file

    The ipd_correction.csv file contains all the information for the
    subjects we've run, and this little helper function will load it in
    and give you the average (horizontal, vertical) offset, in
    ``units``, for ``subject_name``.

    This gives the difference between their centers that you should
    use. It's up to you how exactly to do this, but I recommend moving
    the image in one eye forward by half this amount and the other
    backward by half this amount (separately for horizontal and
    vertical)

    Parameters
    ----------
    ipd_csv : str or pandas.DataFrame
        Either the DataFrame object containing this information or the
        path to the csv file containing the DataFrame
    subject_name : str
        The name of the subject to find the average binocular offset for
    units : {'pix', 'deg'}
        Whether to give the binocular offset in pixels or degrees

    Returns
    -------
    binocular_offset : list
        List of 2 ints (if ``units=='pix'``) or floats (if
        ``units=='deg'``) specifying the offset between the two images
        that you should be using for this subject

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
    """Calculate the pixels per degree for a given set up

    Using the fixation distance (in cm) and width of the monitor in
    pixels and cm, we can quickly calculate the number of pixels per
    degree.

    Parameters
    ----------
    fixation_distance : float
        The fixation distance of the monitor, in cm. Default value is
        for the FancyPants v1 haploscope
    monitor_pix_width : int
        Width of the monitor, in pixels. Default value is for the
        FancyPants v1 haploscope
    monitor_cm_width : float
        Width of the monitor, in cm. Default value is for the FancyPants
        v1 haploscope

    Returns
    -------
    pix_per_deg : float
        Number of pixels per degree. Therefore, multiply this by the
        size of something in degrees in order to get its size in pixels

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
    """clear all keys from psychopy

    I've been having issues with keys remaining in psychopy's buffer,
    being pulled in at a later time (thus, pressing the spacebar once
    gets parsed as pressing it twice, in rapid succession). This takes a
    brief pause, .1 seconds, to rapidly flip the window several times
    (should look like just a gray screen), calling ``event.getKeys()``
    each time in order to make sure no keys are left in the buffer

    Parameters
    ----------
    win : list
        List of Psychopy.visual.Window objects that we want to flip

    """
    for i in range(10):
        [w.flip() for w in win]
        core.wait(.01)
        event.getKeys()


def run_calibration(win, img_pos, stim, vert_or_horiz, flip_text=True):
    """run the actual calibration task

    For a given run, we take the initialized windows, stimuli, and their
    starting conditions, and allow the user to move one around until the
    two stimuli align. This will work for either one or two windows (win
    should be a list in either case, just with 1 or 2 entries), but will
    probably fail for more than that.

    The run continues until the user presses the space bar. We ignore
    all buttons other than the arrow keys and space, which we use to
    start and end the run.

    Parameters
    ----------
    win : list
        List of Psychopy.visual.Window objects where we'll be displaying
        the stimuli. Has been tested when this contains one or two
        windows, but could be extended to more, if you want to, for some
        reason.
    img_pos : list
        List of 2-tuples of ints, same length as stim. The starting
        locations for our stimuli.
    stim : list
        List of Psychopy.visual.TextStim, length 2. The two stimuli
        (square and cross) which the user is adjusting for this task.
    vert_or_horiz : {'vert', 'horiz'}
        Whether we're doing the vertical or horizontal calibration
    flip_text : bool, optional
        Boolean, whether to left-right reverse everything. If True,
        everything will be flipped, as you need on the haploscope. If
        False, everything will be the right way round. This flips the
        text but also reverses the direction the left/right arrow keys
        move the stimuli

    Returns
    -------
    img_pos : tuple
        2-tuple of ints giving the user's final offset

    """
    clear_events(win)

    if vert_or_horiz == 'vert':
        start_text = (u"Press space to begin, q or esc to quit (without saving anything)\nUse "
                      "up/down arrow keys to adjust the vertical position of the line until it"
                      " lies in the center of the circle, then press space")
    elif vert_or_horiz == 'horiz':
        start_text = (u"Press space to begin, q or esc to quit (without saving anything)\nUse "
                      "left/right arrow keys to adjust the horizontal position of the line until"
                      " it lies in the center of the circle, then press space")
    start_text = [visual.TextStim(w, start_text, pos=p, wrapWidth=1000, flipHoriz=flip_text)
                  for w, p in zip(win, img_pos)]
    [text.draw() for text in start_text]
    [w.flip() for w in win]

    keys = []
    while not keys:
        [text.draw() for text in start_text]
        [w.flip() for w in win]
        core.wait(.1)
        keys = event.getKeys(keyList=['space', 'q', 'esc', 'escape'])

    if 'q' in keys or 'esc' in keys or 'escape' in keys:
        # then the user wanted to quit
        print("Quitting out early, not saving any position")
        return None

    clear_events(win)

    # when we're viewing through a mirror, want to flip the left/right
    # direction the arrow keys map to as well
    key_direction = {False: 1, True: -1}[flip_text]

    while True:
        [s.draw() for s in stim]
        [w.flip() for w in win]
        keys = event.getKeys()
        if 'space' in keys or 'q' in keys or 'esc' in keys or 'escape' in keys:
            break
        if vert_or_horiz == 'vert':
            if 'up' in keys:
                img_pos[1][1] += 1
            if 'down' in keys:
                img_pos[1][1] -= 1
        elif vert_or_horiz == 'horiz':
            if 'left' in keys:
                img_pos[1][0] -= key_direction*1
            if 'right' in keys:
                img_pos[1][0] += key_direction*1
        stim[1].pos = img_pos[1]

    # vert_or_horiz=='vert' will evaluate to True, and thus 1, if this
    # is the vertical trial and False, and thus 0, if this is the
    # horizontal one. That's what we want in order to correctly display
    # the right info
    if 'space' in keys:
        print("Final %s offset: %s" % (vert_or_horiz, img_pos[1][vert_or_horiz == 'vert']))
        return img_pos[1][vert_or_horiz == 'vert']
    else:
        # then the user wanted to quit
        print("Quitting out early, not saving any position")
        return None


def ipd_calibration(subject_name, binocular_ipd, output_dir, screen=[0], size=[4096, 2160],
                    fixation_distance=42, monitor_cm_width=69.8, num_runs=3, flip_text=True,
                    default_ipd=6.2, allow_large_ipd=False, line_length=400, line_width=10,
                    circle_radius=25, **window_kwargs):
    """Run the full IPD calibration task

    On a haploscope, two images are presented, one to each eye. The
    construction of any haploscope is done with a certain default
    inter-pupillary distance (IPD) in mind, probably 6.2 cm. If the
    subject has an IPD very different from this, it can be difficult to
    successfully fuse the image, so we want to adjust the images'
    relative centers. We start out by doing a bit of trigonometry to get
    them approximately correct, and then the user does an IPD
    calibration task, where they adjust the location of two objects (a
    square and a cross), presented in separate eyes, until they
    overlap. This is done ``num_runs`` times (each run starts with a bit
    of noise, an integer drawn from a uniform distribution from -5 to 5,
    in both directions), and then we append these results to an
    ipd_correction.csv file in the ``output_dir``, where we're keeping
    track of this information.

    If you want to use the information stored in this csv, the
    ``csv_to_binocular_offset`` function will help you with that

    Parameters
    ----------
    subject_name : str
        Name of the subject. Will be used to record this information in
        the ipd_correction.csv
    binocular_ipd : float
        The subject's inter-pupillary distance (IPD), that is, the
        distance between the subject's eyes, in *cm*
    output_dir : str
        Path to the directory where we'll output the results, in
        ipd_correction.csv
    screen : list
        List of ints giving the identifiers for the screens to run the
        task on. Can contain one or two values.
    size : list
        List of 2 ints, containing (width, height) of the screen(s) in
        pixels. Used for converting between pixels and
        degrees. Currently, all screens must have same size. Default
        value is for the FancyPants v1 haploscope
    fixation_distance : float
        The fixation distance of the monitor, in cm. Default value is
        for the FancyPants v1 haploscope
    monitor_cm_width : float
        Width of the monitor, in cm. Default value is for the FancyPants
        v1 haploscope
    num_runs : int
        Number of times to run this the calibration task. We'll store
        all of them, and then it's expected that we'll average over all
        results for a given subject.
    flip_text : bool, optional
        Boolean, whether to left-right reverse everything. If True,
        everything will be flipped, as you need on the haploscope. If
        False, everything will be the right way round. This flips the
        text but also reverses the direction the left/right arrow keys
        move the stimuli
    default_ipd : float
        The IPD (in cm) of the system's default IPD, that is, the IPD
        where we shouldn't have to adjust the images at all. Used to get
        a "first guess" based on the subject's IPD
    allow_large_ipd : bool
        It's easy to mess up and give an IPD in mm instead of cm, but we
        require cm. In order to help check that, by default, we'll raise
        an Exception if either binocular_ipd or default_ipd is larger
        than 10, because that would be very large. If you really do have
        IPDs larger than 10 cm for either of those values, you can set
        this flag to True and we won't raise the Exception (but we'll
        still raise a warning).
    window_kwargs : kwargs
        Other keyword=value pairs will be passed directly to the
        creation of the Psychopy.visual.Window objects
    line_length : int
        Length of the line stimulus, in pixels
    line_width : int
        Width of the line stimulus, in pixels
    circle_radius : int
        Radius of the circle stimulus, in pixels

    """
    if (binocular_ipd > 10 or default_ipd > 10):
        if not allow_large_ipd:
            raise Exception("Your IPD values are really large! Are you sure you didn't input the "
                            "IPD in mm? Is someone's IPD actually greater than 10 cm? If you're "
                            "sure (and you didn't accidentally input them in mm), run this again"
                            " with the allow_large_ipd flag set to True")
        else:
            warnings.warn("Your IPD values are really large but you say you know what you're"
                          " doing...")
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    monocular_verg_angle = calc_monocular_convergence_angle(binocular_ipd, fixation_distance)
    default_window = {'units': 'pix', 'fullscr': True, 'color': 0, 'colorSpace': 'rgb255',
                      'allowGUI': False}
    for k, v in default_window.items():
        window_kwargs.setdefault(k, v)
    neutral_verg_angle = calc_monocular_convergence_angle(default_ipd)
    pix_per_deg = calc_pix_per_deg(fixation_distance, size[0], monitor_cm_width)
    # guess what the initial offset should be; vertical starts at 0,
    # horizontal is based on difference from neutral_verg_angle
    offset = [int((neutral_verg_angle - monocular_verg_angle) * pix_per_deg), 0]
    # these pairs are horizontal, vertical
    img_pos = [[0, 0], offset]
    print("Using initial binocular offsets: %s" % img_pos)
    # want these to be in increasing order
    screen.sort()
    win = [visual.Window(winType='glfw', screen=screen[0], swapInterval=1, size=size,
                         **window_kwargs)]
    circle_stim = visual.Circle(win[0], units='pix', pos=img_pos[0], radius=circle_radius,
                                lineColor=(1, 1, 1), lineColorSpace='rgb')
    horiz_line_start = [int(-line_length)//2, 0]
    horiz_line_end = [int(line_length)//2, 0]
    # the vertical is just a reversed version of the horizontal (since
    # it goes x, y)
    vert_line_start = horiz_line_start[::-1]
    vert_line_end = horiz_line_end[::-1]
    if len(screen) == 1:
        print('Doing single-monitor mode on screen %s' % screen)
        # line stimuli
        line_stim = [visual.Line(win[0], start=[horiz_line_start, vert_line_start][i],
                                 end=[horiz_line_end, vert_line_end][i], units='pix',
                                 lineWidth=line_width, pos=img_pos[1], lineColor=(1, 1, 1),
                                 lineColorSpace='rgb') for i in range(2)]
    elif len(screen) == 2:
        print("Doing binocular mode on screens %s" % screen)
        # see here for the explanation of swapInterval and share args
        # (basically, in order to make glfw correctly update the two
        # monitors together):
        # https://discourse.psychopy.org/t/strange-behavior-with-retina-displays-external-monitors-in-1-90-2-py2/5485/5
        win.append(visual.Window(winType='glfw', screen=screen[1], swapInterval=0, share=win[0],
                                 size=size, **window_kwargs))
        line_stim = [visual.Line(win[1], start=[horiz_line_start, vert_line_start][i],
                                 lineWidth=line_width, end=[horiz_line_end, vert_line_end][i],
                                 units='pix', pos=img_pos[1], lineColor=(1, 1, 1),
                                 lineColorSpace='rgb') for i in range(2)]
    else:
        raise Exception("Can't handle %s screens!" % len(window_kwargs['screen']))

    calibrated = []
    for i in range(num_runs):
        # need to make sure to do this full copy so the img_pos object
        # doesn't get modified in the other function. we also add a bit
        # of random noise so it's not the same each time
        for j in range(2):
            trial_type = ['vert', 'horiz'][j]
            new_pos = [[int(k + np.random.randint(-5, 5)) for k in l.copy()] for l in img_pos]
            line_stim[j].pos = new_pos[1]
            shift_amt = run_calibration(win, new_pos, [circle_stim, line_stim[j]],
                                        trial_type, flip_text)
            if shift_amt is None:
                # then the user pressed q or esc and we want to quit
                # without saving anything
                return
            calibrated.append([shift_amt, trial_type])
    df = pd.DataFrame({'subject_name': subject_name, 'binocular_ipd': binocular_ipd,
                       'run': list(range(num_runs)), 'screen_width_pix': size[0],
                       'screen_width_cm': monitor_cm_width,
                       'ipd_correction_pix_horizontal': [c[0] for c in calibrated
                                                         if c[1] == 'horiz'],
                       'ipd_correction_pix_vertical': [c[0] for c in calibrated if c[1] == 'vert'],
                       'ipd_correction_deg_horizontal': [c[0] / pix_per_deg for c in calibrated
                                                         if c[1] == 'horiz'],
                       'ipd_correction_deg_vertical': [c[0] / pix_per_deg for c in calibrated
                                                       if c[1] == 'vert'],
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
        description=("Run the full IPD calibration task. "
                     "On a haploscope, two images are presented, one to each eye. The "
                     "construction of any haploscope is done with a certain default "
                     "inter-pupillary distance (IPD) in mind, probably 6.2 cm. If the "
                     "subject has an IPD very different from this, it can be difficult to "
                     "successfully fuse the image, so we want to adjust the images' "
                     "relative centers. We start out by doing a bit of trigonometry to get"
                     " them approximately correct, and then the user does an IPD "
                     "calibration task, where they adjust the location of two objects (a"
                     " square and a cross), presented in separate eyes, until they "
                     "overlap. This is done ``num_runs`` times (each run starts with a bit "
                     "of noise, an integer drawn from a uniform distribution from -5 to 5,"
                     " in both directions), and then we append these results to an "
                     "ipd_correction.csv file in the ``output_dir``, where we're keeping "
                     "track of this information. If you want to use the information stored"
                     "in this csv, the ``csv_to_binocular_offset`` function will help you "
                     "with that"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("subject_name", help="Name of the subject")
    parser.add_argument("binocular_ipd", type=float,
                        help=("Subject's binocular IPD, i.e., the distance between their eyes "
                              "(in cm)"))
    parser.add_argument("--output_dir", '-o', help="directory to place output in",
                        default=op.expanduser("~/Desktop/metamers/ipd"))
    parser.add_argument("--screen", '-s', default=[0], type=int, nargs='+',
                        help=("Screen number to display experiment on"))
    parser.add_argument('--fixation_distance', '-d', default=42, type=float,
                        help="Fixation distance (in cm) of the display")
    parser.add_argument("--size", '-p', nargs=2, help="Size of the screen (in pixels)",
                        default=[4096, 2160], type=float)
    parser.add_argument("--monitor_cm_width", '-c', help="Width of the screen in cm",
                        default=69.8, type=float)
    parser.add_argument("--num_runs", "-n", type=int, default=3,
                        help="Number of times to run the calibration")
    parser.add_argument("--no_flip", '-f', action='store_true',
                        help=("This script is meant to be run on the haploscope. Therefore, we "
                              "left-right flip all text by default. Use this option to disable"
                              " that"))
    parser.add_argument("--allow_large_ipd", action='store_true',
                        help=("It's easy to mess up and give an IPD in mm instead of cm, but we"
                              " require cm. In order to help check that, by default, we'll raise"
                              " an Exception if either binocular_ipd or default_ipd is larger than"
                              " 10, because that would be very large. If you really do have IPDs "
                              "larger than 10 cm for either of those values, you can set this flag"
                              " to True and we won't raise the Exception (but we'll still raise a"
                              " warning)."))
    parser.add_argument("--line_length", '-l', default=400,
                        help="Length of the line stimulus, in pixels")
    parser.add_argument("--line_width", '-w', default=10,
                        help="Width of the line stimulus, in pixels")
    parser.add_argument("--circle_radius", '-r', default=25,
                        help="Radius of the circle stimulus, in pixels")
    args = vars(parser.parse_args())
    flip = not args.pop('no_flip')
    ipd_calibration(flip_text=flip, **args)
