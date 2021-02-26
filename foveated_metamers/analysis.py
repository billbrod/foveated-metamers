"""code to analyze results of psychophysical experiment
"""
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_trials(raw_behavioral_path):
    r"""Summarize trials in order to determine whether subject was correct or not.

    With this, we create a n_trials by 4 array with the following
    structure: [trial number, time of trial end, button pressed, time
    button press was recorded, time when image flashed off].

    If you're using pyglet, as assumed here, PsychoPy should accurately record
    the timing of button presses, and the button press should therefore lie
    between the time of trial end and time when image flashed off (which we
    check in this function), and should lie closer to the time of trial end, on
    the order of 1e-4 secs. Timing data is returned so you can check this.

    If you're using glfw, the button a subject presses during the response
    period will be time-stamped to line up with the beginning of the *next*
    event. Therefore, if everything is working correctly, the 2nd and 4th
    columns of this array should be basically identical, only differing by
    msecs.

    This array is used by the get_responses function to grab the data
    necessary for making the psychophysical curve.

    NOTE: I think whether this works depends on the PsychoPy mode (e.g.,
    pyglet, glfw) you use, so if you change that double-check this

    Parameters
    ----------
    raw_behavioral_path : str
        The str to the hdf5 file that contains the behavioral results,
        as saved by the experiment.py script

    Returns
    -------
    trials : np.array
        The trials summary array, as described above

    """
    f = h5py.File(raw_behavioral_path, 'r')
    trials = []
    button_presses = f['button_presses'][()]
    # these are the button presses we want to ignore. they should only
    # come up if we quit out in the middle and then restarted. also,
    # they're all byte strings
    button_mask = [b[0] not in [b'return', b'5', b'q', b'esc', b'escape', b'space'] for b in button_presses]
    button_presses = button_presses[button_mask]
    # grab the timing events corresponding to the events immediately after the
    # button press at the end of each trial: the post-stimulus pause or the
    # beginning of any pause
    timing_data = np.array([t for t in f['timing_data'][()] if (b'post-' in t[0] and b'on' in t[1])
                            or (b'pause' in t[0] and b'start' in t[1])])
    # remove events corresponding to the beginning of the trial after each
    # pause (because the pause event takes its place).
    timing_data = np.delete(timing_data, np.where([b'pause' in t[0] for t in timing_data])[0]+1, 0)
    trial_end_timing = np.array([t for t in f['timing_data'][()] if (b'-1' in t[0] and b'off' in t[1])])
    for i, trial_beg in enumerate(timing_data[:, 2].astype(float)):
        button_where = np.abs(trial_beg - button_presses[:, 1].astype(float)).argmin()
        trials.append([i, trial_beg, *button_presses[button_where], trial_end_timing[i, 2]])
    f.close()
    trials = np.array(trials).astype(float)
    if (any(trials[:, -1] < trial_end_timing[:, -1].astype(float)) or
        any(trials[:, -1] > timing_data[:, -1].astype(float))):
        raise Exception("Timing info messed up! Somehow the button press wasn't between the end of "
                        "one trial and the beginning of the next!")
    if not all([i==1 or i==2 for i in trials[:, 2]]):
        raise Exception("One of the button presses was something other than 1 or 2!")
    return trials


def plot_timing_info(trials, subject_name, session_number, run_number,
                     figsize=(5,5)):
    """Create scatter plot of timing info.

    This takes the trials array created to summarize subject responses and
    plots the response time vs the time before trial end, with color determined
    by which button was pressed. This is mainly done to make sure everything's
    working as expected: time before trial end should be pretty constant, on
    the order of 1e-4 secs, with no dependence on response time, which will
    vary much more, and no dependence on button press.

    We add a title to identify the subject, session and image set.

    Parameters
    ----------
    trials : np.array
        The n_trials by 5 array created by analysis.summarize_trials
    subject_name : str
        The name of this subject
    session_number : int
        Session number
    run_number : int
        Run number
    figsize : tuple, optional
        size of the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing this plot

    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i in [1, 2]:
        tmp = trials[trials[:, 2] == i]
        response_time = tmp[:, 3] - tmp[:, 4]
        time_before_end = tmp[:, 1] - tmp[:, 3]
        ax.scatter(response_time, time_before_end, c=f'C{i-1}',
                   label=f'button {i}')
    ax.legend()
    lim = np.abs(trials[:, 1] - trials[:, 3]).max()
    # y-values will all be strictly positive, because of the check we do in
    # summarize_trials()
    ax.set(ylabel='Time between button press and trial end', xlabel='Response time',
           ylim=(0, lim + .1*lim),
           title=(f"{subject_name}, session {session_number}, "
                  f"run {run_number}\nThis is a diagnostic plot: y-values should be "
                  "pretty constant, on order of 1e-4 seconds,\nwith no dependency on "
                  "response time, which will vary more but should have no "
                  "relationship with button press."))
    return fig


def _get_values(df, trial_num, correct_answers, idx=[], labels=[],
                dep_variables=['scaling']):
    """Grab appropriate values out of df.

    Helper function for assembling the experiment_df based on a set of
    presentation indices

    """
    sub_df = df[['image_name']]
    tmp = sub_df.loc[idx[0]].to_dict()
    # these two should be identical
    if tmp != sub_df.loc[idx[1]].to_dict():
        raise Exception("Something's gone horribly wrong, the identifying info for trial %s "
                        "is incorrect! %s does not match %s: %s, %s" %
                        (trial_num, idx[0], idx[1], tmp, sub_df.loc[idx[1]].to_dict()))
    tmp.update({f'image_{l}': df.loc[i].seed for i, l in zip(idx, labels)})
    tmp.update({'trial_number': trial_num,
                'correct_response': correct_answers[trial_num]})
    for v in dep_variables:
        # at most one of these will be nan. they could be strings,
        # in which case the isinstance call will fail
        possible_vals = [df.iloc[i][v] for i in idx]
        # this double loop allows to set the value with the first non-NaN we
        # find and then double-check that all non-NaNs have the same value
        for poss_v in possible_vals:
            # we check if it's a float first because we can't call np.isnan on
            # a str
            if not isinstance(poss_v, float) or not np.isnan(poss_v):
                tmp.update({v: poss_v})
        for poss_v in possible_vals:
            if not isinstance(poss_v, float) or not np.isnan(poss_v):
                if poss_v != tmp[v]:
                    raise Exception("Something's gone horribly wrong, dependent variable for "
                                    "images %s in trial %s don't match: %s" %
                                    (idx, trial_num, possible_vals))
    return tmp


def create_experiment_df_split(df, presentation_idx, dep_variables=['scaling']):
    r"""Create a dataframe summarizing the trials of the split-screen experiment.

    This function takes in the dataframe summarizing the stimuli, the
    presentation indices, and some dependent variables, and creates a
    dataframe summarizing each trial in the experiment. We have the
    following columns:
    - 'image_name': the name of the reference image to compare
      against this metamer in the experiment
    - 'image_left_1': the seed of the first image presented on the left
    - 'image_left_2': the seed of the second image presented on the left
    - 'image_right_1': the seed of the first image presented on the right
    - 'image_right_2': the seed of the second image presented on the right
    - 'trial_number': the number of this trial
    - 'correct_response': whether the correct response was 1 (if image_left_1
      and image_left_2 are different) or 2 (if image_right_1 and image_right_2
      are different)
    - 'model': the model used to generate this metamer
    - 'trial_type': whether this trial was metamer_vs_metamer or
      metamer_vs_reference
    - 'unique_seed': for metamer_vs_reference trials, the seed of the metamer.
      for metamer_vs_metamer, None.
    - an additional column for each item in ``dep_variable``

    On some trials, one of the two images presented will be a
    metamer. In that case, their seed (as stored in ``df``) is
    ``np.nan``. In ``expt_df`` we replace this with the str
    ``'reference'``.

    Parameters
    ----------
    df : pd.DataFrame
        The metamer information dataframe, as created by
        stimuli.create_metamer_df
    presentation_idx : np.array
        The 2 x n_trials x 2 array containing the stimuli presentation
        indices for the run being analyzed.
    dep_variable : list, optional
        A list of strs, containing one or more of the columns of ``df``,
        which tells us which additional variable(s) we want to include
        in our dataframe. These are assumed to be identical in the two
        images presented on each trial (function will raise an Exception
        if this not true), and the intended use case if that these will
        be the dependent variables against which we will plot our
        psychometric function

    Returns
    -------
    expt_df : pd.DataFrame
        The experiment information dataframe, see above for description

    """
    dep_variables.append('model')
    correct_answers = np.where(presentation_idx[0, :, 0] != presentation_idx[0, :, 1], 1, 2)
    expt_df = []
    for i, (l, r) in enumerate(zip(presentation_idx[0], presentation_idx[1])):
        tmp = _get_values(df, i, correct_answers, [*l, *r],
                          ['left_1', 'left_2', 'right_1', 'right_2'],
                          dep_variables)
        expt_df.append(pd.DataFrame(tmp, [i]))
    expt_df = pd.concat(expt_df).reset_index(drop=True)
    # all NaNs are where we have a reference image
    expt_df = expt_df.fillna('reference')
    # insert information on trial type: metamer vs metamer or metamer vs
    # reference. if either image is a reference, then this is metamer vs
    # reference; else, it's metamer vs metamer
    metamer_vs_reference = [('reference' in r[1].values) for r in expt_df.iterrows()]
    expt_df['trial_type'] = np.where(metamer_vs_reference, 'metamer_vs_reference',
                                     'metamer_vs_metamer')

    def find_seed(x):
        x = x.tolist()
        x.remove('reference')
        # only makes sense ot do this if there's only one seed (i.e., in
        # metamer vs reference). in metamer vs metamer, return None for now
        if len(x) == 1:
            return x[0]
        else:
            return None
    expt_df['unique_seed'] = expt_df[['image_left_2', 'image_right_2']].apply(find_seed, 1)
    return expt_df


def add_response_info(expt_df, trials, subject_name, session_number, run_number):
    r"""Add information about subject's response and correctness to expt_df.

    This function takes the expt_df, which summarizes the trials of the
    experiment, and adds several additional columns:

    - 'subject_response', which gives the number (1 or 2) the subject pressed
      on this trial
    - 'response_time', which contains the time (in seconds) between when the
      stimulus turned off and when the subject pressed the response button
    - 'hit_or_miss', which contains either 'hit' or 'miss', describing whether
      the subject was correct or not
    - 'hit_or_miss_numeric', which maps the above column columns with {'hit':
      1, 'miss': 0}
    - 'subject_name', which contains the name of the subject corresponding to
      the trials array
    - 'extra_image_set': {'A', 'B'}. Based on subject_name (even-numbered
      subjects get A, odd-numbered get B), this determines the set of extra 5
      images this subject was shown in addition to the 10 all subjects are
      shown (across all sessions).
    - 'session_number', which gives the number of this experimental
      session (determines which image_name values were used).
    - 'run_number', which gives the number for this run.

    Parameters
    ----------
    expt_df : pd.DataFrame
        The experiment information dataframe, as created by
        analysis.create_experiment_df
    trials : np.array
        The n_trials by 5 array created by analysis.summarize_trials
    subject_name : str
        The name of this subject
    session_number : int
        Session number
    Run_number : int
        Run number

    Returns
    -------
    expt_df : pd.DataFrame
        The modified experiment dataframe, with three additional
        columns. See above for description

    """
    # just in case it was an incomplete session
    expt_df = expt_df.iloc[:len(trials)]
    subj_answers = trials[:, 2].astype(int)
    response_time = trials[:, 3] - trials[:, 4]
    expt_df['subject_response'] = subj_answers
    expt_df['response_time'] = response_time
    expt_df['hit_or_miss'] = np.where(expt_df.correct_response == expt_df.subject_response, 'hit',
                                      'miss')
    expt_df['hit_or_miss_numeric'] = expt_df.hit_or_miss.apply(lambda x: {'hit': 1, 'miss': 0}[x])
    expt_df['subject_name'] = subject_name
    try:
        sub_num = int(subject_name.replace('sub-', ''))
        # we alternate sets A and B
        expt_df['extra_image_set'] = {0: 'A', 1: 'B'}[sub_num % 2]
    except ValueError:
        # then this was sub-training, and so there was no extra image set
        expt_df['extra_image_set'] = 'training'
    expt_df['session_number'] = session_number
    expt_df['run_number'] = run_number
    # this is the difference between the time of trial end for the last and
    # first trials, in seconds .So it won't count the tear-down time, but that
    # shouldn't take too long.
    expt_df['approximate_run_length'] = trials[-1, 1] - trials[0, 1]
    return expt_df


def summarize_expt(expt_df, dep_variables=['scaling', 'trial_type'], bootstrap_num=0):
    r"""Summarize expt_df to get proportion correct

    Here, we take the ``expt_df`` summarizing the experiment's trials and the
    subject's responses, and we compute the proportion correct on each trial
    type. We end up with a DataFrame that has the columns ``['subject_name',
    'model', 'trial_type'] + dep_variables`` from ``expt_df``, as well as two
    or three new columns: ``'n_trials'`` (which gives the number of trials in
    that condition), ``'proportion_correct'`` (which gives the proportion of
    time the subject was correct in that condition), and (if
    ``bootstrap_num>0``) ``'bootstrap_num'`` (which runs from 0 to
    ``bootstrap_num`` and gives the bootstrap index).

    Parameters
    ----------
    expt_df : pd.DataFrame
        The experiment information dataframe, after modification by
        analysis.add_response_info
    dep_variable : list, optional
        A list of strs, containing one or more of the columns of
        ``expt_df``, which tells us which additional variable(s) we want
        to include in our dataframe.
    bootstrap_num : int, optional
        How many times to bootstrap when computing the proportion_correct. If
        0, we don't bootstrap (just take the mean across all observations).
        Else, we sample (with replacement) along the same categories as we
        summarized along (i.e., subject, image name, model, and
        ``dep_variables``).

    Returns
    -------
    summary_df : pd.DataFrame

    """
    expt_df = expt_df.copy()

    gb_cols = (['subject_name', 'image_name', 'model'] + dep_variables)

    gb = expt_df.groupby(gb_cols)
    summary_df = gb.count()['trial_number'].reset_index()
    if not bootstrap_num:
        summary_df = summary_df.merge(gb.hit_or_miss_numeric.mean().reset_index())
    else:
        bootstrapped = []
        for i in range(bootstrap_num):
            tmp = gb.sample(frac=1, replace=True).groupby(gb_cols)
            bootstrapped.append(summary_df.merge(tmp.hit_or_miss_numeric.mean().reset_index()))
            bootstrapped[-1]['bootstrap_num'] = i
        summary_df = pd.concat(bootstrapped).reset_index(drop=False)
    summary_df = summary_df.rename(columns={'trial_number': 'n_trials',
                                            'hit_or_miss_numeric': 'proportion_correct'})
    return summary_df
