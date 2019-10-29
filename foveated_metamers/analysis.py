"""code to analyze results of psychophysical experiment
"""
import h5py
import numpy as np
import pandas as pd


def summarize_trials(raw_behavioral_path):
    r"""Summarize trials in order to determine whether subject was correct or not

    With this, we create a n_trials by 4 array with the following
    structure: [trial number, time of trial end, button pressed, time
    button press was recorded].

    Because of how psychopy records the button presses, the button a
    subject presses during the response period will be time-stamped to
    line up with the beginning of the *next* trial; this is what we
    assume here. Therefore, if everything is working correctly, the 2nd
    and 4th columns of this array should b basically identical, only
    differing by msecs.

    This array is used by the get_responses function to grab the data
    necessary for making the psychophysical curve

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
    f = h5py.File(raw_behavioral_path)
    trials = []
    button_presses = f['button_presses'][()]
    # these are the button presses we want to ignore. they should only
    # come up if we quit out in the middle and then restarted. also,
    # they're all byte strings
    button_mask = [b[0] not in [b'5', b'q', b'esc', b'escape', b'space'] for b in button_presses]
    button_presses = button_presses[button_mask]
    for i, trial_beg in enumerate(f['timing_data'][()][7::6][:, 2].astype(float)):
        button_where = np.abs(trial_beg - button_presses[:, 1].astype(float)).argmin()
        trials.append([i, trial_beg, *button_presses[button_where]])
    return np.array(trials).astype(float)


def create_experiment_df(df, presentation_idx, dep_variables=['scaling']):
    r"""Create a dataframe summarizing the trials of the experiment

    This function takes in the dataframe summarizing the stimuli, the
    presentation indices, and some dependent variables, and creates a
    dataframe summarizing each trial in the experiment. We have the
    following columns:
    - 'image_name': the name of the reference image used to generate
      these images
    - 'image_1': the seed of the first image presented
    - 'image_2': the seed of the second image presented
    - 'image_X': the seed of the third image presented
    - 'trial_number': the number of this trial
    - 'correct_response': whether the correct response was 1 (if image_1
      and image_X were identical) or 2 (if image_2 and image_X were
      identical)
    - 'model': the model used to generate this metamer
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
        The n_trials by 3 array containing the stimuli presentation
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
    sub_df = df[['image_name']]
    dep_variables.append('model')
    correct_answers = np.where(presentation_idx[:, 2] == presentation_idx[:, 0], 1, 2)
    expt_df = []
    for i, (a, b, x) in enumerate(presentation_idx):
        tmp = sub_df.loc[a].to_dict()
        # these two should be identical
        if tmp != sub_df.loc[b].to_dict():
            raise Exception("Something's gone horribly wrong, the identifying info for trial %s "
                            "is incorrect! %s does not match %s: %s, %s" %
                            (i, a, b, tmp, sub_df.loc[b].to_dict()))
        tmp.update({'image_1': df.loc[a].seed, 'image_2': df.loc[b].seed,
                    'image_X': df.loc[x].seed, 'trial_number': i,
                    'correct_response': correct_answers[i]})
        for v in dep_variables:
            # at most one of these will be nan. they could be strings,
            # in which case the isinstance call will fail
            if isinstance(df.loc[a][v], float) and np.isnan(df.loc[a][v]):
                tmp.update({v: df.loc[b][v]})
            else:
                tmp.update({v: df.loc[a][v]})
                if not isinstance(df.loc[b][v], float) or not np.isnan(df.loc[b][v]):
                    if df.loc[a][v] != df.loc[b][v]:
                        raise Exception("Something's gone horribly wrong, dependent variable for "
                                        "images %s and %s in trial %s don't match: %s, %s" %
                                        (a, b, i, df.loc[a][v], df.loc[b][v]))
        expt_df.append(pd.DataFrame(tmp, [i]))
    expt_df = pd.concat(expt_df).reset_index(drop=True)
    # all NaNs are where we have a reference image
    expt_df = expt_df.fillna('reference')
    # insert information on trial type: metamer vs metamer or metamer vs
    # reference. if either image is a reference, then this is metamer vs
    # reference; else, it's metamer vs metamer
    metamer_vs_reference = np.logical_or((expt_df.image_1 == 'reference').values,
                                         (expt_df.image_2 == 'reference').values)
    expt_df['trial_type'] = np.where(metamer_vs_reference, 'metamer_vs_reference',
                                     'metamer_vs_metamer')
    return expt_df


def add_response_info(expt_df, trials, subject_name):
    r"""Add information about subject's response and correctness to expt_df

    This function takes the expt_df, which summarizes the trials of the
    experiment, and adds three additional columns: 'subject_response',
    which gives the number (1 or 2) the subject pressed on this trial,
    'hit_or_miss', which contains either 'hit' or 'miss', describing
    whether the subject was correct or not, and 'subject_name', which
    contains the name of the subject corresponding to the trials array.

    Parameters
    ----------
    expt_df : pd.DataFrame
        The experiment information dataframe, as created by
        analysis.create_experiment_df
    trials : np.array
        The n_trials by 4 array created by analysis.summarize_trials
    subject_name : str
        The name of this subject

    Returns
    -------
    expt_df : pd.DataFrame
        The modified experiment dataframe, with three additional
        columns. See above for description

    """
    # just in case it was an incomplete trial
    expt_df = expt_df.iloc[:len(trials)]
    subj_answers = trials[:, 2].astype(int)
    expt_df['subject_response'] = subj_answers
    expt_df['hit_or_miss'] = np.where(expt_df.correct_response == expt_df.subject_response, 'hit',
                                      'miss')
    expt_df['subject_name'] = subject_name
    return expt_df


def summarize_expt(expt_df, dep_variables=['scaling', 'trial_type']):
    r"""Summarize expt_df to get proportion correct

    Here, we take the ``expt_df`` summarizing the experiment's trials
    and the subject's responses, and we compute the proportion correct
    on each trial type. We end up with a DataFrame that has the columns
    ``['subject_name', 'image_name', 'model'] + dep_variables`` from
    ``expt_df``, as well as two new columns: ``'n_trials'`` (which gives
    the number of trials in that condition) and ``'proportion_correct'``
    (which gives the proportion of time the subject was correct in that
    condition).

    Parameters
    ----------
    expt_df : pd.DataFrame
        The experiment information dataframe, after modification by
        analysis.add_response_info
    dep_variable : list, optional
        A list of strs, containing one or more of the columns of
        ``expt_df``, which tells us which additional variable(s) we want
        to include in our dataframe.

    Returns
    -------
    summary_df : pd.DataFrame

    """
    expt_df = expt_df.copy()
    expt_df['hit_or_miss'] = expt_df.hit_or_miss.apply(lambda x: {'hit': 1, 'miss': 0}[x])

    gb = expt_df.groupby(['subject_name', 'image_name', 'model'] + dep_variables)
    summary_df = gb.count()['image_1'].reset_index()
    summary_df = summary_df.merge(gb.hit_or_miss.mean().reset_index())
    summary_df = summary_df.rename(columns={'image_1': 'n_trials',
                                            'hit_or_miss': 'proportion_correct'})
    return summary_df
