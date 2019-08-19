"""code to analyze results of psychophysical experiment
"""
import h5py
import numpy as np


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
    for i, trial_beg in enumerate(f['timing_data'][()][7::6][:, 2].astype(float)):
        button_where = np.abs(trial_beg - f['button_presses'][()][:, 1].astype(float)).argmin()
        trials.append([i, trial_beg, *f['button_presses'][()][button_where]])
    return np.array(trials).astype(float)


def get_responses(df, presentation_idx, trials, dep_variable='scaling'):
    r"""Summarize subject's correct/incorrect as function of dep_variable

    This function returns a dictionary with keys corresponding to the
    different levels of ``dep_variable`` presented in the experiment and
    values that are arrays containing 1s and 0s: 1 for every trial the
    subject was correct, 0 for every trial they were incorrect

    Parameters
    ----------
    df : pd.DataFrame
        The metamer information dataframe, as created by
        stimuli.create_metamer_df
    presentation_idx : np.array
        The n_trials by 3 array containing the stimuli presentation
        indices for the run being analyzed.
    trials : np.array
        The n_trials by 4 array created by analysis.summarize_trials
    dep_variable : str, optional
        One of the columns of ``df``, which tells us which variable we
        want to use as the dependent variable in our psychometric
        function.

    Returns
    -------
    responses : dict
        The response dictionary, as described above

    """
    correct_answers = np.where(presentation_idx[:, 2] == presentation_idx[:, 0], 1, 2)
    subj_answers = trials[:, 2]
    dep_variable_vals = np.where(~np.isnan(df.loc[presentation_idx[:, 0]][dep_variable]),
                                 df.loc[presentation_idx[:, 0]][dep_variable],
                                 df.loc[presentation_idx[:, 1]][dep_variable])
    subj_correct = (correct_answers == subj_answers).astype(int)

    responses = {}
    for k in df[dep_variable].unique():
        if np.isnan(k):
            continue
        dep_variable_trials = np.where(dep_variable_vals[:5] == k)
        responses[k] = subj_correct[dep_variable_trials]
    return responses
