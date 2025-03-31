import os
import json
import itertools
import numpy as np
import shutil
from psychopy import __version__ as psychopy_version
from psychopy import data, gui, logging
from psychopy import core, event, visual
import explorepy
import utils


# Base functions

def list2dict(lst, start=0, step=1):
    return {item: start + step * i for i, item in enumerate(lst)} 

# Dictionairy functions

def print_dct(dct, indent=0):
   for key, value in dct.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


# JSON functions

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# File functions

def get_current_script_name():
    return os.path.splitext(os.path.basename(__file__))[0]

def copy_current_script(path):
    shutil.copy(__file__, path)

def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Example
    -------
    >>> get_all_files('samples/rir_samples', match_and=['3.wav'])
    ['samples/rir_samples/rir3.wav']
    """

    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles

def fixation(fix_type, win, pos, color, size, opacity, **kwargs):
    if fix_type == 'cross':
        return visual.TextStim(win, text='+', pos=pos, color=color, height=size, opacity=opacity, **kwargs)
    elif fix_type == 'dot':
        return visual.Circle(win, radius=size, pos=pos, fillColor=color, lineColor=color, opacity=opacity, **kwargs)
    else:
        raise ValueError("Invalid fixation type. Use 'cross' or 'dot'.")

def textstim(win, text, pos, color, height, **kwargs):
    return visual.TextStim(win, text=text, pos=pos, color=color, height=height, **kwargs)

def check_escape(win, keys, explorer=None):
    if 'escape' in event.getKeys():
        if explorer:
            explorer.stop_recording()
            explorer.disconnect()
        win.close()
        core.quit()

def shuffle_trials_per_block(tasks, n_reps_per_task, random_type):
    task_per_block = [tasks * n_reps_per_task]
    if random_type == 'full':
        task_per_block = [tasks * n_reps_per_task]
        np.random.shuffle(task_per_block)
    if random_type == 'sequential':
        task_per_block = []
        for _ in range(n_reps_per_task):
            np.random.shuffle(tasks)
            task_per_block.extend(tasks)
    return task_per_block

def main():
    task_descriptions = {
        'left_hand': 'Imagine moving your LEFT hand',
        'right_hand': 'Imagine moving your RIGHT hand'
    }

    exp_name = "imaginary_movement"
    exp_version = "v1"

    stimuli = ['no_stim'] 
    n_reps_per_stimulus = 10
    trial_ids = np.repeat(np.arange(len(stimuli)), n_reps_per_stimulus)

    trial_cues = ['wait_before_trial', 'instruct', 'prepare', 'perform', 'wait_after_trial']
    block_cues = ['wait_before_rest', 'rest', 'wait_after_rest', 'wait_before_block', 'block', 'wait_after_block']
    cues = trial_cues + block_cues
    trial_durations = [0.5, 1.0, 2.0, 4.0, 0.5]
    block_durations = [None, 30, 0, None, None, 0]
    cue_duration_dct = {k: v for k, v in zip(trial_cues, trial_durations)}
    cue_duration_dct.update({k: v for k, v in zip(block_cues, block_durations)})

    tasks = ['left_hand', 'right_hand']
    shuffle_type = 'sequential'
    n_reps_per_task = 10
    task_per_block = shuffle_trials_per_block(tasks, n_reps_per_task, shuffle_type)

    stimulus_task_combinations = ['/'.join(tup) for tup in list(itertools.product(stimuli, tasks))]
    max_marker_value = 99
    assert len(stimulus_task_combinations) < max_marker_value - len(cues)
    label_dct = utils.list2dict(stimulus_task_combinations, 1)
    cue_dct = utils.list2dict(cues, max_marker_value, -1)
    marker_dct = {**label_dct, **cue_dct}
    print(label_dct)

    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    data_rootdir = os.path.join(os.getcwd(), "data")
    data_dir = os.path.join(data_rootdir, exp_name, exp_version)
    os.makedirs(data_dir, exist_ok=True)

    shutil.copy(__file__, data_dir)

    exp_info = {
        'exp_name': exp_name,
        'exp_version': exp_version,
        'stimuli': stimuli,
        'n_reps_per_stimulus': n_reps_per_stimulus,
        'cue_duration_dct': cue_duration_dct,
        'task_per_block': task_per_block,
        'marker_dct': marker_dct,
        'data_dir': data_dir,
        'psychopy_version': psychopy_version
    }

    sub_info = {
        'participant': '001',
        'session': '001',
        'data': data_dir
    }

    dlg = gui.DlgFromDict(dictionary=sub_info, sortKeys=False, title="Subject info")
    if dlg.OK == False:
        core.quit()
    sub_info['date'] = data.getDateStr().replace('_', '-')

    sub_fname = f'{sub_info["participant"]}{sub_info["session"]}{sub_info["date"]}'
    sub_fpath = os.path.join(data_dir, sub_fname)
    sub_info['fname'] = sub_fname

    log = logging.LogFile(sub_fpath + '.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)

    explorer, eeg_info = None, {}
    doConnectExplorer = True
    if doConnectExplorer:
        eeg_fname = os.path.join(data_dir, sub_info['fname'])
        eeg_device_name = "Explore_849D"
        eeg_sr = 250
        channel_dct = {
            0: 'TP9',
            1: 'Fz', 2: 'C3', 3: 'Cz', 4: 'C4', 5: 'P3', 6: 'P4', 7: 'AFz', 8: 'TP10'
        }

        dlg = gui.DlgFromDict(dictionary=channel_dct, sortKeys=False, title="EEG channel info")
        if dlg.OK == False:
            core.quit()

        explorer = explorepy.Explore()
        explorer.connect(device_name=eeg_device_name)
        explorer.set_sampling_rate(eeg_sr)
        explorer.record_data(eeg_fname, do_overwrite=True)

        eeg_info = {
            'device_type': 'mentalab',
            'device_name': eeg_device_name,
            'samplerate': eeg_sr,
            'channel_dct': channel_dct
        }

    win_size = (1920, 1080)
    win_color = 'black'
    win = visual.Window(
        size=win_size, fullscr=True, screen=1, allowGUI=True, allowStencil=False,
        units='pix', monitor='testMonitor', colorSpace=u'rgb', color=win_color
    )

    fix_pos = (0, 0)
    fix_type = 'cross'
    fix_color = ('grey', 'grey')
    fix_opacity = (1, 0.5)
    fix_size = 10
    kwargs = {}

    wait_stim = fixation(fix_type, win, fix_pos, color=fix_color[0], size=fix_size, opacity=fix_opacity[0], **kwargs)
    action_stim = fixation(fix_type, win, fix_pos, color=fix_color[1], size=fix_size, opacity=fix_opacity[1], **kwargs)

    rest_stim = wait_stim
    prepare_stim = wait_stim
    perform_stim = action_stim

    text_color = 'grey'
    text_height = 32

    rest_text = 'Rest\nPress button to start'
    rest_textstim = textstim(win, rest_text.upper(), fix_pos, color=text_color, height=text_height)

    task_texts = [f'{task}\n{task_descriptions[task]}\n\nPress button to start' for task in tasks]
    block_textstims = {task: textstim(win, text.upper(), fix_pos, color=text_color, height=text_height)
                       for task, text in zip(tasks, task_texts)}

    instruct_stims = [textstim(win, stim.upper(), fix_pos, color=text_color, height=text_height) for stim in stimuli]

    meta_data = {'experiment': exp_info, 'subject': sub_info, 'eeg': eeg_info}
    with open(f'{sub_fpath}.json', 'w') as f:
        json.dump(meta_data, f, cls=utils.NumpyArrayEncoder)

    keys = event.BuilderKeyResponse()
    keys_continue = ['space']

    globalClock = core.Clock()
    routineTimer = core.CountdownTimer()
    event.Mouse(visible=False)

    print(meta_data)

    for i, task in enumerate(task_per_block):
        marker = marker_dct['wait_before_rest']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')

        rest_textstim.draw()
        win.flip()

        marker = marker_dct['rest']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)

        rest_stim.draw()
        event.waitKeys(keyList=keys_continue)
        win.flip()

        routineTimer.add(cue_duration_dct['rest'])
        while routineTimer.getTime() > 0:
            check_escape(win, keys, explorer)

        marker = marker_dct['wait_after_rest']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)
        win.flip()

        routineTimer.add(cue_duration_dct['wait_after_rest'])
        while routineTimer.getTime() > 0:
            check_escape(win, keys, explorer)

        marker = marker_dct['wait_before_block']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')

        block_textstims[task].draw()
        win.flip()

        marker = marker_dct['block']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)

        np.random.shuffle(trial_ids)
        event.waitKeys(keyList=keys_continue)
        win.flip()

        for trial_id in trial_ids:

            marker = marker_dct['wait_before_trial']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            wait_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['wait_before_trial'])
            while routineTimer.getTime() > 0:
                check_escape(win, keys, explorer)

            marker = int(trial_id)
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            trial_stim = instruct_stims[trial_id]
            trial_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['instruct'])
            while routineTimer.getTime() > 0:
                check_escape(win, keys, explorer)

            marker = marker_dct['prepare']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            prepare_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['prepare'])
            while routineTimer.getTime() > 0:
                check_escape(win, keys, explorer)

            # UPDATE bob
            marker = marker_dct['perform']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            perform_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['perform'])
            while routineTimer.getTime() > 0:
                check_escape(win, keys, explorer)

            marker = marker_dct['wait_after_trial']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            wait_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['wait_after_trial'])
            while routineTimer.getTime() > 0:
                check_escape(win, keys, explorer)

        marker = marker_dct['wait_after_block']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)

        win.flip()

        routineTimer.add(cue_duration_dct['wait_after_block'])
        while routineTimer.getTime() > 0:
            check_escape(win, keys, explorer)

    if doConnectExplorer:
        explorer.stop_recording()
        explorer.disconnect()
    win.close()
    core.quit()

if __name__ == "__main__":
    main()
