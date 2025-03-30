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
import utils_psychopy as utils_psychopy

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
    """
    Experimental paradigm:
        Trial structure: ['wait_before_trial', 'instruct', 'prepare', 'perform', 'wait_after_trial']
        Block structure: ['wait_before_rest', 'rest', 'wait_after_rest', 'wait_before_block', 'block', 'wait_after_block']
    """

    #task descriptions
    task_descriptions = {
        'left_hand': 'Imagine moving your LEFT hand',
        'right_hand': 'Imagine moving your RIGHT hand'
    }


    exp_name = "imaginary_movement"
    exp_version = "v1"

    stimuli = ['left_hand', 'right_hand']
    n_reps_per_stimulus = 5
    trial_ids = np.repeat(np.arange(len(stimuli)), n_reps_per_stimulus)

    trial_cues = ['wait_before_trial', 'instruct', 'prepare', 'perform', 'wait_after_trial']
    block_cues = ['wait_before_rest', 'rest', 'wait_after_rest', 'wait_before_block', 'block', 'wait_after_block']
    cues = trial_cues + block_cues
    trial_durations = [5.0, 2.5, 2.0, 6.0, 1.0]
    block_durations = [None, 10, 0, None, None, 0]
    cue_duration_dct = {k: v for k, v in zip(trial_cues, trial_durations)}
    cue_duration_dct.update({k: v for k, v in zip(block_cues, block_durations)})


    tasks = ['left_hand', 'right_hand']
    shuffle_type = 'sequential'
    n_reps_per_task = 5 
    task_per_block = shuffle_trials_per_block(tasks, n_reps_per_task, shuffle_type)

    stimulus_task_combinations = ['/'.join(tup) for tup in list(itertools.product(stimuli, tasks))]
    max_marker_value = 99
    assert len(stimulus_task_combinations) < max_marker_value - len(cues)
    label_dct = utils.list2dict(stimulus_task_combinations)
    cue_dct = utils.list2dict(cues, max_marker_value, -1)
    marker_dct = {**label_dct, **cue_dct}


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

    sub_fname = f'{sub_info["participant"]}_{sub_info["session"]}_{sub_info["date"]}'
    sub_fpath = os.path.join(data_dir, sub_fname)
    sub_info['fname'] = sub_fname

    log = logging.LogFile(sub_fpath + '.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)

    explorer, eeg_info = None, {}
    doConnectExplorer = True  # (set to true when EEG headset is there)
    if doConnectExplorer:
        eeg_fname = os.path.join(data_dir, sub_info['fname'])
        eeg_device_name = "Explore_849D"
        eeg_sr = 250
        channel_dct = {
           '0': 'Cz', '3': 'C3', '4': 'C4'
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

    # setup PsychoPy
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

    wait_stim = utils_psychopy.fixation(fix_type, win, fix_pos, color=fix_color[0], size=fix_size, opacity=fix_opacity[0], **kwargs)
    action_stim = utils_psychopy.fixation(fix_type, win, fix_pos, color=fix_color[1], size=fix_size, opacity=fix_opacity[1], **kwargs)

    rest_stim = wait_stim
    prepare_stim = wait_stim
    perform_stim = action_stim

    text_color = 'grey'
    text_height = 32
    rest_text = 'Rest\nPress button to start'
    rest_textstim = utils_psychopy.textstim(win, rest_text.upper(), fix_pos, color=text_color, height=text_height)

    task_texts = [f'{task}\n{task_descriptions[task]}\n\nPress button to start' for task in tasks]
    block_textstims = {task: utils_psychopy.textstim(win, text.upper(), fix_pos, color=text_color, height=text_height)
                       for task, text in zip(tasks, task_texts)}

    instruct_stims = [utils_psychopy.textstim(win, stim.upper(), fix_pos, color=text_color, height=text_height) for stim in stimuli]

    # start experiment
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
        ### REST ###
        # Wait before rest (until button press)
        marker = marker_dct['wait_before_rest']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')

        rest_textstim.draw()
        win.flip()

        # Rest (after button press)
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
            utils_psychopy.check_escape(win, keys, explorer)

        # Wait after rest (until button press)
        marker = marker_dct['wait_after_rest']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)
        win.flip()

        routineTimer.add(cue_duration_dct['wait_after_rest'])
        while routineTimer.getTime() > 0:
            utils_psychopy.check_escape(win, keys, explorer)

        ### BLOCK ###
        # Wait before block (until button press)
        marker = marker_dct['wait_before_block']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')

        block_textstims[task].draw()
        win.flip()

        # Block (after button press)
        marker = marker_dct['block']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)

        np.random.shuffle(trial_ids)
        event.waitKeys(keyList=keys_continue)
        win.flip()

        ### TRIAL ###
        for trial_id in trial_ids:
            # Wait before trial
            marker = marker_dct['wait_before_trial']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            wait_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['wait_before_trial'])
            while routineTimer.getTime() > 0:
                utils_psychopy.check_escape(win, keys, explorer)


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
                utils_psychopy.check_escape(win, keys, explorer)

            # Prepare
            marker = marker_dct['prepare']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            prepare_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['prepare'])
            while routineTimer.getTime() > 0:
                utils_psychopy.check_escape(win, keys, explorer)

            # Perform
            marker = marker_dct['perform']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            perform_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['perform'])
            while routineTimer.getTime() > 0:
                utils_psychopy.check_escape(win, keys, explorer)

            # Wait after trial
            marker = marker_dct['wait_after_trial']
            if doConnectExplorer:
                win.callOnFlip(explorer.set_marker, marker)
            win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
            win.callOnFlip(routineTimer.reset)

            wait_stim.draw()
            win.flip()

            routineTimer.add(cue_duration_dct['wait_after_trial'])
            while routineTimer.getTime() > 0:
                utils_psychopy.check_escape(win, keys, explorer)

        # Wait after block
        marker = marker_dct['wait_after_block']
        if doConnectExplorer:
            win.callOnFlip(explorer.set_marker, marker)
        win.logOnFlip(level=logging.EXP, msg=f'MARKER: {marker}')
        win.callOnFlip(routineTimer.reset)

        win.flip()

        routineTimer.add(cue_duration_dct['wait_after_block'])
        while routineTimer.getTime() > 0:
            utils_psychopy.check_escape(win, keys, explorer)

    # Stop recording and disconnect
    if doConnectExplorer:
        explorer.stop_recording()
        explorer.disconnect()
    win.close()
    core.quit()

if __name__ == "__main__":
    main()