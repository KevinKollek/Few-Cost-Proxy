# Helper function: renaming of keys
def change_keys(original_dict):
    """
    Renames certain keys to more readable labels.
    """
    old_keys = [
        'zico', 'grad_norm', 'large_final', 'magnitude_increase', 'movement',
        'synflow', 'snip', 'grasp', 'l2_norm', 'jacov', 'plain', 'fisher',
        'nwot', 'zen', 'epe_nas', 'val_acc', 'loss', 'train_acc',
        'params', 'flop', 'absmaginc'
    ]
    new_keys = [
        'ZiCo', 'Grad-Norm', 'LargeFinal', 'Mag.Inc.', 'Movement',
        'SynFlow', 'Snip', 'GraSP', 'L2-Norm', 'Jacov', 'Plain', 'Fisher',
        'NWOT', 'Zen', 'EPE-NAS', 'Val.Acc.', 'Loss', 'Train.Acc.',
        'Parameter', 'Flops', 'Abs.Mag.Inc.'
    ]
    mapping = dict(zip(old_keys, new_keys))

    new_dict = {}
    for key, value in original_dict.items():
        if key in mapping:
            new_dict[mapping[key]] = value
        else:
            new_dict[key] = value
    return new_dict
