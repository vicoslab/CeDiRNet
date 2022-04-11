from criterions.center_direction_loss import CenterDirectionLoss

def get_criterion(type, loss_opts, model, center_model):

    if type == 'CenterDirectionLoss':
        criterion = CenterDirectionLoss(center_model, **loss_opts)
    else:
        raise Exception("Unknown 'loss_type' in config: only allowed 'CenterDirectionLoss'")

    return criterion