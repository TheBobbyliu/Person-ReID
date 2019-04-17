import torch
def load_state_dict(model, pretrain, args):
    state_dict = model.get_model().state_dict()
    new_state_dict = {}
    loaded_n = 0
    all_n = 0
    rollback_n = 0
    loaded_pretrain_name = []
    for k,v in pretrain.items():
        #if args.rollback:
        #    if 'fc_module' in k:
        #        rollback_n += 1
        #        continue
        if 'model.' in k:
            k = k.lstrip('model.')
        if k in state_dict.keys():
            if v.size() == state_dict[k].size():
                new_state_dict[k] = v
                loaded_pretrain_name.append(k)
                loaded_n += 1
        all_n += 1
    state_dict.update(new_state_dict)
    model.get_model().load_state_dict(state_dict)
    print('\nLoaded State Dict: \ntotal:{}\nloaded:{}\nrollback:{}'.format(all_n, loaded_n, rollback_n))
    base_params = []
    for name, param in model.get_model().named_parameters():
        if name in loaded_pretrain_name:
            base_params.append(param)
    print('base params: {}\n'.format(len(base_params)))
    return base_params
