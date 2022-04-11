import importlib
import argparse, ast

dataset_to_import = {'synt-center-learn-weakly':'synthetic.{}_center_learn_weakly',
                     'sorghum': 'sorghum.{}',
                     'carpk': 'CARPK.{}',
                     'pucpr+': 'PUCPRplus.{}',
                     'acacia_06': 'acacia_06.{}',
                     'acacia_12': 'acacia_12.{}',
                     'oilpalm': 'oilpalm.{}',
                     }


def get_config_args(dataset, type, merge_from_cmd_args=True):
    dataset = dataset.lower()

    if dataset not in dataset_to_import.keys():
        raise Exception('Unknown or missing dataset value')

    if type not in ['train','test']:
        raise Exception('Invalid type of arguments request: supported only train or test')

    config_module = 'config.' + dataset_to_import[dataset].format(type)

    module = importlib.import_module(config_module)

    print('Loading config for dataset=%s and type=%s' % (dataset, type))
    args = module.get_args()

    ########################################################
    # Merge from CMD args if any
    if merge_from_cmd_args:
        class ParseConfigCMDArgs(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, dict())
                for value in values:
                    value_eq = value.split('=')
                    key, value = value_eq[0], value_eq[1:]
                    getattr(namespace, self.dest)[key] = "=".join(value)

        def set_config_val_recursive(config, k, v):
            k0 = k[0]
            if isinstance(config, list):
                k0 = int(k0)
            if isinstance(k, list) and len(k) > 1:
                config[k0] = set_config_val_recursive(config[k0], k[1:], v)
            else:
                config[k0] = v
            return config
        # get any config values from CMD arg that override the config file ones
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--configs', nargs='*', action=ParseConfigCMDArgs, default=dict())

        cmd_args = parser.parse_args()

        for k,v in cmd_args.configs.items():
            try:
                v = ast.literal_eval(v)
            except:
                print('WARNING: cfg %s=%s interpreting as string' % (k,v))
            args = set_config_val_recursive(args, k.split("."), v)
            print("Overriding config with cmd %s=%s" % (k,v))

    ########################################################
    # updated any string with format substitution based on other arg values (only on first level)
    for k in sorted(args.keys()):
        if isinstance(args[k],str):
            args[k] = args[k].format(args=args)

    return args