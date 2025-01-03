import argparse




def add_hparams2parser(hparams_dict):
    parser = argparse.ArgumentParser()

    # all arguments
    for key, val in hparams_dict.items():
        if type(val) is bool:
            parser.add_argument(f'--{key}',
                            action="store_true",
                            default=val)
        else:
            parser.add_argument(f'--{key}',
                            type=type(val),
                            default=val)
            
    args = parser.parse_args()

    return args


