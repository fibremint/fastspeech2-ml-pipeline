# import argparse
# import json
# import os

# from app.preprocess import vctk


# def _parse_args():
#     default_cleaners = ['english_cleaners']

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-base-path', type=str, default='', required=True)
#     parser.add_argument('--data-input-path', type=str, default='./raw-data')
#     parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
#     parser.add_argument('--data-output-path', type=str, default='./before-align')
#     parser.add_argument('--sample-rate', type=int, default=22050)
#     parser.add_argument('--max-wav-value', type=float, default=32767.0)
#     # ref: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
#     # https://stackoverflow.com/questions/43660172/why-does-argparse-include-default-value-for-argsional-argument-even-when-argument
#     parser.add_argument('-c', '--cleaners', action='append', type=str)

#     args = parser.parse_args()
#     args.cleaners = args.cleaners if args.cleaners else default_cleaners

#     return args


def prepare_align(current_data_path: str) -> None:
    import argparse
    import json
    import os

    from app.preprocess import vctk


    def _parse_args():
        default_cleaners = ['english_cleaners']

        parser = argparse.ArgumentParser()
        # parser.add_argument('--data-base-path', type=str, default='', required=True)
        parser.add_argument('--data-input-path', type=str, default='./raw-data')
        parser.add_argument('--fs2-data-relpaths-filename', type=str, default='data-relpaths.json')
        parser.add_argument('--data-output-path', type=str, default='./before-align')
        parser.add_argument('--sample-rate', type=int, default=22050)
        parser.add_argument('--max-wav-value', type=float, default=32767.0)
        # ref: https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
        # https://stackoverflow.com/questions/43660172/why-does-argparse-include-default-value-for-argsional-argument-even-when-argument
        parser.add_argument('-c', '--cleaners', action='append', type=str)

        args = parser.parse_args()
        args.cleaners = args.cleaners if args.cleaners else default_cleaners

        return args


    args  = _parse_args()

    input_path = os.path.join(current_data_path, args.data_input_path)
    with open(f'{current_data_path}/{args.fs2_data_relpaths_filename}', 'r') as f:
        data_relpaths = json.load(f)

    output_path = os.path.join(args.data_base_path, args.data_output_path)

    vctk.prepare_align(data_relpaths=data_relpaths,
                       input_path=input_path,
                       out_dir=output_path,
                       sample_rate=args.sample_rate,
                       max_wav_value=args.max_wav_value,
                       cleaners=args.cleaners)
