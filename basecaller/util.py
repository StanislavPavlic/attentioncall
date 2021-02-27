import argparse


def layers(params_str):
    try:
        return tuple(map(int, params_str.split(',')))

    except:
        raise argparse.ArgumentTypeError(
            "Layer arguments must be in form of out_channels, kernel_size, stride. Example : \'256,10,2\'")


def temperature_info(params_str):
    try:
        temp_start, temp_min, decay = map(float, params_str.split(','))
        return temp_start, temp_min, decay

    except:
        raise argparse.ArgumentTypeError(
            "Temperature information must be temperature_start, minimum_temperature, temperature_decay_factor")
