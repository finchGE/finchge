import configparser
import ast

def load_config(config_file='config.ini'):
    config = _read_config_file(config_file)
    return config.get('ge', None)

def _read_config_file(config_file):
    ''' Reads and parses the config file.
    :param config_file: (str) the path of a .ini file.
    :return cfg: (dict) the dictionary of information in config_file.
    '''
    def _safe_eval(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def _build_dict(items):
        return {item[0]: _safe_eval(item[1]) for item in items}

    cf = configparser.ConfigParser()
    cf.read(config_file)
    cfg = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return cfg