import logging


def get_logger(name, filename):
    logging.root.setLevel(logging.INFO)
    log_format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s'

    filehandler = logging.FileHandler(filename, mode='w')        
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger(name).addHandler(filehandler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger(name).addHandler(console)
    
    return logging.getLogger(name)