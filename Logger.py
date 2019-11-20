import sys


class Logger(object):
    """
    This class enables logging to both stdout and text file
    """
    def __init__(self, logfile_name, terminal=sys.stdout):
        self.terminal = terminal
        self.log = open(logfile_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
