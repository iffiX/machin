from utils import colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s[%(asctime)s] <%(levelname)s>:%(name)s:%(message)s"))
default_logger = colorlog.getLogger("default_logger")
default_logger.addHandler(handler)
default_logger.setLevel(colorlog.INFO)
