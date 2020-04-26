def gen_learning_rate_func(lr_map, logger=None):
    def learning_rate_func(step):
        for i in range(len(lr_map) - 1):
            if lr_map[i][0] <= step < lr_map[i + 1][0]:
                if logger is not None:
                    logger.info("Current learning rate:{}".format(lr_map[i][1]))
                return lr_map[i][1]
        if logger is not None:
            logger.info("Current learning rate:{}".format(lr_map[-1][1]))
        return lr_map[-1][1]

    return learning_rate_func