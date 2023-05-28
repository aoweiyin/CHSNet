import os
import logging
from datetime import datetime
from utils.logger import setlogger


class Tester(object):
    def __init__(self, args):
        """ get save_dir && set logger && save args """
        # sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        sub_dir = datetime.strftime(datetime.now(), '%m%d')  # prepare saving path
        self.save_dir = os.path.join(args.save_dir, sub_dir+'_'+args.tag)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'test.log'))  # set logger
        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def test(self):
        """training one epoch"""
        pass
