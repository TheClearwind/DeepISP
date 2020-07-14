from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)

if __name__ == '__main__':
    logger = Logger("./runs")
