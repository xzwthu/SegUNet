import logging

from .UNet import UNet
from .VNet import VNet

logger = logging.getLogger("weakseg")

key2net = {
    "unet": UNet,
    "vnet": VNet,
}


def get_network(cfg):
    if cfg["network"]["name"] is None:
        logger.info("Using UNet as segmentation structure")
        return UNet

    else:
        net_name = cfg["network"]["name"]
        if net_name not in key2net:
            raise NotImplementedError("network {} not implemented".format(net_name))

        logger.info("Using {} as network".format(net_name))
        return key2net[net_name]