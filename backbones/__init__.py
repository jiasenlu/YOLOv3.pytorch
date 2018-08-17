from . import darknet
from . import resnet
import pdb

def backbone_fn(opt):

	if opt.backbones_network == "resnet50":
		model = resnet(opt, _num_layers=50, _fixed_block=1, pretrained=True)
	elif opt.backbones_network == "resnet101":
		model = resnet(opt, _num_layers=101, _fixed_block=1, pretrained=True)
	elif opt.backbones_network == "darknet21":
		model = darknet.darknet21(opt.weightfile)
	elif opt.backbones_network == "darknet53":
		model = darknet.darknet53(opt.weightfile)
	else:
		pdb.set_trace()
	return model