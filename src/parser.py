import argparse
from pathlib import Path
import json
from train import get_default, get_model, pre_trained
from utils import Utils
from torch import device, cuda
from model import ColorizationModel

OPT : argparse.ArgumentParser

parser = argparse.ArgumentParser(
    description = 'Specify parameters from cmndline.'
)

parser.add_argument(
    '--predict',
    nargs=1,
    type=str,
    default=None,
    help= 'Make a prediction given a source image' \
         +'Will use --pretrained if no other checkpoint provided.'
)

parser.add_argument(
    '--cpu',
    action='store_true',
    help='Override default. Default is gpu if available.'
)

parser.add_argument(
    '--pretrained',
    action='store_true',
    help='Use the pretrained model.'
)

parser.add_argument(
    '--lfc',
    nargs=1,
    type=str,
    default=None,
    help='Load from check point. Provide path.'
)

parser.add_argument(
    '--mkjson',
    action='store_true',
    help='make json param file'

)

parser.add_argument(
    '--load',
    nargs=1,
    type = str,
    default=None,
    help='Load parameters from param.json. Does not override other provided settings.'
)
# parse
def process() -> dict:
    root = Path(__file__).parent.absolute()
    global OPT

    if OPT.mkjson:
        out = get_default()
        with open(root / 'param.json', 'w') as f:
            f.write(json.dumps(out, indent=2))

    if OPT.load is not None:
        fname = root / OPT.load
        if not fname.exists():
            raise FileNotFoundError(f'{fname} not found use option --mkjson to dump default settings')
        with open(fname) as f:
            param = json.loads(f)
    else:
        param = get_default()

    if OPT.cpu or not cuda.is_available():
        param['device'] = device('cpu')
        param['trainer']['gpus'] = 0
    else:
        param['device'] = cuda.current_device()

    ## Init model
    param['model'] = get_model(param['model'], pretrained=OPT.pretrained)
    
    if OPT.lfc:
        pass # load from checkpoint not implemented

    if OPT.predict is not None:
        fname = Path(OPT.predict)
        if not fname.exists():
            raise FileNotFoundError(f'{fname} not found.')
        else:
            model : ColorizationModel
            model = param['model']
            # load from checkpoint not implented yet
            if not OPT.pretrained:
                model = pre_trained(model)
            
            im = Utils.load_im(fname)
            L, _ = Utils.preprocess_im(im)
            ab = model.predict(L)
            im = Utils.postprocess_tens(L, ab)
            out_path = fname.parent / f'{fname.stem}_out{fname.suffix}'
            Utils.save_im(out_path, im)
    return param


OPT = parser.parse_args()