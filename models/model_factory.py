# ====================
# Model Factory Method
# ====================

# Import available models
from models.ewc import TBN_EWC, TSN_EWC
from models.lwf import TBN_LwF, TSN_LwF
from models.icarl import TBN_iCaRL, TSN_iCaRL
from models.cmr_mfn import CMR_MFN
from models.replay import TBN_Replay
from models.der import TBN_DER, TSN_DER
from models.foster import TBN_FOSTER, TSN_FOSTER
from models.mand import TBN_MAND, TSN_MAND
from models.upperbound import TBN_UpperBound

def get_model(model_name, args):
    """Return model instance by name."""

    name = model_name.lower()

    # Dictionary mapping model names to their classes
    model_dict = {
        "tbn_ewc": TBN_EWC,
        "tsn_ewc": TSN_EWC,
        "tbn_lwf": TBN_LwF,
        "tsn_lwf": TSN_LwF,
        "tbn_icarl": TBN_iCaRL,
        "tsn_icarl": TSN_iCaRL,
        'cmr_mfn': CMR_MFN,
        'cmr_mfn_r': CMR_MFN,
        'tbn_replay': TBN_Replay,
        'tbn_der': TBN_DER,
        'tsn_der': TSN_DER,
        'tbn_foster': TBN_FOSTER,
        'tsn_foster': TSN_FOSTER,
        'tbn_mand': TBN_MAND,
        'tsn_mand': TSN_MAND,
        'tbn_mand_wo_morst': TBN_MAND,
        'tbn_upperbound': TBN_UpperBound,
    }

    # Instantiate and return the model if name is valid
    if name in model_dict:
        return model_dict[name](args)  # Create and return model instance
    else:
        raise ValueError(f"Unknown model name: {model_name}")
