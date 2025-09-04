# models/backbones/__init__.py
from models.backbones.tbn import TBN
from models.backbones.tsn import TSN

def get_backbone(args):
    name = args["backbone"].lower()
    if name == "tbn":
        return TBN(
            num_segments=args["num_segments"],
            modality=args["modality"],
            base_model=args["arch"],
            new_length=args.get("new_length", None)
            )
    elif name == "tsn":
        return TSN(
            num_segments=args["num_segments"],
            modality=args["modality"],
            base_model=args["arch"],
            new_length=args.get("new_length", None)
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
