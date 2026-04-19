import torch
path = r"c:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt"
print(f"Deep inspection of {path}...")
try:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        for root_key in ckpt.keys():
            val = ckpt[root_key]
            if isinstance(val, dict):
                print(f"\nRoot Key: {root_key} (dict, {len(val)} items)")
                keys = list(val.keys())
                for k in keys[:5]:
                    print(f"  {k}")
                # Check for nested dicts in feature_projectors
                if root_key == "feature_projectors":
                    for sub_key in val.keys():
                        sub_val = val[sub_key]
                        if isinstance(sub_val, dict):
                            print(f"    Sub-key {sub_key}: dict, {len(sub_val)} items")
                            for sk in list(sub_val.keys())[:3]:
                                print(f"      {sk}")
            else:
                print(f"\nRoot Key: {root_key} ({type(val)})")
    else:
        print(f"Loaded object type: {type(ckpt)}")
except Exception as e:
    print(f"Error: {e}")
