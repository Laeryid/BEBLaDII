import torch
path = r"c:\Experiments\BEBLaDII\kaggle_upload_1_2\AWAKENED_WEIGHTS_FINAL.pt"
print(f"Inspecting {path}...")
try:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print(f"Total keys: {len(keys)}")
        print("First 10 keys:")
        for k in keys[:10]:
            print(f"  {k}")
        
        # Check for nested state_dict
        for sub in ["state_dict", "latentBERT_state_dict"]:
            if sub in ckpt:
                print(f"\nFound nested: {sub}")
                sub_keys = list(ckpt[sub].keys())
                print(f"Sub-keys: {len(sub_keys)}")
                for k in sub_keys[:10]:
                    print(f"    {k}")
    else:
        print(f"Loaded object type: {type(ckpt)}")
except Exception as e:
    print(f"Error: {e}")
