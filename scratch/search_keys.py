import torch
path = r"c:\Experiments\BEBLaDII\storage\experiments\20260414 Phase 1 Reasoning failed start\BEST_MODEL.pt"
try:
    ckpt = torch.load(path, map_location='cpu')
    print(f"Total keys: {len(ckpt)}")
    student_keys = [k for k in ckpt.keys() if "student" in k]
    print(f"Student-related keys: {len(student_keys)}")
    if student_keys:
        print(f"  First 5 student keys: {student_keys[:5]}")
    
    proj_keys = [k for k in ckpt.keys() if "projector" in k]
    print(f"Projector-related keys: {len(proj_keys)}")
    if proj_keys:
        print(f"  First 5 projector keys: {proj_keys[:5]}")
        
except Exception as e:
    print(f"FAIL: {e}")
