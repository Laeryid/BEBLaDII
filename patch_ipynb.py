import json
import os

f = 'C:\\Experiments\\BEBLaDII\\experiments\\phase3\\kaggle\\train_phase3_notebook.ipynb'
d = json.load(open(f, encoding='utf-8'))

for cell in d['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    
    # Check if this is the main training cell
    if any('wandb.login(key=wandb_api)' in s for s in src):
        i_wandb = [i for i, s in enumerate(src) if 'wandb.login(key=wandb_api)' in s][0]
        src.insert(i_wandb+2, '            \n')
        src.insert(i_wandb+3, '            # GCP Auth\n')
        src.insert(i_wandb+4, '            try:\n')
        src.insert(i_wandb+5, '                gcp_sa = user_secrets.get_secret("GCP_SA_JSON")\n')
        src.insert(i_wandb+6, '                with open("gcp_sa.json", "w") as f:\n')
        src.insert(i_wandb+7, '                    f.write(gcp_sa)\n')
        src.insert(i_wandb+8, '                subprocess.run(["gcloud", "auth", "activate-service-account", "--key-file", "gcp_sa.json"], check=True)\n')
        src.insert(i_wandb+9, '                print("[Init] GCP Authentication successful via Kaggle Secrets.")\n')
        src.insert(i_wandb+10, '            except Exception as e_gcp:\n')
        src.insert(i_wandb+11, '                print(f"[Init] WARN: Could not authenticate GCP automatically: {e_gcp}")\n')
        src.insert(i_wandb+12, '                \n')

        i_import = [i for i, s in enumerate(src) if 'from kaggle_secrets import UserSecretsClient' in s][0]
        src.insert(i_import+1, '            import json, os, subprocess\n')

        i_ckpt = [i for i, s in enumerate(src) if '[SAVE] Checkpoint saved' in s][0]
        src.insert(i_ckpt+1, '                try:\n')
        src.insert(i_ckpt+2, '                    import subprocess\n')
        src.insert(i_ckpt+3, '                    subprocess.Popen(["gsutil", "-q", "cp", ckpt_path, "gs://bebladii-weigths-us/planB/phase3/checkpoints/"])\n')
        src.insert(i_ckpt+4, '                    print(f"[SYNC] Started background sync to GCS: {ckpt_path}\\n")\n')
        src.insert(i_ckpt+5, '                except Exception as e:\n')
        src.insert(i_ckpt+6, '                    print(f"[SYNC] Error syncing to GCS: {e}\\n")\n')

        i_final = [i for i, s in enumerate(src) if '[SAVE] Final weights saved' in s][0]
        src.insert(i_final+1, '    try:\n')
        src.insert(i_final+2, '        import subprocess\n')
        src.insert(i_final+3, '        subprocess.run(["gsutil", "-q", "cp", final_path, "gs://bebladii-weigths-us/planB/phase3/checkpoints/"])\n')
        src.insert(i_final+4, '        print(f"[SYNC] Final sync to GCS complete: {final_path}\\n")\n')
        src.insert(i_final+5, '    except Exception as e:\n')
        src.insert(i_final+6, '        print(f"[SYNC] Error syncing final to GCS: {e}\\n")\n')

        cell['source'] = src
        break

json.dump(d, open(f, 'w', encoding='utf-8'), indent=1)
